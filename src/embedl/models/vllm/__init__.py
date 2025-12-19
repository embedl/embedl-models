# Copyright (C) 2025 Embedl AB

"""vLLM integration for FlashHead."""

import importlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig

logger = logging.getLogger(__name__)

import importlib

import torch
from vllm import LLM as _LLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as _AsyncLLM

from embedl.models import DEVICE
from embedl.models.flash_head import FlashHead, get_flash_head_parameters


def _patch_vllm_module(target_path, replacement_module):
    spec = importlib.util.find_spec(replacement_module)
    if spec is None:
        raise ImportError(
            f"Could not find replacement module: {replacement_module}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[target_path] = module


_patch_vllm_module(
    "vllm.v1.sample.sampler", "embedl.models.vllm.patching.sampler"
)
_patch_vllm_module(
    "vllm.model_executor.layers.logits_processor",
    "embedl.models.vllm.patching.logits_processor",
)


def _get_flash_head() -> nn.Module:
    path = "/tmp/embedl_flash_head.pt"
    if not os.path.exists(path):
        return None
    flash_head = torch.load(
        "/tmp/embedl_flash_head.pt", map_location=DEVICE, weights_only=False
    )
    return flash_head


def _set_flash_head(new_flash_head):
    path = "/tmp/embedl_flash_head.pt"
    if new_flash_head:
        torch.save(new_flash_head, path)
    else:
        if os.path.exists(path):
            os.remove(path)


def _update_config(path: str):
    """
    Update the config.json to remove fields that are FlashHead specific.

    We patch vLLM to assume that the model is a standard model that vLLM can
    load, e.g., Llama, Gemma. The model config is updated to protect from
    accidentally running it with the standard vLLM or HuggingFace pipelines.
    The patching logic in `embedl.models.vllm` will inject FlashHead-specific
    generation and the standard vLLM inference will fail.
    """
    config_path = os.path.join(path, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Update architectures
    if "architectures" in config:
        for arch in config["architectures"]:
            if "FlashHead" in arch:
                config["architectures"] = [arch.replace("FlashHead", "")]

    # Update model_type
    if "model_type" in config:
        if "flash_head_" in config["model_type"]:
            config["model_type"] = config["model_type"].replace(
                "flash_head_", ""
            )

    if "auto_map" in config:
        config.pop("auto_map")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def _create_and_update_model(
    model: str,
) -> str:
    """Create a local model and update its config.

    :param model:
        The model string or local path to a model.
    :return:
        The path to the local model with custom config that can be loaded by
        vLLM.
    """
    config = AutoConfig.from_pretrained(model)

    if not hasattr(config, "flash_head_cache_dir"):
        return model

    base_tmp = Path("/tmp/embedl_flash_head")

    logger.info("Preparing local model for '%s'", model)

    # Case 1: model is a local directory
    if os.path.isdir(model):
        base_name = Path(model).name
        local_path = base_tmp / base_name

        if local_path.exists():
            shutil.rmtree(local_path)

        logger.info("Copying local model from %s to %s", model, local_path)
        shutil.copytree(model, local_path)

    # Case 2: model is a Hugging Face repo ID
    else:
        local_path = base_tmp / model.replace("/", "__")

        if local_path.exists():
            logger.info(
                "Model snapshot already exists at %s, skipping download",
                local_path,
            )
        else:
            logger.info("Downloading model '%s' to %s", model, local_path)
            snapshot_download(
                repo_id=model,
                local_dir=local_path,
                local_dir_use_symlinks=False,
            )

    # Always update the config
    logger.info("Updating config.json in %s", local_path)
    _update_config(local_path)

    logger.info("Model ready at %s", local_path)
    return str(local_path)


LM_HEAD_KEYS = [
    "lm_head.weight",
    "model.lm_head.weight",
    "transformer.lm_head.weight",
    "model.embed_tokens.weight",  # tied embedding fallback
]


def _is_local_dir(path: str) -> bool:
    return os.path.isdir(path)


def _resolve_file(model: str, filename: str) -> Optional[str]:
    if _is_local_dir(model):
        p = os.path.join(model, filename)
        return p if os.path.exists(p) else None
    try:
        return hf_hub_download(repo_id=model, filename=filename)
    except Exception:
        return None


def _repo_has_file(model: str, filename: str) -> bool:
    if _is_local_dir(model):
        return os.path.exists(os.path.join(model, filename))
    try:
        files = list_repo_files(model)
        return filename in files
    except Exception:
        return False


def _find_weight_key_in_index(index_json: dict) -> Optional[str]:
    weight_map = index_json.get("weight_map", {})
    for k in LM_HEAD_KEYS:
        if k in weight_map:
            return k
    return None


def _load_lm_head_weight(model: str) -> Tuple[torch.Tensor, str]:
    if _repo_has_file(model, "model.safetensors.index.json"):
        index_path = _resolve_file(model, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        chosen = _find_weight_key_in_index(index)
        if chosen is None:
            raise KeyError(
                f"No lm_head/tied embedding key found in index. "
                f"Looked for: {LM_HEAD_KEYS}"
            )

        shard_name = index["weight_map"][chosen]
        shard_path = _resolve_file(model, shard_name)
        if shard_path is None:
            raise FileNotFoundError(f"Could not resolve shard: {shard_name}")

        # This loads the whole shard, but NOT the whole model.
        sd = load_file(shard_path)
        if chosen not in sd:
            raise KeyError(
                f"Expected {chosen} in {shard_name}, but not found."
            )
        return sd[chosen].cpu(), chosen

    st_path = _resolve_file(model, "model.safetensors")
    if st_path is not None:
        sd = load_file(st_path)
        for k in LM_HEAD_KEYS:
            if k in sd:
                return sd[k].cpu(), k
        raise KeyError(
            f"Could not find lm_head/tied embedding weight in model.safetensors. "
            f"Looked for: {LM_HEAD_KEYS}"
        )

    raise FileNotFoundError(
        f"No supported weight files found for {model}. "
        f"Expected model.safetensors(.index.json) or pytorch_model.bin(.index.json)."
    )


def _load_flash_head_from_checkpoint(
    model: str, dtype=torch.bfloat16, device=DEVICE
):
    config = AutoConfig.from_pretrained(model)

    if not hasattr(config, "flash_head_cache_dir"):
        return None

    cache_dir = config.flash_head_cache_dir
    if _is_local_dir(model) and not os.path.isabs(cache_dir):
        cache_dir = os.path.join(model, cache_dir)

    vocab_size = getattr(config, "vocab_size")
    hidden_size = getattr(config, "hidden_size")

    w, chosen_key = _load_lm_head_weight(model)
    if w.shape != (vocab_size, hidden_size):
        if w.shape == (hidden_size, vocab_size):
            w = w.t().contiguous()
        else:
            raise ValueError(
                f"Unexpected lm_head weight shape {tuple(w.shape)}; "
                f"expected {(vocab_size, hidden_size)} or {(hidden_size, vocab_size)}"
            )

    dummy_lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    dummy_lm_head.weight.data.copy_(w)

    flash_head = FlashHead(
        dummy_lm_head,
        **get_flash_head_parameters(
            dummy_lm_head,
            cache_dir=cache_dir,
            model_or_dir=model,
        ),
        special_token_ids=config.flash_head_special_token_ids,
    ).to(device=device, dtype=dtype)

    print(
        f"[Embedl] FlashHead initialized using '{chosen_key}' and cache {cache_dir}"
    )
    return flash_head


class LLM(_LLM):
    """
    vLLM LLM with FlashHead preloaded.

    This class wraps vLLM's synchronous LLM and ensures FlashHead is loaded and
    registered before model initialization. It also defaults GPU memory
    utilization to 0.75 so FlashHead fits comfortably.

    :param model:
        The model id or local path.
    :param gpu_memory_utilization:
        GPU memory utilization, defaults to 0.75.
    :param args:
        Positional args forwarded to vLLM LLM.
    :param kwargs:
        Keyword args forwarded to vLLM LLM.
    """

    def __init__(
        self, model: str, gpu_memory_utilization: float = 0.75, *args, **kwargs
    ):
        model = _create_and_update_model(model)
        flash_head = _load_flash_head_from_checkpoint(model)
        _set_flash_head(flash_head)

        # Default to 0.75 unless caller overrides
        super().__init__(
            model=model,
            *args,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )


class AsyncLLM(_AsyncLLM):
    """
    vLLM AsyncLLM with FlashHead preloaded.

    This class wraps vLLM's AsyncLLM and ensures FlashHead is loaded and
    registered before engine creation. It also defaults GPU memory utilization
    to 0.75 so FlashHead fits comfortably.

    vLLM async engines are created via `from_engine_args`, so this class
    implements `__new__` and returns an instance produced by vLLM.

    :param model:
        The model id or local path.
    :param gpu_memory_utilization:
        GPU memory utilization, defaults to 0.75.
    :param kwargs:
        Keyword args forwarded into AsyncEngineArgs.
    """

    def __new__(
        cls, model: str, gpu_memory_utilization: float = 0.75, **kwargs
    ):
        model = _create_and_update_model(model)
        flash_head = _load_flash_head_from_checkpoint(model)
        _set_flash_head(flash_head)

        engine_args = AsyncEngineArgs(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs,
        )
        engine = _AsyncLLM.from_engine_args(engine_args)
        return engine
