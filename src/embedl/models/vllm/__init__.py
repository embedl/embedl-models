# Copyright (C) 2025 Embedl AB

"""vLLM integration for FlashHead."""

import importlib
import json
import os
import sys
from typing import Optional, Tuple

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig

from embedl.models.flash_head import FlashHead, get_flash_head_parameters
from vllm import LLM as _LLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as _AsyncLLM


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
        "/tmp/embedl_flash_head.pt", map_location="cuda", weights_only=False
    )
    return flash_head


def _set_flash_head(new_flash_head):
    path = "/tmp/embedl_flash_head.pt"
    if new_flash_head:
        torch.save(new_flash_head, path)
    else:
        if os.path.exists(path):
            os.remove(path)


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
    model: str, dtype=torch.bfloat16, device="cuda"
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
            dummy_lm_head, cache_dir=cache_dir, model_or_dir=model
        ),
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

    :param model: The model id or local path.
    :param args: Positional args forwarded to vLLM LLM.
    :param kwargs: Keyword args forwarded to vLLM LLM (gpu_memory_utilization is
                   set to 0.75 to fit FlashHead unless explicitly provided).
    """

    def __init__(self, model: str, *args, **kwargs):
        flash_head = _load_flash_head_from_checkpoint(model)
        _set_flash_head(flash_head)

        # Default to 0.75 unless caller overrides
        kwargs.setdefault("gpu_memory_utilization", 0.75)

        super().__init__(model=model, *args, **kwargs)


class AsyncLLM(_AsyncLLM):
    """
    vLLM AsyncLLM with FlashHead preloaded.

    This class wraps vLLM's AsyncLLM and ensures FlashHead is loaded and
    registered before engine creation. It also defaults GPU memory utilization
    to 0.75 so FlashHead fits comfortably.

    vLLM async engines are created via `from_engine_args`, so this class
    implements `__new__` and returns an instance produced by vLLM.

    :param model: The model id or local path.
    :param kwargs: Keyword args forwarded into AsyncEngineArgs (gpu_memory_utilization
                   is set to 0.75 to fit FlashHead unless explicitly provided)
    """

    def __new__(cls, model: str, **kwargs):
        flash_head = _load_flash_head_from_checkpoint(model)
        _set_flash_head(flash_head)

        kwargs.setdefault("gpu_memory_utilization", 0.75)

        engine_args = AsyncEngineArgs(model=model, **kwargs)
        engine = _AsyncLLM.from_engine_args(engine_args)
        return engine