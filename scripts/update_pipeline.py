# Copyright (C) 2025 Embedl AB

"""Script to update model definitions with custom classes FlashHead."""

from transformers import pipeline
from huggingface_hub import snapshot_download
from pathlib import Path
from embedl.models.vllm import LLM

import shutil
from huggingface_hub import snapshot_download
from pathlib import Path
import json
from transformers import AutoConfig
from vllm import SamplingParams
import os
from transformers import AutoTokenizer
from pathlib import Path
import shutil


def update_config_local(
    model_id: str,
    local_dir: str,
) -> None:
    """Temporary script to download model artifacts and update configs."""

    # Clear the local folder if it exists
    local_path = Path(local_dir)
    if local_path.exists() and local_path.is_dir():
        shutil.rmtree(local_path)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    config_path = Path(local_dir) / "config.json"
    clustering_config_path = (
        Path(local_dir) / "flash_head_assets/clustering_config.json"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "architectures" in config:
        model_class_name = config["architectures"][0]
        model_type = config["model_type"]
    else:
        raise ValueError("Model architectures not specified")

    config["architectures"] = ["FlashHead" + model_class_name]
    config["model_type"] = f"flash_head_{model_type}"

    original_config = AutoConfig.from_pretrained(model_id)
    config_class_name = type(original_config).__name__
    config["auto_map"] = {
        "AutoModelForCausalLM": f"modeling_flash_head_{model_type}.FlashHead{model_class_name}",
        "AutoConfig": f"configuration_flash_head_{model_type}.FlashHead{config_class_name}",
    }

    with open(clustering_config_path, "r", encoding="utf-8") as f:
        clustering_cfg = json.load(f)

    for key in [
        "n_clusters",
        "creation_time",
        "enforce_equal_cluster_sizes",
    ]:
        if key in clustering_cfg:
            config[key] = clustering_cfg[key]

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    if "llama" in model_type:
        model_subpackage = "llama"
    if "gemma" in model_type:
        model_subpackage = "gemma"
    if "qwen" in model_type:
        model_subpackage = "qwen"

    Path(f"{local_dir}/modeling_flash_head_{model_type}.py").write_text(
        f"from embedl.models.{model_subpackage}.modeling_flash_head "
        f"import FlashHead{model_class_name}\n"
    )

    Path(f"{local_dir}/configuration_flash_head_{model_type}.py").write_text(
        f"from embedl.models.{model_subpackage}.modeling_flash_head "
        f"import FlashHead{config_class_name}\n"
    )
    return str(os.path.abspath(local_dir))


def run_transformers_pipeline(model_id: str) -> None:
    """Run the transformers pipeline with the updated model."""

    hf_pipe = pipeline("text-generation", model_id, trust_remote_code=True)
    print(hf_pipe("Write a Haiku about coffee.", temperature=1.0))


def _qwen_pipeline(model_id: str):
    messages = [{"role": "user", "content": "Write a haiku about coffee."}]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    sampling = SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    llm = LLM(model=model_id, trust_remote_code=True, max_model_len=131072)
    output = llm.generate([text], sampling)
    print(output[0].outputs[0].text)


def _gemma_pipeline(model_id: str):
    sampling = SamplingParams(max_tokens=128, temperature=0.0)
    llm = LLM(model=model_id, trust_remote_code=True, max_model_len=32768)

    prompt = "Write a haiku about coffee."
    output = llm.generate([prompt], sampling)
    print(output[0].outputs[0].text)


def run_vllm_pipeline(model_id: str):
    """Run the vLLM pipeline with the updated model."""
    if "qwen" in model_id:
        _qwen_pipeline(model_id)
    if "gemma" in model_id:
        _gemma_pipeline(model_id)
    else:
        sampling = SamplingParams(max_tokens=128, temperature=0.0)
        llm = LLM(
            model=str(model_id), trust_remote_code=True, max_model_len=4096
        )
        prompt = "Write a haiku about coffee."
        output = llm.generate([prompt], sampling)
        print(output[0].outputs[0].text)

def test_vllm_pipeline_fails(model_id: str) -> None:
    """Test that the standard vLLM pipeline fails for FlashHead models.

    Args:
        model_id (str): A model id or local model.
    """
    from vllm import LLM
    llm = LLM(model=model_id, trust_remote_code=True, max_num_batched_tokens=131072)


ModelRegistry = {
    "llama-1b-quantized": {
        "embedl": "embedl/Llama-3.2-1B-Instruct-FlashHead-W4A16",
        "base": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "llama-1b": {
        "embedl": "embedl/Llama-3.2-1B-Instruct-FlashHead",
        "base": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "gemma-1b-quantized": {
        "embedl": "embedl/gemma-3-1b-it-FlashHead-W4A16",
        "base": "google/gemma-3-1b-it",
    },
    "gemma-1b": {
        "embedl": "embedl/gemma-3-1b-it-FlashHead",
        "base": "google/gemma-3-1b-it",
    },
    "qwen-1.7b": {
        "embedl": "embedl/Qwen3-1.7B-FlashHead-W4A16",
        "base": "Qwen/Qwen3-1.7B",
    },
    "qwen-1.7b-quantized": {
        "embedl": "embedl/Qwen3-1.7B-FlashHead",
        "base": "Qwen/Qwen3-1.7B",
    },
}


if __name__ == "__main__":
    for model_key in ModelRegistry:
        model = ModelRegistry[model_key]

        model_name = model["embedl"].split("/")[-1]
        local_model_id = update_config_local(
            model["embedl"],
            local_dir=f"./{model_name}",
        )
        # This should work
        print("Running vLLM model")
        run_vllm_pipeline(local_model_id)

        # This should fail with the local directory
        print("Running FlashHead model")
        run_transformers_pipeline(local_model_id)

        # This should fail with the local directory
        print("Running FlashHead model")
        test_vllm_pipeline_fails(local_model_id)
