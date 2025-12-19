# Copyright (C) 2025 Embedl AB

"""Embedl models package."""

import os
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ERROR_MSG = """
FlashHead uses a custom generation loop that is not yet supported by transformers.
Please use the patched vLLM generation as:

```
    from embedl.models.vllm import LLM
    from vllm import SamplingParams

    sampling = SamplingParams(max_tokens=128, temperature=0.0)
    llm = LLM(model=model_id, trust_remote_code=True, max_model_len=32768)

    prompt = "Write a haiku about coffee."
    output = llm.generate([prompt], sampling)
    print(output[0].outputs[0].text)

```
"""

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
