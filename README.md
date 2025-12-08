<!-- # Copyright (C) 2025 Embedl AB -->

# Embedl models

Optimized AI models for the edge by [Embedl](https://www.embedl.com/).

## Installation

The package provides optimized models for the edge and can be installed with
```
pip install embedl-models
```

## Models

## FlashHead - efficient langugage models head

Optimized versions of langugage models using FlashHead, Embedl’s efficient
replacement for the traditional language-model head. FlashHead drastically
reduces size while preserving accuracy, enabling H200-level latency on consumer
GPUs (RTX Ada generation).

This model is designed for low-latency inference on NVIDIA RTX GPUs, using a
combination of:
- FlashHead
- Mixed-precision quantization (W4A16)
- Custom vLLM generation

FlashHead produces outputs that match the baseline models within rounding error
on standard evaluations (MMLU-Pro, HellaSwag, GSM8K etc.). Overall, this model
is capable of yielding H200-level latency on consumer-grade GPUs (RTX Ada).

### Usage

The recommended way to use FlashHead is vLLM with the custom generation
pipeline provided in this package.

#### vLLM

```
from vllm import SamplingParams
from embedl.models.vllm import LLM

model_id = "embedl/Llama-3.2-1B-Instruct-FlashHead-W4A16"

sampling = SamplingParams(max_tokens=128, temperature=0.0)
llm = LLM(model=model_id, trust_remote_code=True)

prompt = "Write a haiku about coffee."
output = llm.generate([prompt], sampling)
print(output[0].outputs[0].text)
```

## Roadmap

The following improvements are planned for upcoming releases:

- HuggingFace transformers pipeline
- Benchmarking via vLLM CLI (for detailed latency benchmarking)
- Support for lm-eval-harness (for detailed accuracy measurements)
- Upstream support of models in transformers and vllm
- Compatibility with GGUF, MLC, Llama.cpp, TGI etc.
- Support for additional inference toolchains and devices

Interested in early access to new releases? Missing something?
Contact us (see below).

## License

Please check out the license file for details.

## Contact

For commercial licensing, enterprise support, or hardware co-marketing
opportunities:

Enterprise & Commercial Inquiries
 sales@embedl.com

Technical issues, feedback, and early-access requests
 https://github.com/embedl/embedl-models

More information about Embedl products, and model releases
 https://embedl.com

If you are evaluating on-device inference, building products on NVIDIA RTX/H200,
or exploring custom optimized models for your workloads, we encourage you to
reach out. We actively collaborate with teams integrating local AI into
commercial applications, and we offer:
- Tools for edge AI optimization, compatibility, profiling, provisioning of
hardware (Embedl SDK)
- Online platform for benchmarking models (Embedl HUB)
- Engineering support for on-prem and edge deployments of models
- Guidance on migration from baseline Llama/Qwen/Gemma to optimal models
- Priority access to upcoming model releases
- Partner co-marketing for qualified deployments


For early access to future models or custom porting/optimization work, contact
us directly at sales@embedl.com.
