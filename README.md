<!-- # Copyright (C) 2025 Embedl AB -->

# ⛔ DEPRECATED -- use [`flash-head`](https://github.com/embedl/flash-head) instead

> **This package is deprecated.** FlashHead is now available as a standalone vLLM plugin that activates automatically - no custom imports, no Docker images.
>
> ```bash
> pip install flash-head
> vllm serve embedl/Qwen3-1.7B-FlashHead-W4A16
> ```
>
> See the [flash-head repo](https://github.com/embedl/flash-head) for documentation and the [FlashHead model collection](https://huggingface.co/collections/embedl/flashhead) for available models.

---

## Migration

| Before (`embedl-models`)             | After (`flash-head`)         |
|--------------------------------------|------------------------------|
| `pip install embedl-models`          | `pip install flash-head`     |
| `from embedl.models.vllm import LLM` | `from vllm import LLM`       |
| Custom Docker images                 | Standard vLLM (`vllm serve`) |
| vLLM 0.10.2 pinned                   | vLLM >= 0.14.0               |

## License

Please check out the license file for details or contact legal@embedl.com

## Contact

models@embedl.com | [embedl.com](https://embedl.com)
