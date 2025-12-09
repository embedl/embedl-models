# Copyright (C) 2025 Embedl AB

"""Creates an interactive session with vLLM."""

import asyncio
import time
import uuid
from typing import Optional
import argparse

from embedl.models.vllm import AsyncLLM

from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind


async def _stream_once(
    engine: AsyncLLM, prompt: str, sampling_params: SamplingParams
) -> str:
    request_id = f"repl-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
    print("Assistant: ", end="", flush=True)

    full_text_parts: list[str] = []
    async for output in engine.generate(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params,
    ):
        for completion in output.outputs:
            new_text = completion.text  # DELTA => only newly generated tokens
            if new_text:
                full_text_parts.append(new_text)
                print(new_text, end="", flush=True)
        if output.finished:
            break

    print()  # newline after response
    return "".join(full_text_parts)


def _make_engine(model: str, *, max_model_len: int = 28592) -> AsyncLLM:
    return AsyncLLM(model, max_model_len=max_model_len)


def _make_sampling_params(
    *,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: Optional[int] = 42,
    stop: Optional[list[str]] = None,
) -> SamplingParams:
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        output_kind=RequestOutputKind.DELTA,
        stop=stop or ["\nUser:", "\nYou:"],
    )


async def run_repl(
    *,
    model: str,
    max_model_len: int = 28592,
    system: str = "",
    sampling_params: Optional[SamplingParams] = None,
) -> None:
    """
    Run an interactive streaming REPL.

    Exposes the prior script functionality as a single importable coroutine.
    Maintains a minimal in-memory history in "User: ... / Assistant: ..." format.

    Commands
    --------
    /exit or /quit : quit the REPL
    /reset         : clear chat history

    :param model: Model name or local filesystem path.
    :param max_model_len: Max context length passed to AsyncLLM.
    :param system: Optional system prefix placed before the chat history.
    :param sampling_params: Optional SamplingParams override.
    """
    engine = _make_engine(model, max_model_len=max_model_len)
    sp = sampling_params or _make_sampling_params()

    print("Interactive AsyncLLM streaming REPL")
    print("Type /exit to quit, /reset to clear chat history.\n")

    history: list[str] = []

    try:
        while True:
            user = await asyncio.to_thread(input, "You: ")
            user = user.strip()
            if not user:
                continue

            if user.lower() in {"/exit", "/quit"}:
                break

            if user.lower() == "/reset":
                history.clear()
                print("(history cleared)\n")
                continue

            history.append(f"User: {user}\nAssistant:")
            full_prompt = system + "\n".join(history)

            assistant_reply = await _stream_once(engine, full_prompt, sp)

            history[-1] = f"User: {user}\nAssistant: {assistant_reply}"
            print()

    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlashHead vLLM interactive chat"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="embedl/Llama-3.2-3B-Instruct-FlashHead-W4A16",
        help="Model name or local path (e.g., embedl/Llama-3.2-3B-Instruct-FlashHead-W4A16",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=28592,
        help="Max model context length",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="",
        help="Optional system message prefix before the chat history.",
    )

    args = parser.parse_args()

    asyncio.run(
        run_repl(
            model=args.model,
            max_model_len=args.max_model_len,
            system=args.system,
        )
    )
