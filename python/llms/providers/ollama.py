"""Ollama provider — local LLM inference via Ollama."""

import logging
import time

from .base import BaseProvider
from ..response import LLMResponse, LLMChunk

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Provider for Ollama (local LLM server)."""

    MAX_RETRIES = 3

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")

    def chat(self, messages: list, tools: list = None,
             stream: bool = False, **overrides) -> LLMResponse:
        import ollama

        opts = self._merge_options(overrides)
        model = overrides.get("model", self.model)

        # Build Ollama options dict
        ollama_options = {}
        opt_map = {
            "temperature": "temperature",
            "num_ctx": "num_ctx",
            "num_predict": "num_predict",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
        }
        for key, ollama_key in opt_map.items():
            if key in opts:
                ollama_options[ollama_key] = opts[key]

        kwargs = {
            "model": model,
            "messages": messages,
            "options": ollama_options,
        }
        if tools:
            kwargs["tools"] = tools

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if stream:
                    return self._stream_chat(kwargs)
                else:
                    return self._sync_chat(kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Ollama request failed (attempt {attempt + 1}): {e}. "
                                   f"Retrying in {wait}s...")
                    time.sleep(wait)

        raise ConnectionError(f"Ollama request failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _sync_chat(self, kwargs: dict) -> LLMResponse:
        import ollama

        kwargs["stream"] = False
        data = ollama.chat(**kwargs)

        text = data.message.content or ""
        tool_calls = []
        if data.message.tool_calls:
            for tc in data.message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        prompt_tokens = getattr(data, "prompt_eval_count", 0) or 0
        completion_tokens = getattr(data, "eval_count", 0) or 0
        done_reason = getattr(data, "done_reason", "") or ""

        # Context window warning
        num_ctx = kwargs.get("options", {}).get("num_ctx")
        if num_ctx and prompt_tokens >= num_ctx * 0.95:
            status = "EXCEEDS" if prompt_tokens >= num_ctx else "near"
            logger.warning(
                f"Prompt is {prompt_tokens} tokens — {status} num_ctx limit ({num_ctx}). "
                f"Output quality may degrade. Increase num_ctx in config."
            )

        # Output truncation warning
        if done_reason == "length":
            num_predict = kwargs.get("options", {}).get("num_predict", "?")
            logger.warning(
                f"LLM output truncated — hit num_predict limit ({num_predict}). "
                f"Increase num_predict in config for longer responses."
            )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            done_reason=done_reason,
            model=kwargs["model"],
        )

    def _stream_chat(self, kwargs: dict) -> LLMResponse:
        import ollama

        kwargs["stream"] = True
        response = LLMResponse(model=kwargs["model"])

        def _iter():
            full_text = ""
            all_tool_calls = []

            stream = ollama.chat(**kwargs)
            for chunk in stream:
                chunk_text = ""
                chunk_tools = []

                if chunk.message.content:
                    chunk_text = chunk.message.content
                    full_text += chunk_text

                if chunk.message.tool_calls:
                    for tc in chunk.message.tool_calls:
                        tc_dict = {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                        all_tool_calls.append(tc_dict)
                        chunk_tools.append(tc_dict)

                if chunk_text or chunk_tools:
                    yield LLMChunk(text=chunk_text, tool_calls=chunk_tools)

            # After stream completes, update the response object
            response.text = full_text
            response.tool_calls = all_tool_calls
            response.prompt_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
            response.completion_tokens = getattr(chunk, "eval_count", 0) or 0
            response.done_reason = getattr(chunk, "done_reason", "") or ""

            # Context window warning
            num_ctx = kwargs.get("options", {}).get("num_ctx")
            if num_ctx and response.prompt_tokens >= num_ctx * 0.95:
                status = "EXCEEDS" if response.prompt_tokens >= num_ctx else "near"
                logger.warning(
                    f"Prompt is {response.prompt_tokens} tokens — {status} "
                    f"num_ctx limit ({num_ctx}). Increase num_ctx in config."
                )

            if response.done_reason == "length":
                logger.warning("LLM output truncated — hit num_predict limit.")

        response._stream_iter = _iter()
        return response
