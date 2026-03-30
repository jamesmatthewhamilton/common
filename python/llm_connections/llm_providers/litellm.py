"""LiteLLM provider — OpenAI-compatible API via httpx. 100% offline, no telemetry."""

import json
import logging
import time

from .base import BaseProvider
from ..response import LLMResponse, LLMChunk

logger = logging.getLogger(__name__)


class LitellmProvider(BaseProvider):
    """Provider for OpenAI-compatible endpoints (LiteLLM, vLLM, TGI, etc.)."""

    MAX_RETRIES = 3

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip("/")
        if not self.base_url:
            raise ValueError("LiteLLM provider requires 'base_url' in config")
        self.api_key = config.get("api_key", "")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, messages: list, tools: list = None,
                       stream: bool = False, **opts) -> dict:
        payload = {
            "model": opts.pop("model", self.model),
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        # Map options to OpenAI format
        if "temperature" in opts:
            payload["temperature"] = opts["temperature"]
        if "max_tokens" in opts or "num_predict" in opts:
            payload["max_tokens"] = opts.get("max_tokens") or opts.get("num_predict")
        if "top_p" in opts:
            payload["top_p"] = opts["top_p"]
        return payload

    def chat(self, messages: list, tools: list = None,
             stream: bool = False, **overrides) -> LLMResponse:
        import httpx

        opts = self._merge_options(overrides)
        payload = self._build_payload(messages, tools, stream, **opts)
        url = f"{self.base_url}/chat/completions"

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if stream:
                    return self._stream_chat(url, payload)
                else:
                    return self._sync_chat(url, payload)
            except httpx.ConnectError as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(f"LiteLLM request failed (attempt {attempt + 1}): {e}. "
                                   f"Retrying in {wait}s...")
                    time.sleep(wait)

        raise ConnectionError(
            f"LiteLLM request failed after {self.MAX_RETRIES} attempts: {last_error}"
        )

    def _sync_chat(self, url: str, payload: dict) -> LLMResponse:
        import httpx

        with httpx.Client(timeout=300) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        text = message.get("content", "") or ""
        tool_calls = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "name": func.get("name", ""),
                "arguments": args,
            })

        done_reason = choice.get("finish_reason", "")
        if done_reason == "length":
            logger.warning("LLM output truncated — hit max_tokens limit.")

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            done_reason=done_reason,
            model=data.get("model", payload.get("model", "")),
        )

    def _stream_chat(self, url: str, payload: dict) -> LLMResponse:
        import httpx

        payload["stream"] = True
        response = LLMResponse(model=payload.get("model", self.model))

        def _iter():
            full_text = ""
            all_tool_calls = {}  # index -> {name, arguments_str}

            with httpx.Client(timeout=300) as client:
                with client.stream("POST", url, headers=self._headers(),
                                   json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})

                        chunk_text = delta.get("content", "") or ""
                        if chunk_text:
                            full_text += chunk_text

                        # Accumulate tool call deltas
                        chunk_tools = []
                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            if idx not in all_tool_calls:
                                all_tool_calls[idx] = {
                                    "name": "",
                                    "arguments_str": "",
                                }
                            func = tc_delta.get("function", {})
                            if "name" in func:
                                all_tool_calls[idx]["name"] = func["name"]
                            if "arguments" in func:
                                all_tool_calls[idx]["arguments_str"] += func["arguments"]

                        if chunk_text:
                            yield LLMChunk(text=chunk_text)

                        # Check for finish
                        finish = choice.get("finish_reason")
                        if finish:
                            response.done_reason = finish

                        # Capture usage if present (sent in final chunk)
                        usage = data.get("usage")
                        if usage:
                            response.prompt_tokens = usage.get("prompt_tokens", 0)
                            response.completion_tokens = usage.get("completion_tokens", 0)

            # If server didn't send usage, estimate from content
            if not response.prompt_tokens:
                # Estimate: count chars in all messages / 4
                total_chars = sum(len(json.dumps(m)) for m in payload.get("messages", []))
                response.prompt_tokens = total_chars // 4
                response.completion_tokens = len(full_text) // 4

            # Finalize tool calls
            final_tools = []
            for idx in sorted(all_tool_calls.keys()):
                tc = all_tool_calls[idx]
                args_str = tc["arguments_str"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}
                final_tools.append({
                    "name": tc["name"],
                    "arguments": args,
                })

            response.text = full_text
            response.tool_calls = final_tools

            if response.done_reason == "length":
                logger.warning("LLM output truncated — hit max_tokens limit.")

        response._stream_iter = _iter()
        return response
