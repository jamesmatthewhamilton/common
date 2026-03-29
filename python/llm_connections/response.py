"""LLM response types."""

from dataclasses import dataclass, field


@dataclass
class LLMChunk:
    """A single chunk from a streaming response."""
    text: str = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class LLMResponse:
    """Complete LLM response. Also iterable for streaming."""
    text: str = ""
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    done_reason: str = ""
    model: str = ""

    # Internal: for streaming support
    _stream_iter: object = field(default=None, repr=False)
    _finished: bool = field(default=False, repr=False)

    def __iter__(self):
        if self._stream_iter is None:
            return iter([])
        return self

    def __next__(self) -> LLMChunk:
        if self._stream_iter is None or self._finished:
            raise StopIteration
        try:
            return next(self._stream_iter)
        except StopIteration:
            self._finished = True
            raise
