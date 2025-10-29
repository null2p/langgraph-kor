"""SSE 사양에 따라 \n, \r, \r\n으로 줄을 분할하기 위해 httpx_sse에서 수정되었습니다."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import orjson

from langgraph_sdk.schema import StreamPart

BytesLike = bytes | bytearray | memoryview


class BytesLineDecoder:
    """
    텍스트에서 줄을 증분적으로 읽기를 처리합니다.

    stdllib bytes splitlines과 동일한 동작을 하지만,
    입력을 반복적으로 처리합니다.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.trailing_cr: bool = False

    def decode(self, text: bytes) -> list[BytesLike]:
        # https://docs.python.org/3/glossary.html#term-universal-newlines 참조
        NEWLINE_CHARS = b"\n\r"

        # 후행 `\r`을 항상 다음 디코드 반복으로 밀어넣습니다.
        if self.trailing_cr:
            text = b"\r" + text
            self.trailing_cr = False
        if text.endswith(b"\r"):
            self.trailing_cr = True
            text = text[:-1]

        if not text:
            # 참고: 빈 텍스트 입력의 엣지 케이스는 실제로 발생하지 않습니다.
            # 다른 httpx 내부에서 이 값을 필터링하기 때문입니다.
            return []  # pragma: no cover

        trailing_newline = text[-1] in NEWLINE_CHARS
        lines = text.splitlines()

        if len(lines) == 1 and not trailing_newline:
            # 새 줄이 없으면 입력을 버퍼링하고 계속합니다.
            self.buffer.extend(lines[0])
            return []

        if self.buffer:
            # splitlines 결과의 첫 번째 부분에 기존 버퍼를 포함합니다.
            self.buffer.extend(lines[0])
            lines = [self.buffer] + lines[1:]
            self.buffer = bytearray()

        if not trailing_newline:
            # splitlines의 마지막 세그먼트가 개행으로 종료되지 않은 경우,
            # 출력에서 제거하고 새 버퍼를 시작합니다.
            self.buffer.extend(lines.pop())

        return lines

    def flush(self) -> list[BytesLike]:
        if not self.buffer and not self.trailing_cr:
            return []

        lines = [self.buffer]
        self.buffer = bytearray()
        self.trailing_cr = False
        return lines


class SSEDecoder:
    def __init__(self) -> None:
        self._event = ""
        self._data = bytearray()
        self._last_event_id = ""
        self._retry: int | None = None

    @property
    def last_event_id(self) -> str | None:
        """마지막으로 본 이벤트 식별자를 반환합니다."""

        return self._last_event_id or None

    def decode(self, line: bytes) -> StreamPart | None:
        # 참조: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation  # noqa: E501

        if not line:
            if (
                not self._event
                and not self._data
                and not self._last_event_id
                and self._retry is None
            ):
                return None

            sse = StreamPart(
                event=self._event,
                data=orjson.loads(self._data) if self._data else None,  # type: ignore[invalid-argument-type]
            )

            # 참고: SSE 사양에 따라 last_event_id를 재설정하지 않습니다.
            self._event = ""
            self._data = bytearray()
            self._retry = None

            return sse

        if line.startswith(b":"):
            return None

        fieldname, _, value = line.partition(b":")

        if value.startswith(b" "):
            value = value[1:]

        if fieldname == b"event":
            self._event = value.decode()
        elif fieldname == b"data":
            self._data.extend(value)
        elif fieldname == b"id":
            if b"\0" in value:
                pass
            else:
                self._last_event_id = value.decode()
        elif fieldname == b"retry":
            try:
                self._retry = int(value)
            except (TypeError, ValueError):
                pass
        else:
            pass  # 필드는 무시됩니다.

        return None


async def aiter_lines_raw(response: httpx.Response) -> AsyncIterator[BytesLike]:
    decoder = BytesLineDecoder()
    async for chunk in response.aiter_bytes():
        for line in decoder.decode(chunk):
            yield line
    for line in decoder.flush():
        yield line


def iter_lines_raw(response: httpx.Response) -> Iterator[BytesLike]:
    decoder = BytesLineDecoder()
    for chunk in response.iter_bytes():
        for line in decoder.decode(chunk):
            yield line
    for line in decoder.flush():
        yield line
