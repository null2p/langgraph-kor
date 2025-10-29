"""임베딩 함수 및 LangChain의 Embeddings 인터페이스 작업을 위한 유틸리티입니다.

이 모듈은 임의의 임베딩 함수(동기 및 비동기 모두)를 LangChain의 Embeddings
인터페이스로 래핑하는 도구를 제공합니다. 이를 통해 동기 및 비동기 작업에 대한
지원을 유지하면서 LangChain 호환 도구와 함께 사용자 정의 임베딩 함수를 사용할 수 있습니다.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings

EmbeddingsFunc = Callable[[Sequence[str]], list[list[float]]]
"""동기 임베딩 함수의 타입입니다.

함수는 문자열 시퀀스를 받아 임베딩 목록을 반환해야 하며,
각 임베딩은 float 목록입니다. 임베딩의 차원은 모든 입력에 대해
일관되어야 합니다.
"""

AEmbeddingsFunc = Callable[[Sequence[str]], Awaitable[list[list[float]]]]
"""비동기 임베딩 함수의 타입입니다.

EmbeddingsFunc와 유사하지만 임베딩으로 해결되는 awaitable을 반환합니다.
"""


def ensure_embeddings(
    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str | None,
) -> Embeddings:
    """임베딩 함수가 LangChain의 Embeddings 인터페이스를 준수하도록 보장합니다.

    이 함수는 임의의 임베딩 함수를 래핑하여 LangChain의 Embeddings 인터페이스와
    호환되도록 만듭니다. 동기 및 비동기 함수 모두 처리합니다.

    Args:
        embed: 기존 Embeddings 인스턴스 또는 텍스트를 임베딩으로 변환하는 함수.
            함수가 비동기인 경우 동기 및 비동기 작업 모두에 사용됩니다.

    Returns:
        제공된 함수를 래핑하는 Embeddings 인스턴스.

    ??? example "예제"

        동기 임베딩 함수 래핑:

        ```python
        def my_embed_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = ensure_embeddings(my_embed_fn)
        result = embeddings.embed_query("hello")  # [0.1, 0.2] 반환
        ```

        비동기 임베딩 함수 래핑:

        ```python
        async def my_async_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = ensure_embeddings(my_async_fn)
        result = await embeddings.aembed_query("hello")  # [0.1, 0.2] 반환
        ```

        프로바이더 문자열을 사용하여 임베딩 초기화:

        ```python
        # langchain>=0.3.9 및 langgraph-checkpoint>=2.0.11 필요
        embeddings = ensure_embeddings("openai:text-embedding-3-small")
        result = embeddings.embed_query("hello")
        ```
    """
    if embed is None:
        raise ValueError("embed must be provided")
    if isinstance(embed, str):
        init_embeddings = _get_init_embeddings()
        if init_embeddings is None:
            from importlib.metadata import PackageNotFoundError, version

            try:
                lc_version = version("langchain")
                version_info = f"Found langchain version {lc_version}, but"
            except PackageNotFoundError:
                version_info = "langchain is not installed;"

            raise ValueError(
                f"Could not load embeddings from string '{embed}'. {version_info} "
                "loading embeddings by provider:identifier string requires langchain>=0.3.9 "
                "as well as the provider-specific package. "
                "Install LangChain with: pip install 'langchain>=0.3.9' "
                "and the provider-specific package (e.g., 'langchain-openai>=0.3.0'). "
                "Alternatively, specify 'embed' as a compatible Embeddings object or python function."
            )
        return init_embeddings(embed)

    if isinstance(embed, Embeddings):
        return embed
    return EmbeddingsLambda(embed)


class EmbeddingsLambda(Embeddings):
    """임베딩 함수를 LangChain의 Embeddings 인터페이스로 변환하는 래퍼입니다.

    이 클래스는 임의의 임베딩 함수를 LangChain 호환 도구와 함께 사용할 수 있도록 합니다.
    동기 및 비동기 작업을 모두 지원하며 다음을 처리할 수 있습니다:
    1. 동기 작업용 동기 함수 (비동기 작업은 동기 함수를 사용)
    2. 동기/비동기 작업 모두를 위한 비동기 함수 (동기 작업은 오류 발생)

    임베딩 함수는 텍스트의 의미를 포착하는 고정 차원 벡터로 텍스트를 변환해야 합니다.

    Args:
        func: 텍스트를 임베딩으로 변환하는 함수. 동기 또는 비동기 가능.
            비동기인 경우 비동기 작업에 사용되지만 동기 작업은 오류를 발생시킵니다.
            동기인 경우 동기 및 비동기 작업 모두에 사용됩니다.

    ??? example "예제"

        동기 함수 사용:

        ```python
        def my_embed_fn(texts):
            # 각 텍스트에 대한 2D 임베딩 반환
            return [[0.1, 0.2] for _ in texts]

        embeddings = EmbeddingsLambda(my_embed_fn)
        result = embeddings.embed_query("hello")  # [0.1, 0.2] 반환
        await embeddings.aembed_query("hello")  # 역시 [0.1, 0.2] 반환
        ```

        비동기 함수 사용:

        ```python
        async def my_async_fn(texts):
            return [[0.1, 0.2] for _ in texts]

        embeddings = EmbeddingsLambda(my_async_fn)
        await embeddings.aembed_query("hello")  # [0.1, 0.2] 반환
        # 참고: embed_query()는 오류를 발생시킴
        ```
    """

    def __init__(
        self,
        func: EmbeddingsFunc | AEmbeddingsFunc,
    ) -> None:
        if func is None:
            raise ValueError("func must be provided")
        if _is_async_callable(func):
            self.afunc = func
        else:
            self.func = func

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록을 벡터로 임베딩합니다.

        Args:
            texts: 임베딩으로 변환할 텍스트 목록.

        Returns:
            입력 텍스트당 하나씩, 임베딩 목록. 각 임베딩은 float 목록입니다.

        Raises:
            ValueError: 인스턴스가 비동기 함수로만 초기화된 경우.
        """
        func = getattr(self, "func", None)
        if func is None:
            raise ValueError(
                "EmbeddingsLambda was initialized with an async function but no sync function. "
                "Use aembed_documents for async operation or provide a sync function."
            )
        return func(texts)

    def embed_query(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩합니다.

        Args:
            text: 임베딩으로 변환할 텍스트.

        Returns:
            float 목록으로 된 임베딩 벡터.

        Note:
            이것은 단일 텍스트로 embed_documents를 호출하고
            첫 번째 결과를 가져오는 것과 동일합니다.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """텍스트 목록을 비동기적으로 벡터로 임베딩합니다.

        Args:
            texts: 임베딩으로 변환할 텍스트 목록.

        Returns:
            입력 텍스트당 하나씩, 임베딩 목록. 각 임베딩은 float 목록입니다.

        Note:
            비동기 함수가 제공되지 않은 경우 동기 구현으로 대체됩니다.
        """
        afunc = getattr(self, "afunc", None)
        if afunc is None:
            return await super().aembed_documents(texts)
        return await afunc(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """단일 텍스트를 비동기적으로 임베딩합니다.

        Args:
            text: 임베딩으로 변환할 텍스트.

        Returns:
            float 목록으로 된 임베딩 벡터.

        Note:
            이것은 단일 텍스트로 aembed_documents를 호출하고
            첫 번째 결과를 가져오는 것과 동일합니다.
        """
        afunc = getattr(self, "afunc", None)
        if afunc is None:
            return await super().aembed_query(text)
        return (await afunc([text]))[0]


def get_text_at_path(obj: Any, path: str | list[str]) -> list[str]:
    """경로 표현식 또는 사전 토큰화된 경로를 사용하여 객체에서 텍스트를 추출합니다.

    Args:
        obj: 텍스트를 추출할 객체
        path: 경로 문자열 또는 사전 토큰화된 경로 목록.

    !!! info "처리되는 경로 타입"
        - 단순 경로: "field1.field2"
        - 배열 인덱싱: "[0]", "[*]", "[-1]"
        - 와일드카드: "*"
        - 다중 필드 선택: "{field1,field2}"
        - 다중 필드의 중첩 경로: "{field1,nested.field2}"
    """
    if not path or path == "$":
        return [json.dumps(obj, sort_keys=True, ensure_ascii=False)]

    tokens = tokenize_path(path) if isinstance(path, str) else path

    def _extract_from_obj(obj: Any, tokens: list[str], pos: int) -> list[str]:
        if pos >= len(tokens):
            if isinstance(obj, (str, int, float, bool)):
                return [str(obj)]
            elif obj is None:
                return []
            elif isinstance(obj, (list, dict)):
                return [json.dumps(obj, sort_keys=True, ensure_ascii=False)]
            return []

        token = tokens[pos]
        results = []

        if token.startswith("[") and token.endswith("]"):
            if not isinstance(obj, list):
                return []

            index = token[1:-1]
            if index == "*":
                for item in obj:
                    results.extend(_extract_from_obj(item, tokens, pos + 1))
            else:
                try:
                    idx = int(index)
                    if idx < 0:
                        idx = len(obj) + idx
                    if 0 <= idx < len(obj):
                        results.extend(_extract_from_obj(obj[idx], tokens, pos + 1))
                except (ValueError, IndexError):
                    return []

        elif token.startswith("{") and token.endswith("}"):
            if not isinstance(obj, dict):
                return []

            fields = [f.strip() for f in token[1:-1].split(",")]
            for field in fields:
                nested_tokens = tokenize_path(field)
                if nested_tokens:
                    current_obj: dict | None = obj
                    for nested_token in nested_tokens:
                        if (
                            isinstance(current_obj, dict)
                            and nested_token in current_obj
                        ):
                            current_obj = current_obj[nested_token]
                        else:
                            current_obj = None
                            break
                    if current_obj is not None:
                        if isinstance(current_obj, (str, int, float, bool)):
                            results.append(str(current_obj))
                        elif isinstance(current_obj, (list, dict)):
                            results.append(
                                json.dumps(
                                    current_obj, sort_keys=True, ensure_ascii=False
                                )
                            )

        # 와일드카드 처리
        elif token == "*":
            if isinstance(obj, dict):
                for value in obj.values():
                    results.extend(_extract_from_obj(value, tokens, pos + 1))
            elif isinstance(obj, list):
                for item in obj:
                    results.extend(_extract_from_obj(item, tokens, pos + 1))

        # 일반 필드 처리
        else:
            if isinstance(obj, dict) and token in obj:
                results.extend(_extract_from_obj(obj[token], tokens, pos + 1))

        return results

    return _extract_from_obj(obj, tokens, 0)


# Private utility functions


def tokenize_path(path: str) -> list[str]:
    """경로를 구성 요소로 토큰화합니다.

    !!! info "처리되는 타입"
        - 단순 경로: "field1.field2"
        - 배열 인덱싱: "[0]", "[*]", "[-1]"
        - 와일드카드: "*"
        - 다중 필드 선택: "{field1,field2}"
    """
    if not path:
        return []

    tokens = []
    current: list[str] = []
    i = 0
    while i < len(path):
        char = path[i]

        if char == "[":  # 배열 인덱스 처리
            if current:
                tokens.append("".join(current))
                current = []
            bracket_count = 1
            index_chars = ["["]
            i += 1
            while i < len(path) and bracket_count > 0:
                if path[i] == "[":
                    bracket_count += 1
                elif path[i] == "]":
                    bracket_count -= 1
                index_chars.append(path[i])
                i += 1
            tokens.append("".join(index_chars))
            continue

        elif char == "{":  # 다중 필드 선택 처리
            if current:
                tokens.append("".join(current))
                current = []
            brace_count = 1
            field_chars = ["{"]
            i += 1
            while i < len(path) and brace_count > 0:
                if path[i] == "{":
                    brace_count += 1
                elif path[i] == "}":
                    brace_count -= 1
                field_chars.append(path[i])
                i += 1
            tokens.append("".join(field_chars))
            continue

        elif char == ".":  # 일반 필드 처리
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
        i += 1

    if current:
        tokens.append("".join(current))

    return tokens


def _is_async_callable(
    func: Any,
) -> bool:
    """함수가 비동기인지 확인합니다.

    이것은 async def 함수와 비동기 __call__ 메서드를 가진 클래스를 모두 포함합니다.

    Args:
        func: 확인할 함수 또는 호출 가능한 객체.

    Returns:
        함수가 비동기이면 True, 그렇지 않으면 False.
    """
    return (
        asyncio.iscoroutinefunction(func)
        or hasattr(func, "__call__")  # noqa: B004
        and asyncio.iscoroutinefunction(func.__call__)
    )


@functools.lru_cache
def _get_init_embeddings() -> Callable[[str], Embeddings] | None:
    try:
        from langchain.embeddings import init_embeddings  # type: ignore

        return init_embeddings
    except ImportError:
        return None


__all__ = [
    "ensure_embeddings",
    "EmbeddingsFunc",
    "AEmbeddingsFunc",
]
