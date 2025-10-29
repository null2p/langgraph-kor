"""LangGraph를 위한 비공개 타이핑 유틸리티입니다."""

from __future__ import annotations

from dataclasses import Field
from typing import Any, ClassVar, Protocol, TypeAlias

from pydantic import BaseModel
from typing_extensions import TypedDict


class TypedDictLikeV1(Protocol):
    """TypedDict처럼 동작하는 타입을 나타내는 프로토콜입니다.

    버전 1: 키에 `ClassVar`를 사용합니다."""

    __required_keys__: ClassVar[frozenset[str]]
    __optional_keys__: ClassVar[frozenset[str]]


class TypedDictLikeV2(Protocol):
    """TypedDict처럼 동작하는 타입을 나타내는 프로토콜입니다.

    버전 2: 키에 `ClassVar`를 사용하지 않습니다."""

    __required_keys__: frozenset[str]
    __optional_keys__: frozenset[str]


class DataclassLike(Protocol):
    """dataclass처럼 동작하는 타입을 나타내는 프로토콜입니다.

    dataclasses의 비공개 _DataclassT에서 영감을 받아 유사한 프로토콜을 바운드로 사용합니다."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


StateLike: TypeAlias = TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel
"""상태와 유사한 타입에 대한 타입 별칭입니다.

`TypedDict`, `dataclass` 또는 Pydantic `BaseModel`일 수 있습니다.
참고: 타입 검사의 제한으로 인해 `TypedDict` 또는 `dataclass`를 직접 사용할 수 없습니다.
"""

MISSING = object()
"""설정되지 않은 센티널 값입니다."""


class DeprecatedKwargs(TypedDict):
    """추가 키워드 인자에 사용할 TypedDict로, 지원 중단된 인자에 대한 타입 검사 경고를 활성화합니다."""


EMPTY_SEQ: tuple[str, ...] = tuple()
"""빈 문자열 시퀀스입니다."""
