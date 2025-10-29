from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def _freeze(obj: Any, depth: int = 10) -> Hashable:
    if isinstance(obj, Hashable) or depth <= 0:
        # 이미 해시 가능하므로 동결할 필요 없음
        return obj
    elif isinstance(obj, Mapping):
        # 키를 정렬하여 {"a":1,"b":2} == {"b":2,"a":1}가 되도록 함
        return tuple(sorted((k, _freeze(v, depth - 1)) for k, v in obj.items()))
    elif isinstance(obj, Sequence):
        return tuple(_freeze(x, depth - 1) for x in obj)
    # numpy / pandas 등은 자체 .tobytes()를 제공할 수 있음
    elif hasattr(obj, "tobytes"):
        return (
            type(obj).__name__,
            obj.tobytes(),
            obj.shape if hasattr(obj, "shape") else None,
        )
    return obj  # 문자열, 정수, frozen=True인 dataclass 등


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """인자와 키워드 인자를 사용하여 해시 가능한 키를 생성하는 기본 캐시 키 함수입니다."""
    import pickle

    # 프로토콜 5는 속도와 크기 사이의 좋은 균형을 제공함
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
