from __future__ import annotations

import dataclasses
import types
import weakref
from collections.abc import Generator, Sequence
from typing import Annotated, Any, Optional, Union, get_origin, get_type_hints

from pydantic import BaseModel
from typing_extensions import NotRequired, ReadOnly, Required

from langgraph._internal._typing import MISSING


def _is_optional_type(type_: Any) -> bool:
    """타입이 Optional인지 확인합니다."""

    # 새로운 union 구문(PEP 604) 처리: str | None
    if isinstance(type_, types.UnionType):
        return any(
            arg is type(None) or _is_optional_type(arg) for arg in type_.__args__
        )

    if hasattr(type_, "__origin__") and hasattr(type_, "__args__"):
        origin = get_origin(type_)
        if origin is Optional:
            return True
        if origin is Union:
            return any(
                arg is type(None) or _is_optional_type(arg) for arg in type_.__args__
            )
        if origin is Annotated:
            return _is_optional_type(type_.__args__[0])
        return origin is None
    if hasattr(type_, "__bound__") and type_.__bound__ is not None:
        return _is_optional_type(type_.__bound__)
    return type_ is None


def _is_required_type(type_: Any) -> bool | None:
    """어노테이션이 Required/NotRequired로 표시되어 있는지 확인합니다.

    Returns:
        - 필수인 경우 True
        - 필수가 아닌 경우 False
        - 둘 다 어노테이션되지 않은 경우 None
    """
    origin = get_origin(type_)
    if origin is Required:
        return True
    if origin is NotRequired:
        return False
    if origin is Annotated or getattr(origin, "__args__", None):
        # 참고: https://typing.readthedocs.io/en/latest/spec/typeddict.html#interaction-with-annotated
        return _is_required_type(type_.__args__[0])
    return None


def _is_readonly_type(type_: Any) -> bool:
    """어노테이션이 ReadOnly로 표시되어 있는지 확인합니다.

    Returns:
        - 읽기 전용인 경우 True
        - 읽기 전용이 아닌 경우 False
    """

    # 참고: https://typing.readthedocs.io/en/latest/spec/typeddict.html#typing-readonly-type-qualifier
    origin = get_origin(type_)
    if origin is Annotated:
        return _is_readonly_type(type_.__args__[0])
    if origin is ReadOnly:
        return True
    return False


_DEFAULT_KEYS: frozenset[str] = frozenset()


def get_field_default(name: str, type_: Any, schema: type[Any]) -> Any:
    """상태 스키마의 필드에 대한 기본값을 결정합니다.

    다음을 기반으로 합니다:
        TypedDict인 경우:
            - Required/NotRequired
            - total=False -> 모든 것이 선택적
        - 타입 어노테이션 (Optional/Union[None])
    """
    optional_keys = getattr(schema, "__optional_keys__", _DEFAULT_KEYS)
    irq = _is_required_type(type_)
    if name in optional_keys:
        # total=False 또는 명시적 NotRequired 중 하나입니다.
        # 타입 어노테이션은 이것을 무시합니다.
        if irq:
            # 이전 버전의 Python 및 명시적 Required인 경우는 예외입니다
            return ...
        return None
    if irq is not None:
        if irq:
            # Required[<type>] 처리
            # (NotRequired 및 total=False는 이미 처리했습니다)
            return ...
        # 이전 버전의 Python에 대한 NotRequired[<type>] 처리
        return None
    if dataclasses.is_dataclass(schema):
        field_info = next(
            (f for f in dataclasses.fields(schema) if f.name == name), None
        )
        if field_info:
            if (
                field_info.default is not dataclasses.MISSING
                and field_info.default is not ...
            ):
                return field_info.default
            elif field_info.default_factory is not dataclasses.MISSING:
                return field_info.default_factory()
    # 참고: ReadOnly 속성은 무시합니다.
    # 의미가 없기 때문입니다. (노드에서 상태를 변경하는지 여부는 중요하지 않습니다)
    # 노드에서 상태를 변경해도 그래프 상태에는 영향을 미치지 않습니다.
    # 기본 케이스는 어노테이션입니다
    if _is_optional_type(type_):
        return None
    return ...


def get_enhanced_type_hints(
    type: type[Any],
) -> Generator[tuple[str, Any, Any, str | None], None, None]:
    """제공된 타입에서 기본값과 설명을 추출하려고 시도합니다. config 스키마에 사용됩니다."""
    for name, typ in get_type_hints(type).items():
        default = None
        description = None

        # Pydantic 모델
        try:
            if hasattr(type, "model_fields") and name in type.model_fields:
                field = type.model_fields[name]

                if hasattr(field, "description") and field.description is not None:
                    description = field.description

                if hasattr(field, "default") and field.default is not None:
                    default = field.default
                    if (
                        hasattr(default, "__class__")
                        and getattr(default.__class__, "__name__", "")
                        == "PydanticUndefinedType"
                    ):
                        default = None

        except (AttributeError, KeyError, TypeError):
            pass

        # TypedDict, dataclass
        try:
            if hasattr(type, "__dict__"):
                type_dict = getattr(type, "__dict__")

                if name in type_dict:
                    default = type_dict[name]
        except (AttributeError, KeyError, TypeError):
            pass

        yield name, typ, default, description


def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    """Pydantic 상태 업데이트를 (key, value) 튜플의 리스트로 가져옵니다."""
    if isinstance(input, BaseModel):
        keep = input.model_fields_set
        defaults = {k: v.default for k, v in type(input).model_fields.items()}
    else:
        keep = None
        defaults = {}

    # 참고: Pydantic에 대한 이 동작은 다소 우아하지 않지만,
    # 하위 호환성을 위해 유지합니다
    # input이 Pydantic 모델인 경우, 기본값과 다르거나
    # keep 집합에 있는 값만 업데이트합니다
    return [
        (k, value)
        for k in keys
        if (value := getattr(input, k, MISSING)) is not MISSING
        and (
            value is not None
            or defaults.get(k, MISSING) is not None
            or (keep is not None and k in keep)
        )
    ]


ANNOTATED_KEYS_CACHE: weakref.WeakKeyDictionary[type[Any], tuple[str, ...]] = (
    weakref.WeakKeyDictionary()
)


def get_cached_annotated_keys(obj: type[Any]) -> tuple[str, ...]:
    """Python 클래스의 캐시된 어노테이션 키를 반환합니다."""
    if obj in ANNOTATED_KEYS_CACHE:
        return ANNOTATED_KEYS_CACHE[obj]
    if isinstance(obj, type):
        keys: list[str] = []
        for base in reversed(obj.__mro__):
            ann = base.__dict__.get("__annotations__")
            # Python 3.14+에서 Pydantic 모델은 __annotations__에 디스크립터를 사용하므로
            # __dict__.get이 None을 반환하면 getattr로 대체해야 합니다
            if ann is None:
                ann = getattr(base, "__annotations__", None)
            if ann is None or isinstance(ann, types.GetSetDescriptorType):
                continue
            keys.extend(ann.keys())
        return ANNOTATED_KEYS_CACHE.setdefault(obj, tuple(keys))
    else:
        raise TypeError(f"Expected a type, got {type(obj)}. ")
