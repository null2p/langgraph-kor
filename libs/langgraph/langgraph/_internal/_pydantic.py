from __future__ import annotations

import sys
import typing
import warnings
from contextlib import nullcontext
from dataclasses import is_dataclass
from functools import lru_cache
from typing import (
    Any,
    cast,
    overload,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)
from pydantic import (
    create_model as _create_model_base,
)
from pydantic.fields import FieldInfo
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
)
from typing_extensions import TypedDict


@overload
def get_fields(model: type[BaseModel]) -> dict[str, FieldInfo]: ...


@overload
def get_fields(model: BaseModel) -> dict[str, FieldInfo]: ...


def get_fields(
    model: type[BaseModel] | BaseModel,
) -> dict[str, FieldInfo]:
    """Pydantic 모델의 필드 이름을 가져옵니다."""
    if hasattr(model, "model_fields"):
        return model.model_fields

    if hasattr(model, "__fields__"):
        return model.__fields__
    msg = f"Expected a Pydantic model. Got {type(model)}"
    raise TypeError(msg)


_SchemaConfig = ConfigDict(
    arbitrary_types_allowed=True, frozen=True, protected_namespaces=()
)

NO_DEFAULT = object()


def _create_root_model(
    name: str,
    type_: Any,
    module_name: str | None = None,
    default_: object = NO_DEFAULT,
) -> type[BaseModel]:
    """베이스 클래스를 생성합니다."""

    def schema(
        cls: type[BaseModel],
        by_alias: bool = True,  # noqa: FBT001,FBT002
        ref_template: str = DEFAULT_REF_TEMPLATE,
    ) -> dict[str, Any]:
        # schema가 슈퍼클래스에 정의되지 않았다고 불평합니다
        schema_ = super(cls, cls).schema(  # type: ignore[misc]
            by_alias=by_alias, ref_template=ref_template
        )
        schema_["title"] = name
        return schema_

    def model_json_schema(
        cls: type[BaseModel],
        by_alias: bool = True,  # noqa: FBT001,FBT002
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
    ) -> dict[str, Any]:
        # model_json_schema가 슈퍼클래스에 정의되지 않았다고 불평합니다
        schema_ = super(cls, cls).model_json_schema(  # type: ignore[misc]
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
        )
        schema_["title"] = name
        return schema_

    base_class_attributes = {
        "__annotations__": {"root": type_},
        "model_config": ConfigDict(arbitrary_types_allowed=True),
        "schema": classmethod(schema),
        "model_json_schema": classmethod(model_json_schema),
        "__module__": module_name or "langchain_core.runnables.utils",
    }

    if default_ is not NO_DEFAULT:
        base_class_attributes["root"] = default_
    with warnings.catch_warnings():
        custom_root_type = type(name, (RootModel,), base_class_attributes)
    return cast("type[BaseModel]", custom_root_type)


@lru_cache(maxsize=256)
def _create_root_model_cached(
    model_name: str,
    type_: Any,
    *,
    module_name: str | None = None,
    default_: object = NO_DEFAULT,
) -> type[BaseModel]:
    return _create_root_model(
        model_name, type_, default_=default_, module_name=module_name
    )


@lru_cache(maxsize=256)
def _create_model_cached(
    model_name: str,
    /,
    **field_definitions: Any,
) -> type[BaseModel]:
    return _create_model_base(
        model_name,
        __config__=_SchemaConfig,
        **_remap_field_definitions(field_definitions),
    )


# 예약된 이름은 BaseModel에서 내부적으로 사용하는 모든 `public` 이름/메서드를 포함해야 합니다.
# 이렇게 하면 예약된 이름을 최신 상태로 유지할 수 있습니다.
# 참고로 예약된 이름은 다음과 같습니다:
# "construct", "copy", "dict", "from_orm", "json", "parse_file", "parse_obj",
# "parse_raw", "schema", "schema_json", "update_forward_refs", "validate",
# "model_computed_fields", "model_config", "model_construct", "model_copy",
# "model_dump", "model_dump_json", "model_extra", "model_fields",
# "model_fields_set", "model_json_schema", "model_parametrized_name",
# "model_post_init", "model_rebuild", "model_validate", "model_validate_json",
# "model_validate_strings"
_RESERVED_NAMES = {key for key in dir(BaseModel) if not key.startswith("_")}


def _remap_field_definitions(field_definitions: dict[str, Any]) -> dict[str, Any]:
    """내부 pydantic 필드와의 충돌을 피하기 위해 필드를 다시 매핑합니다."""

    remapped = {}
    for key, value in field_definitions.items():
        if key.startswith("_") or key in _RESERVED_NAMES:
            # 내부 pydantic 필드와의 충돌을 피하기 위해 접두사를 추가합니다
            if isinstance(value, FieldInfo):
                msg = (
                    f"Remapping for fields starting with '_' or fields with a name "
                    f"matching a reserved name {_RESERVED_NAMES} is not supported if "
                    f" the field is a pydantic Field instance. Got {key}."
                )
                raise NotImplementedError(msg)
            type_, default_ = value
            remapped[f"private_{key}"] = (
                type_,
                Field(
                    default=default_,
                    alias=key,
                    serialization_alias=key,
                    title=key.lstrip("_").replace("_", " ").title(),
                ),
            )
        else:
            remapped[key] = value
    return remapped


def create_model(
    model_name: str,
    *,
    field_definitions: dict[str, Any] | None = None,
    root: Any | None = None,
) -> type[BaseModel]:
    """주어진 필드 정의로 pydantic 모델을 생성합니다.

    주의:
        langchain 패키지 외부에서는 사용하지 마세요. 이 API는
        언제든지 변경될 수 있습니다.

    Args:
        model_name: 모델의 이름입니다.
        module_name: 모델이 정의된 모듈의 이름입니다.
            Pydantic에서 forward reference를 해결하는 데 사용됩니다.
        field_definitions: 모델의 필드 정의입니다.
        root: 루트 모델(RootModel)의 타입입니다.

    Returns:
        Type[BaseModel]: 생성된 모델입니다.
    """
    field_definitions = field_definitions or {}

    if root:
        if field_definitions:
            msg = (
                "When specifying __root__ no other "
                f"fields should be provided. Got {field_definitions}"
            )
            raise NotImplementedError(msg)

        if isinstance(root, tuple):
            kwargs = {"type_": root[0], "default_": root[1]}
        else:
            kwargs = {"type_": root}

        try:
            named_root_model = _create_root_model_cached(model_name, **kwargs)
        except TypeError:
            # _create_root_model_cached에 전달된 인자 중 해시 가능하지 않은 것이 있습니다
            named_root_model = _create_root_model(
                model_name,
                **kwargs,
            )
        return named_root_model

    # root가 없고 필드 정의만 있습니다
    names = set(field_definitions.keys())

    capture_warnings = False

    for name in names:
        # 또한 예약되지 않은 이름이 사용되는 경우(예: model_id 또는 model_name)
        if name.startswith("model"):
            capture_warnings = True

    with warnings.catch_warnings() if capture_warnings else nullcontext():
        if capture_warnings:
            warnings.filterwarnings(action="ignore")
        try:
            return _create_model_cached(model_name, **field_definitions)
        except TypeError:
            # 필드 정의 중 해시 가능하지 않은 것이 있습니다
            return _create_model_base(
                model_name,
                __config__=_SchemaConfig,
                **_remap_field_definitions(field_definitions),
            )


def is_supported_by_pydantic(type_: Any) -> bool:
    """주어진 "복잡한" 타입이 pydantic에서 지원되는지 확인합니다.

    int, str 등의 원시 타입에 대해서는 False를 반환합니다.

    이 확인은 dataclass, TypedDict 등의 컨테이너 타입을 위한 것입니다.
    """
    if is_dataclass(type_):
        return True

    if isinstance(type_, type) and issubclass(type_, BaseModel):
        return True

    if hasattr(type_, "__orig_bases__"):
        for base in type_.__orig_bases__:
            if base is TypedDict:
                return True
            elif base is typing.TypedDict:  # noqa: TID251
                # 이 경우 typing.TypedDict를 사용하는 것이 괜찮으므로 TID251을 무시합니다.
                # Pydantic은 Python 3.12부터 typing.TypedDict를 지원합니다.
                # 이전 버전에서는 typing_extensions.TypedDict만 지원됩니다.
                if sys.version_info >= (3, 12):
                    return True
    return False
