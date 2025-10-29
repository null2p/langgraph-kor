from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, Sequence
from os import getenv
from typing import Any, cast

from langchain_core.callbacks import (
    AsyncCallbackManager,
    BaseCallbackManager,
    CallbackManager,
    Callbacks,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import (
    CONFIG_KEYS,
    COPIABLE_KEYS,
    var_child_runnable_config,
)
from langgraph.checkpoint.base import CheckpointMetadata

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    NS_END,
    NS_SEP,
)

DEFAULT_RECURSION_LIMIT = int(getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "25"))


def recast_checkpoint_ns(ns: str) -> str:
    """체크포인트 네임스페이스에서 태스크 ID를 제거합니다.

    Args:
        ns: 태스크 ID가 포함된 체크포인트 네임스페이스입니다.

    Returns:
        str: 태스크 ID가 없는 체크포인트 네임스페이스입니다.
    """
    return NS_SEP.join(
        part.split(NS_END)[0] for part in ns.split(NS_SEP) if not part.isdigit()
    )


def patch_configurable(
    config: RunnableConfig | None, patch: dict[str, Any]
) -> RunnableConfig:
    if config is None:
        return {CONF: patch}
    elif CONF not in config:
        return {**config, CONF: patch}
    else:
        return {**config, CONF: {**config[CONF], **patch}}


def patch_checkpoint_map(
    config: RunnableConfig | None, metadata: CheckpointMetadata | None
) -> RunnableConfig:
    if config is None:
        return config
    elif parents := (metadata.get("parents") if metadata else None):
        conf = config[CONF]
        return patch_configurable(
            config,
            {
                CONFIG_KEY_CHECKPOINT_MAP: {
                    **parents,
                    conf[CONFIG_KEY_CHECKPOINT_NS]: conf[CONFIG_KEY_CHECKPOINT_ID],
                },
            },
        )
    else:
        return config


def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    """여러 config를 하나로 병합합니다.

    Args:
        *configs: 병합할 config들입니다.

    Returns:
        RunnableConfig: 병합된 config입니다.
    """
    base: RunnableConfig = {}
    # 키가 리터럴이 아니더라도, 두 dict가 같은 타입이므로
    # 이것은 올바릅니다
    for config in configs:
        if config is None:
            continue
        for key, value in config.items():
            if not value:
                continue
            if key == "metadata":
                if base_value := base.get(key):
                    base[key] = {**base_value, **value}  # type: ignore
                else:
                    base[key] = value  # type: ignore[literal-required]
            elif key == "tags":
                if base_value := base.get(key):
                    base[key] = [*base_value, *value]  # type: ignore
                else:
                    base[key] = value  # type: ignore[literal-required]
            elif key == CONF:
                if base_value := base.get(key):
                    base[key] = {**base_value, **value}  # type: ignore[dict-item]
                else:
                    base[key] = value
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                # callbacks는 None, list[handler] 또는 manager일 수 있으므로
                # 두 callbacks 값을 병합하는 경우의 수는 6가지입니다
                if isinstance(value, list):
                    if base_callbacks is None:
                        base["callbacks"] = value.copy()
                    elif isinstance(base_callbacks, list):
                        base["callbacks"] = base_callbacks + value
                    else:
                        # base_callbacks는 manager입니다
                        mngr = base_callbacks.copy()
                        for callback in value:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                elif isinstance(value, BaseCallbackManager):
                    # value는 manager입니다
                    if base_callbacks is None:
                        base["callbacks"] = value.copy()
                    elif isinstance(base_callbacks, list):
                        mngr = value.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                        # base_callbacks도 manager입니다
                        base["callbacks"] = base_callbacks.merge(value)
                else:
                    raise NotImplementedError
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            else:
                base[key] = config[key]  # type: ignore[literal-required]
    if CONF not in base:
        base[CONF] = {}
    return base


def patch_config(
    config: RunnableConfig | None,
    *,
    callbacks: Callbacks = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    run_name: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig:
    """새로운 값으로 config를 패치합니다.

    Args:
        config: 패치할 config입니다.
        callbacks: 설정할 callbacks입니다.
        recursion_limit: 설정할 재귀 제한입니다.
        max_concurrency: 실행할 동시 단계의 최대 수로, 병렬화된 단계에도 적용됩니다.
        run_name: 설정할 run 이름입니다.
        configurable: 설정할 configurable입니다.

    Returns:
        RunnableConfig: 패치된 config입니다.
    """
    config = config.copy() if config is not None else {}
    if callbacks is not None:
        # callbacks를 교체하는 경우 run_name을 제거해야 합니다
        # 이는 원래 callbacks와 동일한 run에만 적용되어야 하기 때문입니다
        config["callbacks"] = callbacks
        if "run_name" in config:
            del config["run_name"]
        if "run_id" in config:
            del config["run_id"]
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name is not None:
        config["run_name"] = run_name
    if configurable is not None:
        config[CONF] = {**config.get(CONF, {}), **configurable}
    return config


def get_callback_manager_for_config(
    config: RunnableConfig, tags: Sequence[str] | None = None
) -> CallbackManager:
    """config를 위한 callback manager를 가져옵니다.

    Args:
        config: config입니다.

    Returns:
        CallbackManager: callback manager입니다.
    """
    from langchain_core.callbacks.manager import CallbackManager

    # tags를 병합합니다
    all_tags = config.get("tags")
    if all_tags is not None and tags is not None:
        all_tags = [*all_tags, *tags]
    elif tags is not None:
        all_tags = list(tags)
    # 기존 callbacks가 있으면 사용합니다
    if (callbacks := config.get("callbacks")) and isinstance(
        callbacks, CallbackManager
    ):
        if all_tags:
            callbacks.add_tags(all_tags)
        if metadata := config.get("metadata"):
            callbacks.add_metadata(metadata)
        return callbacks
    else:
        # 그렇지 않으면 새 manager를 생성합니다
        return CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=all_tags,
            inheritable_metadata=config.get("metadata"),
        )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
    tags: Sequence[str] | None = None,
) -> AsyncCallbackManager:
    """config를 위한 비동기 callback manager를 가져옵니다.

    Args:
        config: config입니다.

    Returns:
        AsyncCallbackManager: 비동기 callback manager입니다.
    """
    from langchain_core.callbacks.manager import AsyncCallbackManager

    # tags를 병합합니다
    all_tags = config.get("tags")
    if all_tags is not None and tags is not None:
        all_tags = [*all_tags, *tags]
    elif tags is not None:
        all_tags = list(tags)
    # 기존 callbacks가 있으면 사용합니다
    if (callbacks := config.get("callbacks")) and isinstance(
        callbacks, AsyncCallbackManager
    ):
        if all_tags:
            callbacks.add_tags(all_tags)
        if metadata := config.get("metadata"):
            callbacks.add_metadata(metadata)
        return callbacks
    else:
        # 그렇지 않으면 새 manager를 생성합니다
        return AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=all_tags,
            inheritable_metadata=config.get("metadata"),
        )


def _is_not_empty(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    else:
        return value is not None


def ensure_config(*configs: RunnableConfig | None) -> RunnableConfig:
    """모든 키를 포함하는 config를 반환하며, 제공된 config들을 병합합니다.

    Args:
        *configs: 기본값을 보장하기 전에 병합할 config들입니다.

    Returns:
        RunnableConfig: 병합되고 보장된 config입니다.
    """
    empty = RunnableConfig(
        tags=[],
        metadata=ChainMap(),
        callbacks=None,
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        configurable={},
    )
    if var_config := var_child_runnable_config.get():
        empty.update(
            {
                k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                for k, v in var_config.items()
                if _is_not_empty(v)
            },
        )
    for config in configs:
        if config is None:
            continue
        for k, v in config.items():
            if _is_not_empty(v) and k in CONFIG_KEYS:
                if k == CONF:
                    empty[k] = cast(dict, v).copy()
                else:
                    empty[k] = v  # type: ignore[literal-required]
        for k, v in config.items():
            if _is_not_empty(v) and k not in CONFIG_KEYS:
                empty[CONF][k] = v
    _empty_metadata = empty["metadata"]
    for key, value in empty[CONF].items():
        if _exclude_as_metadata(key, value, _empty_metadata):
            continue
        _empty_metadata[key] = value
    return empty


_OMIT = ("key", "token", "secret", "password", "auth")


def _exclude_as_metadata(key: str, value: Any, metadata: Mapping[str, Any]) -> bool:
    key_lower = key.casefold()
    return (
        key.startswith("__")
        or not isinstance(value, (str, int, float, bool))
        or key in metadata
        or any(substr in key_lower for substr in _OMIT)
    )
