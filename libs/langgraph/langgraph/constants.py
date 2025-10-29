import sys
from typing import Any
from warnings import warn

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINTER,
    TASKS,
)
from langgraph.warnings import LangGraphDeprecatedSinceV10

__all__ = (
    "TAG_NOSTREAM",
    "TAG_HIDDEN",
    "START",
    "END",
    # 하위 호환성을 위해 유지됨 (주로 langgraph-api), v2 (또는 그 이전)에서 제거될 예정
    "CONF",
    "TASKS",
    "CONFIG_KEY_CHECKPOINTER",
)

# --- 공개 상수 ---
TAG_NOSTREAM = sys.intern("nostream")
"""채팅 모델의 스트리밍을 비활성화하는 태그입니다."""
TAG_HIDDEN = sys.intern("langsmith:hidden")
"""특정 추적/스트리밍 환경에서 노드/엣지를 숨기는 태그입니다."""
END = sys.intern("__end__")
"""그래프 스타일 Pregel의 마지막 (가상일 수 있는) 노드입니다."""
START = sys.intern("__start__")
"""그래프 스타일 Pregel의 첫 번째 (가상일 수 있는) 노드입니다."""


def __getattr__(name: str) -> Any:
    if name in ["Send", "Interrupt"]:
        warn(
            f"Importing {name} from langgraph.constants is deprecated. "
            f"Please use 'from langgraph.types import {name}' instead.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )

        from importlib import import_module

        module = import_module("langgraph.types")
        return getattr(module, name)

    try:
        from importlib import import_module

        private_constants = import_module("langgraph._internal._constants")
        attr = getattr(private_constants, name)
        warn(
            f"Importing {name} from langgraph.constants is deprecated. "
            f"This constant is now private and should not be used directly. "
            "Please let the LangGraph team know if you need this value.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        return attr
    except AttributeError:
        pass

    raise AttributeError(f"module has no attribute '{name}'")
