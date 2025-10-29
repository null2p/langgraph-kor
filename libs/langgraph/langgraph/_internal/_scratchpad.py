import dataclasses
from collections.abc import Callable
from typing import Any

from langgraph.types import _DC_KWARGS


@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int
    stop: int
    # 호출
    call_counter: Callable[[], int]
    # 인터럽트
    interrupt_counter: Callable[[], int]
    get_null_resume: Callable[[bool], Any]
    resume: list[Any]
    # 서브그래프
    subgraph_counter: Callable[[], int]
