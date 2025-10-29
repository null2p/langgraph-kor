import asyncio
import sys
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.store.base import BaseStore

from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.types import StreamWriter


def _no_op_stream_writer(c: Any) -> None:
    pass


def get_config() -> RunnableConfig:
    if sys.version_info < (3, 11):
        try:
            if asyncio.current_task():
                raise RuntimeError(
                    "Python 3.11 or later required to use this in an async context"
                )
        except RuntimeError:
            pass
    if var_config := var_child_runnable_config.get():
        return var_config
    else:
        raise RuntimeError("Called get_config outside of a runnable context")


def get_store() -> BaseStore:
    """런타임에 그래프 노드 또는 엔트리포인트 작업 내부에서 LangGraph 스토어에 액세스합니다.

    StateGraph 또는 [entrypoint][langgraph.func.entrypoint]가 스토어로 초기화된 경우,
    모든 [StateGraph][langgraph.graph.StateGraph] 노드 또는
    함수형 API [task][langgraph.func.task] 내부에서 호출할 수 있습니다. 예:

    ```python
    # StateGraph와 함께 사용
    graph = (
        StateGraph(...)
        ...
        .compile(store=store)
    )

    # 또는 entrypoint와 함께 사용
    @entrypoint(store=store)
    def workflow(inputs):
        ...
    ```

    !!! warning "Python < 3.11과 비동기"

        Python < 3.11을 사용하고 LangGraph를 비동기적으로 실행하는 경우,
        `get_store()`는 [contextvar](https://docs.python.org/3/library/contextvars.html) 전파([Python >= 3.11](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)에서만 사용 가능)를 사용하므로 작동하지 않습니다.


    예제: StateGraph와 함께 사용
        ```python
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START
        from langgraph.store.memory import InMemoryStore
        from langgraph.config import get_store

        store = InMemoryStore()
        store.put(("values",), "foo", {"bar": 2})


        class State(TypedDict):
            foo: int


        def my_node(state: State):
            my_store = get_store()
            stored_value = my_store.get(("values",), "foo").value["bar"]
            return {"foo": stored_value + 1}


        graph = (
            StateGraph(State)
            .add_node(my_node)
            .add_edge(START, "my_node")
            .compile(store=store)
        )

        graph.invoke({"foo": 1})
        ```

        ```pycon
        {"foo": 3}
        ```

    예제: 함수형 API와 함께 사용
        ```python
        from langgraph.func import entrypoint, task
        from langgraph.store.memory import InMemoryStore
        from langgraph.config import get_store

        store = InMemoryStore()
        store.put(("values",), "foo", {"bar": 2})


        @task
        def my_task(value: int):
            my_store = get_store()
            stored_value = my_store.get(("values",), "foo").value["bar"]
            return stored_value + 1


        @entrypoint(store=store)
        def workflow(value: int):
            return my_task(value).result()


        workflow.invoke(1)
        ```

        ```pycon
        3
        ```
    """
    return get_config()[CONF][CONFIG_KEY_RUNTIME].store


def get_stream_writer() -> StreamWriter:
    """런타임에 그래프 노드 또는 엔트리포인트 작업 내부에서 LangGraph [StreamWriter][langgraph.types.StreamWriter]에 액세스합니다.

    모든 [StateGraph][langgraph.graph.StateGraph] 노드 또는
    함수형 API [task][langgraph.func.task] 내부에서 호출할 수 있습니다.

    !!! warning "Python < 3.11과 비동기"

        Python < 3.11을 사용하고 LangGraph를 비동기적으로 실행하는 경우,
        `get_stream_writer()`는 [contextvar](https://docs.python.org/3/library/contextvars.html) 전파([Python >= 3.11](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)에서만 사용 가능)를 사용하므로 작동하지 않습니다.

    예제: StateGraph와 함께 사용
        ```python
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, START
        from langgraph.config import get_stream_writer


        class State(TypedDict):
            foo: int


        def my_node(state: State):
            my_stream_writer = get_stream_writer()
            my_stream_writer({"custom_data": "Hello!"})
            return {"foo": state["foo"] + 1}


        graph = (
            StateGraph(State)
            .add_node(my_node)
            .add_edge(START, "my_node")
            .compile(store=store)
        )

        for chunk in graph.stream({"foo": 1}, stream_mode="custom"):
            print(chunk)
        ```

        ```pycon
        {"custom_data": "Hello!"}
        ```

    예제: 함수형 API와 함께 사용
        ```python
        from langgraph.func import entrypoint, task
        from langgraph.config import get_stream_writer


        @task
        def my_task(value: int):
            my_stream_writer = get_stream_writer()
            my_stream_writer({"custom_data": "Hello!"})
            return value + 1


        @entrypoint(store=store)
        def workflow(value: int):
            return my_task(value).result()


        for chunk in workflow.stream(1, stream_mode="custom"):
            print(chunk)
        ```

        ```pycon
        {"custom_data": "Hello!"}
        ```
    """
    runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
    return runtime.stream_writer
