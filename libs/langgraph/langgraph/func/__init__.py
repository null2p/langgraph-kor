from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
    get_args,
    get_origin,
    overload,
)

from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from typing_extensions import Unpack

from langgraph._internal._constants import CACHE_NS_WRITES, PREVIOUS
from langgraph._internal._typing import MISSING, DeprecatedKwargs
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.constants import END, START
from langgraph.pregel import Pregel
from langgraph.pregel._call import (
    P,
    SyncAsyncFuture,
    T,
    call,
    get_runnable_for_entrypoint,
    identifier,
)
from langgraph.pregel._read import PregelNode
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph.types import _DC_KWARGS, CachePolicy, RetryPolicy, StreamMode
from langgraph.typing import ContextT
from langgraph.warnings import LangGraphDeprecatedSinceV05, LangGraphDeprecatedSinceV10

__all__ = ("task", "entrypoint")


class _TaskFunction(Generic[P, T]):
    def __init__(
        self,
        func: Callable[P, Awaitable[T]] | Callable[P, T],
        *,
        retry_policy: Sequence[RetryPolicy],
        cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            if hasattr(func, "__func__"):
                # 클래스 메서드 처리
                # 참고: 여러 작업에서 공유될 수 있는 원래 클래스 메서드를 수정하지 않기 위해
                # 인스턴스 메서드를 수정합니다
                instance_method = functools.partial(func.__func__, func.__self__)  # type: ignore [union-attr]
                instance_method.__name__ = name  # type: ignore [attr-defined]
                func = instance_method
            else:
                # 일반 함수 / partial / 호출 가능한 클래스 등 처리
                func.__name__ = name
        self.func = func
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        functools.update_wrapper(self, func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]:
        return call(
            self.func,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
            *args,
            **kwargs,
        )

    def clear_cache(self, cache: BaseCache) -> None:
        """이 작업의 캐시를 지웁니다."""
        if self.cache_policy is not None:
            cache.clear(((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),))

    async def aclear_cache(self, cache: BaseCache) -> None:
        """이 작업의 캐시를 지웁니다."""
        if self.cache_policy is not None:
            await cache.aclear(
                ((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),)
            )


@overload
def task(
    __func_or_none__: None = None,
    *,
    name: str | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> Callable[
    [Callable[P, Awaitable[T]] | Callable[P, T]],
    _TaskFunction[P, T],
]: ...


@overload
def task(__func_or_none__: Callable[P, Awaitable[T]]) -> _TaskFunction[P, T]: ...


@overload
def task(__func_or_none__: Callable[P, T]) -> _TaskFunction[P, T]: ...


def task(
    __func_or_none__: Callable[P, Awaitable[T]] | Callable[P, T] | None = None,
    *,
    name: str | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy[Callable[P, str | bytes]] | None = None,
    **kwargs: Unpack[DeprecatedKwargs],
) -> (
    Callable[[Callable[P, Awaitable[T]] | Callable[P, T]], _TaskFunction[P, T]]
    | _TaskFunction[P, T]
):
    """`task` 데코레이터를 사용하여 LangGraph 작업을 정의합니다.

    !!! important "비동기 함수에는 Python 3.11 이상이 필요합니다"
        `task` 데코레이터는 동기 및 비동기 함수를 모두 지원합니다. 비동기
        함수를 사용하려면 Python 3.11 이상을 사용해야 합니다.

    작업은 [`entrypoint`][langgraph.func.entrypoint] 내부나
    `StateGraph` 내부에서만 호출할 수 있습니다. 작업은 일반 함수처럼 호출할 수 있지만
    다음과 같은 차이점이 있습니다:

    - 체크포인터가 활성화된 경우, 함수의 입력과 출력은 직렬화 가능해야 합니다.
    - 데코레이트된 함수는 entrypoint 또는 `StateGraph` 내부에서만 호출할 수 있습니다.
    - 함수를 호출하면 future가 생성됩니다. 이를 통해 작업을 쉽게 병렬화할 수 있습니다.

    Args:
        name: 작업의 선택적 이름입니다. 제공되지 않으면 함수 이름이 사용됩니다.
        retry_policy: 실패 시 작업에 사용할 선택적 재시도 정책(또는 정책 목록)입니다.
        cache_policy: 작업에 사용할 선택적 캐시 정책입니다. 작업 결과를 캐싱할 수 있습니다.

    Returns:
        데코레이터로 사용될 때 호출 가능한 함수입니다.

    Example: 동기 작업
        ```python
        from langgraph.func import entrypoint, task


        @task
        def add_one(a: int) -> int:
            return a + 1


        @entrypoint()
        def add_one(numbers: list[int]) -> list[int]:
            futures = [add_one(n) for n in numbers]
            results = [f.result() for f in futures]
            return results


        # entrypoint 호출
        add_one.invoke([1, 2, 3])  # [2, 3, 4] 반환
        ```

    Example: 비동기 작업
        ```python
        import asyncio
        from langgraph.func import entrypoint, task


        @task
        async def add_one(a: int) -> int:
            return a + 1


        @entrypoint()
        async def add_one(numbers: list[int]) -> list[int]:
            futures = [add_one(n) for n in numbers]
            return asyncio.gather(*futures)


        # entrypoint 호출
        await add_one.ainvoke([1, 2, 3])  # [2, 3, 4] 반환
        ```
    """
    if (retry := kwargs.get("retry", MISSING)) is not MISSING:
        warnings.warn(
            "`retry` is deprecated and will be removed. Please use `retry_policy` instead.",
            category=LangGraphDeprecatedSinceV05,
            stacklevel=2,
        )
        if retry_policy is None:
            retry_policy = retry  # type: ignore[assignment]

    retry_policies: Sequence[RetryPolicy] = (
        ()
        if retry_policy is None
        else (retry_policy,)
        if isinstance(retry_policy, RetryPolicy)
        else retry_policy
    )

    def decorator(
        func: Callable[P, Awaitable[T]] | Callable[P, T],
    ) -> Callable[P, SyncAsyncFuture[T]]:
        return _TaskFunction(
            func, retry_policy=retry_policies, cache_policy=cache_policy, name=name
        )

    if __func_or_none__ is not None:
        return decorator(__func_or_none__)

    return decorator


R = TypeVar("R")
S = TypeVar("S")


# 데코레이터는 `final` 속성을 지원하기 위해 클래스로 래핑되었습니다.
# 이 형태에서 `final` 속성은 IDE 자동 완성 및
# 타입 체크 도구와 잘 작동해야 합니다.
# 또한 이 정보를 API 참조에서 표시할 수 있습니다.
class entrypoint(Generic[ContextT]):
    """`entrypoint` 데코레이터를 사용하여 LangGraph 워크플로를 정의합니다.

    ### 함수 시그니처

    데코레이트된 함수는 **단일 매개변수**를 받아야 하며, 이는 함수에 대한 입력
    역할을 합니다. 이 입력 매개변수는 모든 타입이 될 수 있습니다. 딕셔너리를 사용하여
    함수에 **여러 매개변수**를 전달할 수 있습니다.

    ### 주입 가능한 매개변수

    데코레이트된 함수는 런타임에 자동으로 주입될
    추가 매개변수에 대한 액세스를 요청할 수 있습니다. 이러한 매개변수는 다음과 같습니다:

    | 매개변수         | 설명                                                                                                  |
    |------------------|------------------------------------------------------------------------------------------------------|
    | **`config`**     | 런타임 구성 값을 보유하는 구성 객체(일명 `RunnableConfig`)입니다.                                      |
    | **`previous`**   | 주어진 스레드에 대한 이전 반환 값입니다(체크포인터가 제공될 때만 사용 가능).                             |
    | **`runtime`**    | context, store, writer를 포함하는 현재 실행에 대한 정보를 담고 있는 `Runtime` 객체입니다.               |

    entrypoint 데코레이터는 동기 함수 또는 비동기 함수에 적용할 수 있습니다.

    ### 상태 관리

    **`previous`** 매개변수는 동일한 스레드 ID에서 entrypoint의 이전
    호출의 반환 값에 액세스하는 데 사용할 수 있습니다. 이 값은 체크포인터가
    제공될 때만 사용 가능합니다.

    **`previous`**를 반환 값과 다르게 하고 싶다면, `entrypoint.final`
    객체를 사용하여 값을 반환하면서 체크포인트에 다른 값을 저장할 수 있습니다.

    Args:
        checkpointer: 실행 간에 상태를 유지할 수 있는 워크플로를 생성하기 위한
            체크포인터를 지정합니다.
        store: 일반화된 키-값 저장소입니다. 일부 구현은 선택적 `index` 구성을 통해
            시맨틱 검색 기능을 지원할 수 있습니다.
        cache: 워크플로 결과를 캐싱하는 데 사용할 캐시입니다.
        context_schema: 워크플로에 전달될 컨텍스트 객체의 스키마를 지정합니다.
        cache_policy: 워크플로 결과를 캐싱하는 데 사용할 캐시 정책입니다.
        retry_policy: 실패 시 워크플로에 사용할 재시도 정책(또는 정책 목록)입니다.

    !!! warning "`config_schema` 더 이상 사용되지 않음"
        `config_schema` 매개변수는 v0.6.0에서 더 이상 사용되지 않으며 v2.0.0에서 지원이 제거됩니다.
        실행 범위 컨텍스트의 스키마를 지정하려면 `context_schema`를 사용하십시오.


    Example: entrypoint와 task 사용하기
        ```python
        import time

        from langgraph.func import entrypoint, task
        from langgraph.types import interrupt, Command
        from langgraph.checkpoint.memory import InMemorySaver

        @task
        def compose_essay(topic: str) -> str:
            time.sleep(1.0)  # 느린 작업 시뮬레이션
            return f"An essay about {topic}"

        @entrypoint(checkpointer=InMemorySaver())
        def review_workflow(topic: str) -> dict:
            \"\"\"에세이 생성 및 검토를 위한 워크플로를 관리합니다.

            워크플로에는 다음이 포함됩니다:
            1. 주어진 주제에 대한 에세이 생성.
            2. 생성된 에세이의 사람 검토를 위해 워크플로 중단.

            워크플로를 재개하면 compose_essay 작업은 다시 실행되지 않습니다.
            결과가 체크포인터에 의해 캐시되기 때문입니다.

            Args:
                topic: 에세이의 주제.

            Returns:
                dict: 생성된 에세이와 사람의 검토를 포함하는 딕셔너리.
            \"\"\"
            essay_future = compose_essay(topic)
            essay = essay_future.result()
            human_review = interrupt({
                \"question\": \"검토를 제공해 주세요\",
                \"essay\": essay
            })
            return {
                \"essay\": essay,
                \"review\": human_review,
            }

        # 워크플로에 대한 구성 예제
        config = {
            \"configurable\": {
                \"thread_id\": \"some_thread\"
            }
        }

        # 에세이 주제
        topic = \"cats\"

        # 워크플로를 스트림하여 에세이를 생성하고 사람의 검토를 기다림
        for result in review_workflow.stream(topic, config):
            print(result)

        # 인터럽트 후 제공된 사람의 검토 예제
        human_review = \"This essay is great.\"

        # 제공된 사람의 검토로 워크플로 재개
        for result in review_workflow.stream(Command(resume=human_review), config):
            print(result)
        ```

    Example: 이전 반환 값에 액세스하기
        체크포인터가 활성화되면 함수는 동일한 스레드 ID에서
        이전 호출의 이전 반환 값에 액세스할 수 있습니다.

        ```python
        from typing import Optional

        from langgraph.checkpoint.memory import MemorySaver

        from langgraph.func import entrypoint


        @entrypoint(checkpointer=InMemorySaver())
        def my_workflow(input_data: str, previous: Optional[str] = None) -> str:
            return "world"


        config = {"configurable": {"thread_id": "some_thread"}}
        my_workflow.invoke("hello", config)
        ```

    Example: entrypoint.final을 사용하여 값 저장하기
        `entrypoint.final` 객체를 사용하면 값을 반환하면서
        체크포인트에 다른 값을 저장할 수 있습니다. 이 값은 동일한 스레드 ID가
        사용되는 한 `previous` 매개변수를 통해 entrypoint의 다음 호출에서
        액세스할 수 있습니다.

        ```python
        from typing import Any

        from langgraph.checkpoint.memory import MemorySaver

        from langgraph.func import entrypoint


        @entrypoint(checkpointer=InMemorySaver())
        def my_workflow(
            number: int,
            *,
            previous: Any = None,
        ) -> entrypoint.final[int, int]:
            previous = previous or 0
            # 이렇게 하면 호출자에게 이전 값을 반환하면서
            # 체크포인트에 2 * number를 저장하며, 이는 다음 호출에서
            # `previous` 매개변수에 사용됩니다.
            return entrypoint.final(value=previous, save=2 * number)


        config = {"configurable": {"thread_id": "some_thread"}}

        my_workflow.invoke(3, config)  # 0 (previous가 None이었음)
        my_workflow.invoke(1, config)  # 6 (previous가 이전 호출의 3 * 2였음)
        ```
    """

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        cache: BaseCache | None = None,
        context_schema: type[ContextT] | None = None,
        cache_policy: CachePolicy | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> None:
        """entrypoint 데코레이터를 초기화합니다."""
        if (config_schema := kwargs.get("config_schema", MISSING)) is not MISSING:
            warnings.warn(
                "`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
                category=LangGraphDeprecatedSinceV10,
                stacklevel=2,
            )
            if context_schema is None:
                context_schema = cast(type[ContextT], config_schema)

        if (retry := kwargs.get("retry", MISSING)) is not MISSING:
            warnings.warn(
                "`retry` is deprecated and will be removed. Please use `retry_policy` instead.",
                category=LangGraphDeprecatedSinceV05,
                stacklevel=2,
            )
            if retry_policy is None:
                retry_policy = cast("RetryPolicy | Sequence[RetryPolicy]", retry)

        self.checkpointer = checkpointer
        self.store = store
        self.cache = cache
        self.cache_policy = cache_policy
        self.retry_policy = retry_policy
        self.context_schema = context_schema

    @dataclass(**_DC_KWARGS)
    class final(Generic[R, S]):
        """entrypoint에서 반환할 수 있는 프리미티브입니다.

        이 프리미티브를 사용하면 entrypoint의 반환 값과는
        별도로 체크포인터에 값을 저장할 수 있습니다.

        Example: 반환 값과 저장 값 분리하기
            ```python
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.func import entrypoint


            @entrypoint(checkpointer=InMemorySaver())
            def my_workflow(
                number: int,
                *,
                previous: Any = None,
            ) -> entrypoint.final[int, int]:
                previous = previous or 0
                # 이렇게 하면 호출자에게 이전 값을 반환하면서
                # 체크포인트에 2 * number를 저장하며, 이는 다음 호출에서
                # `previous` 매개변수에 사용됩니다.
                return entrypoint.final(value=previous, save=2 * number)


            config = {"configurable": {"thread_id": "1"}}

            my_workflow.invoke(3, config)  # 0 (previous가 None이었음)
            my_workflow.invoke(1, config)  # 6 (previous가 이전 호출의 3 * 2였음)
            ```
        """

        value: R
        """반환할 값입니다. `None`이더라도 항상 반환됩니다."""
        save: S
        """다음 체크포인트를 위한 상태 값입니다.

        `None`이더라도 항상 저장됩니다.
        """

    def __call__(self, func: Callable[..., Any]) -> Pregel:
        """함수를 Pregel 그래프로 변환합니다.

        Args:
            func: 변환할 함수입니다. 동기 및 비동기 함수를 모두 지원합니다.

        Returns:
            Pregel 그래프입니다.
        """
        # StreamWriter에 쓰는 함수로 제너레이터를 래핑
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            raise NotImplementedError(
                "Generators are not supported in the Functional API."
            )

        bound = get_runnable_for_entrypoint(func)
        stream_mode: StreamMode = "updates"

        # 입력 및 출력 타입 가져오기
        sig = inspect.signature(func)
        first_parameter_name = next(iter(sig.parameters.keys()), None)
        if not first_parameter_name:
            raise ValueError("Entrypoint function must have at least one parameter")
        input_type = (
            sig.parameters[first_parameter_name].annotation
            if sig.parameters[first_parameter_name].annotation
            is not inspect.Signature.empty
            else Any
        )

        def _pluck_return_value(value: Any) -> Any:
            """entrypoint.final 객체에서 return_ 값을 추출하거나 그대로 전달합니다."""
            return value.value if isinstance(value, entrypoint.final) else value

        def _pluck_save_value(value: Any) -> Any:
            """entrypoint.final 객체에서 save 값을 가져오거나 그대로 전달합니다."""
            return value.save if isinstance(value, entrypoint.final) else value

        output_type, save_type = Any, Any
        if sig.return_annotation is not inspect.Signature.empty:
            # 사용자가 entrypoint.final을 올바르게 매개변수화하지 않음
            if (
                sig.return_annotation is entrypoint.final
            ):  # 매개변수화되지 않은 entrypoint.final
                output_type = save_type = Any
            else:
                origin = get_origin(sig.return_annotation)
                if origin is entrypoint.final:
                    type_annotations = get_args(sig.return_annotation)
                    if len(type_annotations) != 2:
                        raise TypeError(
                            "Please an annotation for both the return_ and "
                            "the save values."
                            "For example, `-> entrypoint.final[int, str]` would assign a "
                            "return_ a type of `int` and save the type `str`."
                        )
                    output_type, save_type = get_args(sig.return_annotation)
                else:
                    output_type = save_type = sig.return_annotation

        return Pregel(
            nodes={
                func.__name__: PregelNode(
                    bound=bound,
                    triggers=[START],
                    channels=START,
                    writers=[
                        ChannelWrite(
                            [
                                ChannelWriteEntry(END, mapper=_pluck_return_value),
                                ChannelWriteEntry(PREVIOUS, mapper=_pluck_save_value),
                            ]
                        )
                    ],
                )
            },
            channels={
                START: EphemeralValue(input_type),
                END: LastValue(output_type, END),
                PREVIOUS: LastValue(save_type, PREVIOUS),
            },
            input_channels=START,
            output_channels=END,
            stream_channels=END,
            stream_mode=stream_mode,
            stream_eager=True,
            checkpointer=self.checkpointer,
            store=self.store,
            cache=self.cache,
            cache_policy=self.cache_policy,
            retry_policy=self.retry_policy or (),
            context_schema=self.context_schema,  # type: ignore[arg-type]
        )
