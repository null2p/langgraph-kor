from __future__ import annotations

import asyncio
import enum
import inspect
import sys
import warnings
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterator,
    Sequence,
)
from contextlib import AsyncExitStack, contextmanager
from contextvars import Context, Token, copy_context
from functools import partial, wraps
from typing import (
    Any,
    Optional,
    Protocol,
    TypeGuard,
    cast,
)

from langchain_core.runnables.base import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
)
from langchain_core.runnables.base import (
    RunnableLike as LCRunnableLike,
)
from langchain_core.runnables.config import (
    run_in_executor,
    var_child_runnable_config,
)
from langchain_core.runnables.utils import Input, Output
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.store.base import BaseStore

from langgraph._internal._config import (
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_RUNTIME,
)
from langgraph._internal._typing import MISSING
from langgraph.types import StreamWriter

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = None  # type: ignore


def _set_config_context(
    config: RunnableConfig, run: Any = None
) -> Token[RunnableConfig | None]:
    """자식 Runnable config + tracing context를 설정합니다.

    Args:
        config: 설정할 config입니다.
    """
    config_token = var_child_runnable_config.set(config)
    if run is not None:
        from langsmith.run_helpers import _set_tracing_context

        _set_tracing_context({"parent": run})
    return config_token


def _unset_config_context(token: Token[RunnableConfig | None], run: Any = None) -> None:
    """자식 Runnable config + tracing context를 해제합니다.

    Args:
        token: 재설정할 config 토큰입니다.
    """
    var_child_runnable_config.reset(token)
    if run is not None:
        from langsmith.run_helpers import _set_tracing_context

        _set_tracing_context(
            {
                "parent": None,
                "project_name": None,
                "tags": None,
                "metadata": None,
                "enabled": None,
                "client": None,
            }
        )


@contextmanager
def set_config_context(
    config: RunnableConfig, run: Any = None
) -> Generator[Context, None, None]:
    """자식 Runnable config + tracing context를 설정합니다.

    Args:
        config: 설정할 config입니다.
    """
    ctx = copy_context()
    config_token = ctx.run(_set_config_context, config, run)
    try:
        yield ctx
    finally:
        ctx.run(_unset_config_context, config_token, run)


# Python 3.11 이전에는 네이티브 StrEnum을 사용할 수 없습니다
class StrEnum(str, enum.Enum):
    """문자열 enum입니다."""


# 모든 타입이 허용됨을 나타내는 특수 타입
ANY_TYPE = object()

ASYNCIO_ACCEPTS_CONTEXT = sys.version_info >= (3, 11)

# 런타임에 노드/태스크/도구에 주입될 수 있는 키워드 인자 목록입니다.
# 명명된 인자는 서로 다른 타입으로 나타나는 경우 여러 번 나타날 수 있습니다.
KWARGS_CONFIG_KEYS: tuple[tuple[str, tuple[Any, ...], str, Any], ...] = (
    (
        "config",
        (
            RunnableConfig,
            "RunnableConfig",
            Optional[RunnableConfig],  # noqa: UP045
            "Optional[RunnableConfig]",
            inspect.Parameter.empty,
        ),
        # 현재는 config를 직접 사용하며, 결국 Runtime에서 제거될 것임
        "N/A",
        inspect.Parameter.empty,
    ),
    (
        "writer",
        (StreamWriter, "StreamWriter", inspect.Parameter.empty),
        "stream_writer",
        lambda _: None,
    ),
    (
        "store",
        (
            BaseStore,
            "BaseStore",
            inspect.Parameter.empty,
        ),
        "store",
        inspect.Parameter.empty,
    ),
    (
        "store",
        (
            Optional[BaseStore],  # noqa: UP045
            "Optional[BaseStore]",
        ),
        "store",
        None,
    ),
    (
        "previous",
        (ANY_TYPE,),
        "previous",
        inspect.Parameter.empty,
    ),
    (
        "runtime",
        (ANY_TYPE,),
        # 이 블록에는 절대 도달하지 않으며, runtime을 직접 주입함
        "N/A",
        inspect.Parameter.empty,
    ),
)
"""함수에 전달할 수 있는 kwargs의 목록과 해당하는
config 키, 기본값 및 타입 어노테이션입니다.

`Runtime` 객체에서 `invoke`, `ainvoke`, `stream` 및 `astream`의
kwargs로 런타임에 주입할 수 있는 키워드 인자를 구성하는 데 사용됩니다.

config 객체에서 키워드가 주입되려면 함수 서명에
동일한 이름과 일치하는 타입 어노테이션을 가진 kwarg가 포함되어야 합니다.

각 튜플에는 다음이 포함됩니다:
- 함수 서명의 kwarg 이름
- kwarg의 타입 어노테이션
- 값을 가져올 `Runtime` 속성 (해당하지 않는 경우 N/A)

이것은 완전히 내부용이며 BaseStore | None과 같이 형식화된
forward reference와 optional 타입을 해결하기 위해
`get_type_hints`를 사용하도록 추가로 리팩토링되어야 합니다.
"""

VALID_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class _RunnableWithWriter(Protocol[Input, Output]):
    def __call__(self, state: Input, *, writer: StreamWriter) -> Output: ...


class _RunnableWithStore(Protocol[Input, Output]):
    def __call__(self, state: Input, *, store: BaseStore) -> Output: ...


class _RunnableWithWriterStore(Protocol[Input, Output]):
    def __call__(
        self, state: Input, *, writer: StreamWriter, store: BaseStore
    ) -> Output: ...


class _RunnableWithConfigWriter(Protocol[Input, Output]):
    def __call__(
        self, state: Input, *, config: RunnableConfig, writer: StreamWriter
    ) -> Output: ...


class _RunnableWithConfigStore(Protocol[Input, Output]):
    def __call__(
        self, state: Input, *, config: RunnableConfig, store: BaseStore
    ) -> Output: ...


class _RunnableWithConfigWriterStore(Protocol[Input, Output]):
    def __call__(
        self,
        state: Input,
        *,
        config: RunnableConfig,
        writer: StreamWriter,
        store: BaseStore,
    ) -> Output: ...


RunnableLike = (
    LCRunnableLike
    | _RunnableWithWriter[Input, Output]
    | _RunnableWithStore[Input, Output]
    | _RunnableWithWriterStore[Input, Output]
    | _RunnableWithConfigWriter[Input, Output]
    | _RunnableWithConfigStore[Input, Output]
    | _RunnableWithConfigWriterStore[Input, Output]
)


class RunnableCallable(Runnable):
    """동기 및 비동기 함수를 필요로 하는 훨씬 간단한 버전의 RunnableLambda입니다."""

    def __init__(
        self,
        func: Callable[..., Any | Runnable] | None,
        afunc: Callable[..., Awaitable[Any | Runnable]] | None = None,
        *,
        name: str | None = None,
        tags: Sequence[str] | None = None,
        trace: bool = True,
        recurse: bool = True,
        explode_args: bool = False,
        **kwargs: Any,
    ) -> None:
        self.name = name
        if self.name is None:
            if func:
                try:
                    if func.__name__ != "<lambda>":
                        self.name = func.__name__
                except AttributeError:
                    pass
            elif afunc:
                try:
                    self.name = afunc.__name__
                except AttributeError:
                    pass
        self.func = func
        self.afunc = afunc
        self.tags = tags
        self.kwargs = kwargs
        self.trace = trace
        self.recurse = recurse
        self.explode_args = explode_args
        # 서명 확인
        if func is None and afunc is None:
            raise ValueError("At least one of func or afunc must be provided.")

        self.func_accepts: dict[str, tuple[str, Any]] = {}
        params = inspect.signature(cast(Callable, func or afunc)).parameters

        for kw, typ, runtime_key, default in KWARGS_CONFIG_KEYS:
            p = params.get(kw)

            if p is None or p.kind not in VALID_KINDS:
                # 파라미터를 찾을 수 없거나 유효한 종류가 아니면 건너뜀
                continue

            if typ != (ANY_TYPE,) and p.annotation not in typ:
                # 특정 타입이 필요하지만 함수 어노테이션이
                # 예상 타입과 일치하지 않음

                # 이것이 잘못된 타이핑의 config 파라미터인 경우 경고를 발생시킴
                # 이전에는 모든 타입을 지원했지만 더 정확한 타이핑으로 이동 중이기 때문
                if kw == "config" and p.annotation != inspect.Parameter.empty:
                    warnings.warn(
                        f"The 'config' parameter should be typed as 'RunnableConfig' or "
                        f"'RunnableConfig | None', not '{p.annotation}'. ",
                        UserWarning,
                        stacklevel=4,
                    )
                continue

            # 함수가 kwarg를 받아들이면 주입할 키 / runtime 속성을 저장
            self.func_accepts[kw] = (runtime_key, default)

    def __repr__(self) -> str:
        repr_args = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"name", "func", "afunc", "config", "kwargs", "trace"}
        }
        return f"{self.get_name()}({', '.join(f'{k}={v!r}' for k, v in repr_args.items())})"

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        if self.func is None:
            raise TypeError(
                f'No synchronous function provided to "{self.name}".'
                "\nEither initialize with a synchronous function or invoke"
                " via the async API (ainvoke, astream, etc.)"
            )
        if config is None:
            config = ensure_config()
        if self.explode_args:
            args, _kwargs = input
            kwargs = {**self.kwargs, **_kwargs, **kwargs}
        else:
            args = (input,)
            kwargs = {**self.kwargs, **kwargs}

        runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)

        for kw, (runtime_key, default) in self.func_accepts.items():
            # kwarg가 이미 설정되어 있으면 설정된 값을 사용
            if kw in kwargs:
                continue

            kw_value: Any = MISSING
            if kw == "config":
                kw_value = config
            elif runtime:
                if kw == "runtime":
                    kw_value = runtime
                else:
                    try:
                        kw_value = getattr(runtime, runtime_key)
                    except AttributeError:
                        pass

            if kw_value is MISSING:
                if default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Missing required config key '{runtime_key}' for '{self.name}'."
                    )
                kw_value = default
            kwargs[kw] = kw_value

        if self.trace:
            callback_manager = get_callback_manager_for_config(config, self.tags)
            run_manager = callback_manager.on_chain_start(
                None,
                input,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            try:
                child_config = patch_config(config, callbacks=run_manager.get_child())
                # run 가져오기
                for h in run_manager.handlers:
                    if isinstance(h, LangChainTracer):
                        run = h.run_map.get(str(run_manager.run_id))
                        break
                else:
                    run = None
                # 컨텍스트에서 실행
                with set_config_context(child_config, run) as context:
                    ret = context.run(self.func, *args, **kwargs)
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                run_manager.on_chain_end(ret)
        else:
            ret = self.func(*args, **kwargs)
        if self.recurse and isinstance(ret, Runnable):
            return ret.invoke(input, config)
        return ret

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        if not self.afunc:
            return self.invoke(input, config)
        if config is None:
            config = ensure_config()
        if self.explode_args:
            args, _kwargs = input
            kwargs = {**self.kwargs, **_kwargs, **kwargs}
        else:
            args = (input,)
            kwargs = {**self.kwargs, **kwargs}

        runtime = config.get(CONF, {}).get(CONFIG_KEY_RUNTIME)

        for kw, (runtime_key, default) in self.func_accepts.items():
            # kwarg가 이미 설정되어 있으면 설정된 값을 사용
            if kw in kwargs:
                continue

            kw_value: Any = MISSING
            if kw == "config":
                kw_value = config
            elif runtime:
                if kw == "runtime":
                    kw_value = runtime
                else:
                    try:
                        kw_value = getattr(runtime, runtime_key)
                    except AttributeError:
                        pass
            if kw_value is MISSING:
                if default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Missing required config key '{runtime_key}' for '{self.name}'."
                    )
                kw_value = default
            kwargs[kw] = kw_value

        if self.trace:
            callback_manager = get_async_callback_manager_for_config(config, self.tags)
            run_manager = await callback_manager.on_chain_start(
                None,
                input,
                name=config.get("run_name") or self.name,
                run_id=config.pop("run_id", None),
            )
            try:
                child_config = patch_config(config, callbacks=run_manager.get_child())
                coro = cast(Coroutine[None, None, Any], self.afunc(*args, **kwargs))
                if ASYNCIO_ACCEPTS_CONTEXT:
                    for h in run_manager.handlers:
                        if isinstance(h, LangChainTracer):
                            run = h.run_map.get(str(run_manager.run_id))
                            break
                    else:
                        run = None
                    with set_config_context(child_config, run) as context:
                        ret = await asyncio.create_task(coro, context=context)
                else:
                    ret = await coro
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise
            else:
                await run_manager.on_chain_end(ret)
        else:
            ret = await self.afunc(*args, **kwargs)
        if self.recurse and isinstance(ret, Runnable):
            return await ret.ainvoke(input, config)
        return ret


def is_async_callable(
    func: Any,
) -> TypeGuard[Callable[..., Awaitable]]:
    """함수가 비동기인지 확인합니다."""
    return (
        inspect.iscoroutinefunction(func)
        or hasattr(func, "__call__")
        and inspect.iscoroutinefunction(func.__call__)
    )


def is_async_generator(
    func: Any,
) -> TypeGuard[Callable[..., AsyncIterator]]:
    """함수가 비동기 제너레이터인지 확인합니다."""
    return (
        inspect.isasyncgenfunction(func)
        or hasattr(func, "__call__")
        and inspect.isasyncgenfunction(func.__call__)
    )


def coerce_to_runnable(
    thing: RunnableLike, *, name: str | None, trace: bool
) -> Runnable:
    """runnable-like 객체를 Runnable로 강제 변환합니다.

    Args:
        thing: runnable-like 객체입니다.

    Returns:
        Runnable입니다.
    """
    if isinstance(thing, Runnable):
        return thing
    elif is_async_generator(thing) or inspect.isgeneratorfunction(thing):
        return RunnableLambda(thing, name=name)
    elif callable(thing):
        if is_async_callable(thing):
            return RunnableCallable(None, thing, name=name, trace=trace)
        else:
            return RunnableCallable(
                thing,
                wraps(thing)(partial(run_in_executor, None, thing)),  # type: ignore[arg-type]
                name=name,
                trace=trace,
            )
    elif isinstance(thing, dict):
        return RunnableParallel(thing)
    else:
        raise TypeError(
            f"Expected a Runnable, callable or dict."
            f"Instead got an unsupported type: {type(thing)}"
        )


class RunnableSeq(Runnable):
    """각각의 출력이 다음의 입력이 되는 `Runnable`의 시퀀스입니다.

    `RunnableSeq`는 LangGraph 내부에서 사용하는 `RunnableSequence`의 더 간단한 버전입니다.
    """

    def __init__(
        self,
        *steps: RunnableLike,
        name: str | None = None,
        trace_inputs: Callable[[Any], Any] | None = None,
    ) -> None:
        """새로운 RunnableSeq를 생성합니다.

        Args:
            steps: 시퀀스에 포함할 단계들입니다.
            name: `Runnable`의 이름입니다.

        Raises:
            ValueError: 시퀀스가 2개 미만의 단계를 가진 경우.
        """
        steps_flat: list[Runnable] = []
        for step in steps:
            if isinstance(step, RunnableSequence):
                steps_flat.extend(step.steps)
            elif isinstance(step, RunnableSeq):
                steps_flat.extend(step.steps)
            else:
                steps_flat.append(coerce_to_runnable(step, name=None, trace=True))
        if len(steps_flat) < 2:
            raise ValueError(
                f"RunnableSeq must have at least 2 steps, got {len(steps_flat)}"
            )
        self.steps = steps_flat
        self.name = name
        self.trace_inputs = trace_inputs

    def __or__(
        self,
        other: Any,
    ) -> Runnable:
        if isinstance(other, RunnableSequence):
            return RunnableSeq(
                *self.steps,
                other.first,
                *other.middle,
                other.last,
                name=self.name or other.name,
            )
        elif isinstance(other, RunnableSeq):
            return RunnableSeq(
                *self.steps,
                *other.steps,
                name=self.name or other.name,
            )
        else:
            return RunnableSeq(
                *self.steps,
                coerce_to_runnable(other, name=None, trace=True),
                name=self.name,
            )

    def __ror__(
        self,
        other: Any,
    ) -> Runnable:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                other.first,
                *other.middle,
                other.last,
                *self.steps,
                name=other.name or self.name,
            )
        elif isinstance(other, RunnableSeq):
            return RunnableSeq(
                *other.steps,
                *self.steps,
                name=other.name or self.name,
            )
        else:
            return RunnableSequence(
                coerce_to_runnable(other, name=None, trace=True),
                *self.steps,
                name=self.name,
            )

    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        if config is None:
            config = ensure_config()
        # 콜백과 컨텍스트 설정
        callback_manager = get_callback_manager_for_config(config)
        # 루트 실행 시작
        run_manager = callback_manager.on_chain_start(
            None,
            self.trace_inputs(input) if self.trace_inputs is not None else input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        # 모든 단계를 순차적으로 실행
        try:
            for i, step in enumerate(self.steps):
                # 각 단계를 자식 실행으로 표시
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i + 1}")
                )
                # 첫 번째 단계는 실제 노드이고,
                # 나머지는 컨텍스트에서 실행할 필요가 없는 writer입니다
                if i == 0:
                    # run 객체 가져오기
                    for h in run_manager.handlers:
                        if isinstance(h, LangChainTracer):
                            run = h.run_map.get(str(run_manager.run_id))
                            break
                    else:
                        run = None
                    # 컨텍스트에서 실행
                    with set_config_context(config, run) as context:
                        input = context.run(step.invoke, input, config, **kwargs)
                else:
                    input = step.invoke(input, config)
        # 루트 실행 종료
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(input)
            return input

    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Any:
        if config is None:
            config = ensure_config()
        # 콜백 설정
        callback_manager = get_async_callback_manager_for_config(config)
        # 루트 실행 시작
        run_manager = await callback_manager.on_chain_start(
            None,
            self.trace_inputs(input) if self.trace_inputs is not None else input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # 모든 단계를 순차적으로 실행
        try:
            for i, step in enumerate(self.steps):
                # 각 단계를 자식 실행으로 표시
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i + 1}")
                )
                # 첫 번째 단계는 실제 노드이고,
                # 나머지는 컨텍스트에서 실행할 필요가 없는 writer입니다
                if i == 0:
                    if ASYNCIO_ACCEPTS_CONTEXT:
                        # run 객체 가져오기
                        for h in run_manager.handlers:
                            if isinstance(h, LangChainTracer):
                                run = h.run_map.get(str(run_manager.run_id))
                                break
                        else:
                            run = None
                        # 컨텍스트에서 실행
                        with set_config_context(config, run) as context:
                            input = await asyncio.create_task(
                                step.ainvoke(input, config, **kwargs), context=context
                            )
                    else:
                        input = await step.ainvoke(input, config, **kwargs)
                else:
                    input = await step.ainvoke(input, config)
        # 루트 실행 종료
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(input)
            return input

    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        if config is None:
            config = ensure_config()
        # 콜백 설정
        callback_manager = get_callback_manager_for_config(config)
        # 루트 실행 시작
        run_manager = callback_manager.on_chain_start(
            None,
            self.trace_inputs(input) if self.trace_inputs is not None else input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        # run 객체 가져오기
        for h in run_manager.handlers:
            if isinstance(h, LangChainTracer):
                run = h.run_map.get(str(run_manager.run_id))
                break
        else:
            run = None
        # 첫 번째 단계 config 생성
        config = patch_config(
            config,
            callbacks=run_manager.get_child(f"seq:step:{1}"),
        )
        # 모두 컨텍스트에서 실행
        with set_config_context(config, run) as context:
            try:
                # 마지막 단계들을 스트림
                # 각 단계의 입력 스트림을 다음 단계로 변환
                # 입력 스트림 변환을 기본적으로 지원하지 않는 단계는
                # 사용 가능한 모든 입력을 메모리에 버퍼링한 다음 출력을 내보냅니다
                for idx, step in enumerate(self.steps):
                    if idx == 0:
                        iterator = step.stream(input, config, **kwargs)
                    else:
                        config = patch_config(
                            config,
                            callbacks=run_manager.get_child(f"seq:step:{idx + 1}"),
                        )
                        iterator = step.transform(iterator, config)
                # 필요한 경우 astream_log() 출력에서 streamed_output을 채웁니다
                if _StreamingCallbackHandler is not None:
                    for h in run_manager.handlers:
                        if isinstance(h, _StreamingCallbackHandler):
                            iterator = h.tap_output_iter(run_manager.run_id, iterator)
                # 최종 출력으로 소비
                output = context.run(_consume_iter, iterator)
                # 시퀀스는 출력을 내보내지 않으므로, 제너레이터로 표시하기 위해 yield
                yield
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                run_manager.on_chain_end(output)

    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        if config is None:
            config = ensure_config()
        # 콜백 설정
        callback_manager = get_async_callback_manager_for_config(config)
        # 루트 실행 시작
        run_manager = await callback_manager.on_chain_start(
            None,
            self.trace_inputs(input) if self.trace_inputs is not None else input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        # 마지막 단계들을 스트림
        # 각 단계의 입력 스트림을 다음 단계로 변환
        # 입력 스트림 변환을 기본적으로 지원하지 않는 단계는
        # 사용 가능한 모든 입력을 메모리에 버퍼링한 다음 출력을 내보냅니다
        if ASYNCIO_ACCEPTS_CONTEXT:
            # run 객체 가져오기
            for h in run_manager.handlers:
                if isinstance(h, LangChainTracer):
                    run = h.run_map.get(str(run_manager.run_id))
                    break
            else:
                run = None
            # 첫 번째 단계 config 생성
            config = patch_config(
                config,
                callbacks=run_manager.get_child(f"seq:step:{1}"),
            )
            # 모두 컨텍스트에서 실행
            with set_config_context(config, run) as context:
                try:
                    async with AsyncExitStack() as stack:
                        for idx, step in enumerate(self.steps):
                            if idx == 0:
                                aiterator = step.astream(input, config, **kwargs)
                            else:
                                config = patch_config(
                                    config,
                                    callbacks=run_manager.get_child(
                                        f"seq:step:{idx + 1}"
                                    ),
                                )
                                aiterator = step.atransform(aiterator, config)
                            if hasattr(aiterator, "aclose"):
                                stack.push_async_callback(aiterator.aclose)
                        # 필요한 경우 astream_log() 출력에서 streamed_output을 채웁니다
                        if _StreamingCallbackHandler is not None:
                            for h in run_manager.handlers:
                                if isinstance(h, _StreamingCallbackHandler):
                                    aiterator = h.tap_output_aiter(
                                        run_manager.run_id, aiterator
                                    )
                        # 최종 출력으로 소비
                        output = await asyncio.create_task(
                            _consume_aiter(aiterator), context=context
                        )
                        # 시퀀스는 출력을 내보내지 않으므로, 제너레이터로 표시하기 위해 yield
                        yield
                except BaseException as e:
                    await run_manager.on_chain_error(e)
                    raise
                else:
                    await run_manager.on_chain_end(output)
        else:
            try:
                async with AsyncExitStack() as stack:
                    for idx, step in enumerate(self.steps):
                        config = patch_config(
                            config,
                            callbacks=run_manager.get_child(f"seq:step:{idx + 1}"),
                        )
                        if idx == 0:
                            aiterator = step.astream(input, config, **kwargs)
                        else:
                            aiterator = step.atransform(aiterator, config)
                        if hasattr(aiterator, "aclose"):
                            stack.push_async_callback(aiterator.aclose)
                    # 필요한 경우 astream_log() 출력에서 streamed_output을 채웁니다
                    if _StreamingCallbackHandler is not None:
                        for h in run_manager.handlers:
                            if isinstance(h, _StreamingCallbackHandler):
                                aiterator = h.tap_output_aiter(
                                    run_manager.run_id, aiterator
                                )
                    # 최종 출력으로 소비
                    output = await _consume_aiter(aiterator)
                    # 시퀀스는 출력을 내보내지 않으므로, 제너레이터로 표시하기 위해 yield
                    yield
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise
            else:
                await run_manager.on_chain_end(output)


def _consume_iter(it: Iterator[Any]) -> Any:
    """이터레이터를 소비합니다."""
    output: Any = None
    add_supported = False
    for chunk in it:
        # 최종 출력을 수집합니다
        if output is None:
            output = chunk
        elif add_supported:
            try:
                output = output + chunk
            except TypeError:
                output = chunk
                add_supported = False
        else:
            output = chunk
    return output


async def _consume_aiter(it: AsyncIterator[Any]) -> Any:
    """비동기 이터레이터를 소비합니다."""
    output: Any = None
    add_supported = False
    async for chunk in it:
        # 최종 출력을 수집합니다
        if add_supported:
            try:
                output = output + chunk
            except TypeError:
                output = chunk
                add_supported = False
        else:
            output = chunk
    return output
