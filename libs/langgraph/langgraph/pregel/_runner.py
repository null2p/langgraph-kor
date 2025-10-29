from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
import time
import weakref
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from functools import partial
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
)

from langchain_core.callbacks import Callbacks

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CALL,
    CONFIG_KEY_SCRATCHPAD,
    ERROR,
    INTERRUPT,
    NO_WRITES,
    RESUME,
    RETURN,
)
from langgraph._internal._future import chain_future, run_coroutine_threadsafe
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph._internal._typing import MISSING
from langgraph.constants import TAG_HIDDEN
from langgraph.errors import GraphBubbleUp, GraphInterrupt
from langgraph.pregel._algo import Call
from langgraph.pregel._executor import Submit
from langgraph.pregel._retry import arun_with_retry, run_with_retry
from langgraph.types import (
    CachePolicy,
    PregelExecutableTask,
    RetryPolicy,
)

F = TypeVar("F", concurrent.futures.Future, asyncio.Future)
E = TypeVar("E", threading.Event, asyncio.Event)

# 예외 traceback에서 제외할 파일 이름 목록
# 참고: 프레임이 traceback의 마지막 프레임인 경우 재귀적으로 제거됩니다
EXCLUDED_FRAME_FNAMES = (
    "langgraph/pregel/retry.py",
    "langgraph/pregel/runner.py",
    "langgraph/pregel/executor.py",
    "langgraph/utils/runnable.py",
    "langchain_core/runnables/config.py",
    "concurrent/futures/thread.py",
    "concurrent/futures/_base.py",
)

SKIP_RERAISE_SET: weakref.WeakSet[concurrent.futures.Future | asyncio.Future] = (
    weakref.WeakSet()
)


class FuturesDict(Generic[F, E], dict[F, PregelExecutableTask | None]):
    event: E
    callback: weakref.ref[Callable[[PregelExecutableTask, BaseException | None], None]]
    counter: int
    done: set[F]
    lock: threading.Lock

    def __init__(
        self,
        event: E,
        callback: weakref.ref[
            Callable[[PregelExecutableTask, BaseException | None], None]
        ],
        future_type: type[F],
        # used for generic typing, newer py supports FutureDict[...](...)
    ) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.event = event
        self.callback = callback
        self.counter = 0
        self.done: set[F] = set()

    def __setitem__(
        self,
        key: F,
        value: PregelExecutableTask | None,
    ) -> None:
        super().__setitem__(key, value)  # type: ignore[index]
        if value is not None:
            with self.lock:
                self.event.clear()
                self.counter += 1
            key.add_done_callback(partial(self.on_done, value))

    def on_done(
        self,
        task: PregelExecutableTask,
        fut: F,
    ) -> None:
        try:
            if cb := self.callback():
                cb(task, _exception(fut))
        finally:
            with self.lock:
                self.done.add(fut)
                self.counter -= 1
                if self.counter == 0 or _should_stop_others(self.done):
                    self.event.set()


class PregelRunner:
    """Pregel 태스크 집합을 동시에 실행하고, 쓰기를 커밋하며,
    출력할 내용이 있을 때 호출자에게 제어를 양도하고,
    적절한 경우 다른 태스크를 중단하는 역할을 합니다."""

    def __init__(
        self,
        *,
        submit: weakref.ref[Submit],
        put_writes: weakref.ref[Callable[[str, Sequence[tuple[str, Any]]], None]],
        use_astream: bool = False,
        node_finished: Callable[[str], None] | None = None,
    ) -> None:
        self.submit = submit
        self.put_writes = put_writes
        self.use_astream = use_astream
        self.node_finished = node_finished

    def tick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: float | None = None,
        retry_policy: Sequence[RetryPolicy] | None = None,
        get_waiter: Callable[[], concurrent.futures.Future[None]] | None = None,
        schedule_task: Callable[
            [PregelExecutableTask, int, Call | None],
            PregelExecutableTask | None,
        ],
    ) -> Iterator[None]:
        tasks = tuple(tasks)
        futures = FuturesDict(
            callback=weakref.WeakMethod(self.commit),
            event=threading.Event(),
            future_type=concurrent.futures.Future,
        )
        # 호출자에게 제어를 반환합니다
        yield
        # timeout과 waiter가 없는 단일 태스크인 경우 빠른 경로
        if len(tasks) == 0:
            return
        elif len(tasks) == 1 and timeout is None and get_waiter is None:
            t = tasks[0]
            try:
                run_with_retry(
                    t,
                    retry_policy,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _call,
                            weakref.ref(t),
                            retry_policy=retry_policy,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                        ),
                    },
                )
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, exc)
                if reraise and futures:
                    # future가 완료된 후에 다시 발생합니다
                    fut: concurrent.futures.Future = concurrent.futures.Future()
                    fut.set_exception(exc)
                    futures.done.add(fut)
                elif reraise:
                    if tb := exc.__traceback__:
                        while tb.tb_next is not None and any(
                            tb.tb_frame.f_code.co_filename.endswith(name)
                            for name in EXCLUDED_FRAME_FNAMES
                        ):
                            tb = tb.tb_next
                        exc.__traceback__ = tb
                    raise
            if not futures:  # `t`가 다른 태스크를 스케줄했을 수 있습니다
                return
            else:
                tasks = ()  # 이 태스크를 다시 스케줄하지 않습니다
        # 요청된 경우 waiter 태스크를 추가합니다
        if get_waiter is not None:
            futures[get_waiter()] = None
        # 태스크를 스케줄합니다
        for t in tasks:
            fut = self.submit()(  # type: ignore[misc]
                run_with_retry,
                t,
                retry_policy,
                configurable={
                    CONFIG_KEY_CALL: partial(
                        _call,
                        weakref.ref(t),
                        retry_policy=retry_policy,
                        futures=weakref.ref(futures),
                        schedule_task=schedule_task,
                        submit=self.submit,
                    ),
                },
                __reraise_on_exit__=reraise,
            )
            futures[fut] = t
        # 태스크를 실행하고, 하나가 실패하거나 모두 완료될 때까지 기다립니다.
        # 각 태스크는 다른 모든 동시 태스크로부터 독립적입니다
        # 각 태스크가 완료될 때마다 업데이트/디버그 출력을 yield합니다
        end_time = timeout + time.monotonic() if timeout else None
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=(max(0, end_time - time.monotonic()) if end_time else None),
            )
            if not done:
                break  # 타임아웃되었습니다
            for fut in done:
                task = futures.pop(fut)
                if task is None:
                    # waiter 태스크가 완료되었으므로 다른 태스크를 스케줄합니다
                    if inflight and get_waiter is not None:
                        futures[get_waiter()] = None
            else:
                # 루프 변수에 대한 참조를 제거합니다
                del fut, task
            # 다른 태스크를 중단할 수도 있습니다
            if _should_stop_others(done):
                break
            # 호출자에게 제어를 반환합니다
            yield
        # wait for done callbacks
        futures.event.wait(
            timeout=(max(0, end_time - time.monotonic()) if end_time else None)
        )
        # give control back to the caller
        yield
        # panic on failure or timeout
        try:
            _panic_or_proceed(
                futures.done.union(f for f, t in futures.items() if t is not None),
                panic=reraise,
            )
        except Exception as exc:
            if tb := exc.__traceback__:
                while tb.tb_next is not None and any(
                    tb.tb_frame.f_code.co_filename.endswith(name)
                    for name in EXCLUDED_FRAME_FNAMES
                ):
                    tb = tb.tb_next
                exc.__traceback__ = tb
            raise

    async def atick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: float | None = None,
        retry_policy: Sequence[RetryPolicy] | None = None,
        get_waiter: Callable[[], asyncio.Future[None]] | None = None,
        schedule_task: Callable[
            [PregelExecutableTask, int, Call | None],
            Awaitable[PregelExecutableTask | None],
        ],
    ) -> AsyncIterator[None]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = tuple(tasks)
        futures = FuturesDict(
            callback=weakref.WeakMethod(self.commit),
            event=asyncio.Event(),
            future_type=asyncio.Future,
        )
        # give control back to the caller
        yield
        # fast path if single task with no waiter and no timeout
        if len(tasks) == 0:
            return
        elif len(tasks) == 1 and get_waiter is None and timeout is None:
            t = tasks[0]
            try:
                await arun_with_retry(
                    t,
                    retry_policy,
                    stream=self.use_astream,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _acall,
                            weakref.ref(t),
                            stream=self.use_astream,
                            retry_policy=retry_policy,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                            loop=loop,
                        ),
                    },
                )
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, exc)
                if reraise and futures:
                    # will be re-raised after futures are done
                    fut: asyncio.Future = loop.create_future()
                    fut.set_exception(exc)
                    futures.done.add(fut)
                elif reraise:
                    if tb := exc.__traceback__:
                        while tb.tb_next is not None and any(
                            tb.tb_frame.f_code.co_filename.endswith(name)
                            for name in EXCLUDED_FRAME_FNAMES
                        ):
                            tb = tb.tb_next
                        exc.__traceback__ = tb
                    raise
            if not futures:  # `t`가 다른 태스크를 스케줄했을 수 있습니다
                return
            else:
                tasks = ()  # 이 태스크를 다시 스케줄하지 않습니다
        # 요청된 경우 waiter 태스크를 추가합니다
        if get_waiter is not None:
            futures[get_waiter()] = None
        # 태스크를 스케줄합니다
        for t in tasks:
            fut = cast(
                asyncio.Future,
                self.submit()(  # type: ignore[misc]
                    arun_with_retry,
                    t,
                    retry_policy,
                    stream=self.use_astream,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _acall,
                            weakref.ref(t),
                            retry_policy=retry_policy,
                            stream=self.use_astream,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                            loop=loop,
                        ),
                    },
                    __name__=t.name,
                    __cancel_on_exit__=True,
                    __reraise_on_exit__=reraise,
                ),
            )
            futures[fut] = t
        # 태스크를 실행하고, 하나가 실패하거나 모두 완료될 때까지 기다립니다.
        # 각 태스크는 다른 모든 동시 태스크로부터 독립적입니다
        # 각 태스크가 완료될 때마다 업데이트/디버그 출력을 yield합니다
        end_time = timeout + loop.time() if timeout else None
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = await asyncio.wait(
                futures,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=(max(0, end_time - loop.time()) if end_time else None),
            )
            if not done:
                break  # 타임아웃되었습니다
            for fut in done:
                task = futures.pop(fut)
                if task is None:
                    # waiter 태스크가 완료되었으므로 다른 태스크를 스케줄합니다
                    if inflight and get_waiter is not None:
                        futures[get_waiter()] = None
            else:
                # 루프 변수에 대한 참조를 제거합니다
                del fut, task
            # 다른 태스크를 중단할 수도 있습니다
            if _should_stop_others(done):
                break
            # 호출자에게 제어를 반환합니다
            yield
        # wait for done callbacks
        await asyncio.wait_for(
            futures.event.wait(),
            timeout=(max(0, end_time - loop.time()) if end_time else None),
        )
        # give control back to the caller
        yield
        # cancel waiter task
        for fut in futures:
            fut.cancel()
        # panic on failure or timeout
        try:
            _panic_or_proceed(
                futures.done.union(f for f, t in futures.items() if t is not None),
                timeout_exc_cls=asyncio.TimeoutError,
                panic=reraise,
            )
        except Exception as exc:
            if tb := exc.__traceback__:
                while tb.tb_next is not None and any(
                    tb.tb_frame.f_code.co_filename.endswith(name)
                    for name in EXCLUDED_FRAME_FNAMES
                ):
                    tb = tb.tb_next
                exc.__traceback__ = tb
            raise

    def commit(
        self,
        task: PregelExecutableTask,
        exception: BaseException | None,
    ) -> None:
        if isinstance(exception, asyncio.CancelledError):
            # for cancelled tasks, also save error in task,
            # so loop can finish super-step
            task.writes.append((ERROR, exception))
            self.put_writes()(task.id, task.writes)  # type: ignore[misc]
        elif exception:
            if isinstance(exception, GraphInterrupt):
                # save interrupt to checkpointer
                if exception.args[0]:
                    writes = [(INTERRUPT, exception.args[0])]
                    if resumes := [w for w in task.writes if w[0] == RESUME]:
                        writes.extend(resumes)
                    self.put_writes()(task.id, writes)  # type: ignore[misc]
            elif isinstance(exception, GraphBubbleUp):
                # exception will be raised in _panic_or_proceed
                pass
            else:
                # save error to checkpointer
                task.writes.append((ERROR, exception))
                self.put_writes()(task.id, task.writes)  # type: ignore[misc]
        else:
            if self.node_finished and (
                task.config is None or TAG_HIDDEN not in task.config.get("tags", [])
            ):
                self.node_finished(task.name)
            if not task.writes:
                # add no writes marker
                task.writes.append((NO_WRITES, None))
            # save task writes to checkpointer
            self.put_writes()(task.id, task.writes)  # type: ignore[misc]


def _should_stop_others(
    done: set[F],
) -> bool:
    """태스크가 실패했는지 확인하고, 그렇다면 다른 모든 태스크를 취소합니다.
    GraphInterrupt는 실패로 간주되지 않습니다."""
    for fut in done:
        if fut.cancelled():
            continue
        elif exc := fut.exception():
            if not isinstance(exc, GraphBubbleUp) and fut not in SKIP_RERAISE_SET:
                return True

    return False


def _exception(
    fut: concurrent.futures.Future[Any] | asyncio.Future[Any],
) -> BaseException | None:
    """CancelledError를 발생시키지 않고 future에서 예외를 반환합니다."""
    if fut.cancelled():
        if isinstance(fut, asyncio.Future):
            return asyncio.CancelledError()
        else:
            return concurrent.futures.CancelledError()
    else:
        return fut.exception()


def _panic_or_proceed(
    futs: set[concurrent.futures.Future] | set[asyncio.Future],
    *,
    timeout_exc_cls: type[Exception] = TimeoutError,
    panic: bool = True,
) -> None:
    """실패한 태스크가 있으면 나머지 태스크를 취소하고, panic이 True이면 예외를 다시 발생시킵니다."""
    done: set[concurrent.futures.Future[Any] | asyncio.Future[Any]] = set()
    inflight: set[concurrent.futures.Future[Any] | asyncio.Future[Any]] = set()
    for fut in futs:
        if fut.cancelled():
            continue
        elif fut.done():
            done.add(fut)
        else:
            inflight.add(fut)
    interrupts: list[GraphInterrupt] = []
    while done:
        # if any task failed
        fut = done.pop()
        if exc := _exception(fut):
            # cancel all pending tasks
            while inflight:
                inflight.pop().cancel()
            # raise the exception
            if panic:
                if isinstance(exc, GraphInterrupt):
                    # collect interrupts
                    interrupts.append(exc)
                elif fut not in SKIP_RERAISE_SET:
                    raise exc
    # raise combined interrupts
    if interrupts:
        raise GraphInterrupt(tuple(i for exc in interrupts for i in exc.args[0]))
    if inflight:
        # if we got here means we timed out
        while inflight:
            # cancel all pending tasks
            inflight.pop().cancel()
        # raise timeout error
        raise timeout_exc_cls("Timed out")


def _call(
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    callbacks: Callbacks = None,
    futures: weakref.ref[FuturesDict],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None], PregelExecutableTask | None
    ],
    submit: weakref.ref[Submit],
) -> concurrent.futures.Future[Any]:
    if inspect.iscoroutinefunction(func):
        raise RuntimeError("In an sync context async tasks cannot be called")

    fut: concurrent.futures.Future | None = None
    # schedule PUSH tasks, collect futures
    scratchpad: PregelScratchpad = task().config[CONF][CONFIG_KEY_SCRATCHPAD]  # type: ignore[union-attr]
    # schedule the next task, if the callback returns one
    if next_task := schedule_task(
        task(),  # type: ignore[arg-type]
        scratchpad.call_counter(),
        Call(
            func,
            input,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            callbacks=callbacks,
        ),
    ):
        if fut := next(
            (
                f
                for f, t in futures().items()  # type: ignore[union-attr]
                if t is not None and t == next_task.id
            ),
            None,
        ):
            # if the parent task was retried,
            # the next task might already be running
            pass
        elif next_task.writes:
            # if it already ran, return the result
            fut = concurrent.futures.Future()
            ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
            if ret is not MISSING:
                fut.set_result(ret)
            elif exc := next((v for c, v in next_task.writes if c == ERROR), None):
                fut.set_exception(
                    exc if isinstance(exc, BaseException) else Exception(exc)
                )
            else:
                fut.set_result(None)
        else:
            # schedule the next task
            fut = submit()(  # type: ignore[misc]
                run_with_retry,
                next_task,
                retry_policy,
                configurable={
                    CONFIG_KEY_CALL: partial(
                        _call,
                        weakref.ref(next_task),
                        futures=futures,
                        retry_policy=retry_policy,
                        callbacks=callbacks,
                        schedule_task=schedule_task,
                        submit=submit,
                    ),
                },
                __reraise_on_exit__=False,
                # starting a new task in the next tick ensures
                # updates from this tick are committed/streamed first
                __next_tick__=True,
            )
            # exceptions for call() tasks are raised into the parent task
            # so we should not re-raise at the end of the tick
            SKIP_RERAISE_SET.add(fut)
            futures()[fut] = next_task  # type: ignore[index]
    fut = cast(asyncio.Future | concurrent.futures.Future, fut)
    # return a chained future to ensure commit() callback is called
    # before the returned future is resolved, to ensure stream order etc
    return chain_future(fut, concurrent.futures.Future())


def _acall(
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    callbacks: Callbacks = None,
    # injected dependencies
    futures: weakref.ref[FuturesDict],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None],
        Awaitable[PregelExecutableTask | None],
    ],
    submit: weakref.ref[Submit],
    loop: asyncio.AbstractEventLoop,
    stream: bool = False,
) -> asyncio.Future[Any] | concurrent.futures.Future[Any]:
    # return a chained future to ensure commit() callback is called
    # before the returned future is resolved, to ensure stream order etc
    try:
        in_async = asyncio.current_task() is not None
    except RuntimeError:
        in_async = False
    # if in async context return an async future, otherwise return a sync future
    if in_async:
        fut: asyncio.Future[Any] | concurrent.futures.Future[Any] = asyncio.Future(
            loop=loop
        )
    else:
        fut = concurrent.futures.Future()
    # schedule the next task
    run_coroutine_threadsafe(
        _acall_impl(
            fut,
            task,
            func,
            input,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            callbacks=callbacks,
            futures=futures,
            schedule_task=schedule_task,
            submit=submit,
            loop=loop,
            stream=stream,
        ),
        loop,
        lazy=False,
    )
    return fut


async def _acall_impl(
    destination: asyncio.Future[Any] | concurrent.futures.Future[Any],
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    callbacks: Callbacks = None,
    # injected dependencies
    futures: weakref.ref[FuturesDict[asyncio.Future, asyncio.Event]],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None],
        Awaitable[PregelExecutableTask | None],
    ],
    submit: weakref.ref[Submit],
    loop: asyncio.AbstractEventLoop,
    stream: bool = False,
) -> None:
    try:
        fut: asyncio.Future | None = None
        # schedule PUSH tasks, collect futures
        scratchpad: PregelScratchpad = task().config[CONF][CONFIG_KEY_SCRATCHPAD]  # type: ignore[union-attr]
        # schedule the next task, if the callback returns one
        if next_task := await schedule_task(
            task(),  # type: ignore[arg-type]
            scratchpad.call_counter(),
            Call(
                func,
                input,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                callbacks=callbacks,
            ),
        ):
            if fut := next(
                (
                    f
                    for f, t in futures().items()  # type: ignore[union-attr]
                    if t is not None and t == next_task.id
                ),
                None,
            ):
                # if the parent task was retried,
                # the next task might already be running
                pass
            elif next_task.writes:
                # if it already ran, return the result
                fut = asyncio.Future(loop=loop)
                ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
                if ret is not MISSING:
                    fut.set_result(ret)
                elif exc := next((v for c, v in next_task.writes if c == ERROR), None):
                    fut.set_exception(
                        exc if isinstance(exc, BaseException) else Exception(exc)
                    )
                else:
                    fut.set_result(None)
                futures()[fut] = next_task  # type: ignore[index]
            else:
                # schedule the next task
                fut = cast(
                    asyncio.Future,
                    submit()(  # type: ignore[misc]
                        arun_with_retry,
                        next_task,
                        retry_policy,
                        stream=stream,
                        configurable={
                            CONFIG_KEY_CALL: partial(
                                _acall,
                                weakref.ref(next_task),
                                stream=stream,
                                futures=futures,
                                schedule_task=schedule_task,
                                submit=submit,
                                loop=loop,
                            ),
                        },
                        __name__=next_task.name,
                        __cancel_on_exit__=True,
                        __reraise_on_exit__=False,
                        # starting a new task in the next tick ensures
                        # updates from this tick are committed/streamed first
                        __next_tick__=True,
                    ),
                )
                # exceptions for call() tasks are raised into the parent task
                # so we should not re-raise at the end of the tick
                SKIP_RERAISE_SET.add(fut)
                futures()[fut] = next_task  # type: ignore[index]
        if fut is not None:
            chain_future(fut, destination)
        else:
            destination.set_exception(RuntimeError("Task not scheduled"))
    except Exception as exc:
        destination.set_exception(exc)
