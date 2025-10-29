from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
import sys
import types
from collections.abc import Awaitable, Coroutine, Generator
from typing import TypeVar, cast

T = TypeVar("T")
AnyFuture = asyncio.Future | concurrent.futures.Future

CONTEXT_NOT_SUPPORTED = sys.version_info < (3, 11)
EAGER_NOT_SUPPORTED = sys.version_info < (3, 12)


def _get_loop(fut: asyncio.Future) -> asyncio.AbstractEventLoop:
    # 가능한 경우 Future.get_loop()을 호출하려고 시도합니다.
    # 그렇지 않으면 이전 '_loop' 속성을 사용합니다.
    try:
        get_loop = fut.get_loop
    except AttributeError:
        pass
    else:
        return get_loop()
    return fut._loop


def _convert_future_exc(exc: BaseException) -> BaseException:
    exc_class = type(exc)
    if exc_class is concurrent.futures.CancelledError:
        return asyncio.CancelledError(*exc.args)
    elif exc_class is concurrent.futures.TimeoutError:
        return asyncio.TimeoutError(*exc.args)
    elif exc_class is concurrent.futures.InvalidStateError:
        return asyncio.InvalidStateError(*exc.args)
    else:
        return exc


def _set_concurrent_future_state(
    concurrent: concurrent.futures.Future,
    source: AnyFuture,
) -> None:
    """future의 상태를 concurrent.futures.Future로 복사합니다."""
    assert source.done()
    if source.cancelled():
        concurrent.cancel()
    if not concurrent.set_running_or_notify_cancel():
        return
    exception = source.exception()
    if exception is not None:
        concurrent.set_exception(_convert_future_exc(exception))
    else:
        result = source.result()
        concurrent.set_result(result)


def _copy_future_state(source: AnyFuture, dest: asyncio.Future) -> None:
    """다른 Future에서 상태를 복사하는 내부 헬퍼입니다.

    다른 Future는 concurrent.futures.Future일 수 있습니다.
    """
    if dest.done():
        return
    assert source.done()
    if dest.cancelled():
        return
    if source.cancelled():
        dest.cancel()
    else:
        exception = source.exception()
        if exception is not None:
            dest.set_exception(_convert_future_exc(exception))
        else:
            result = source.result()
            dest.set_result(result)


def _chain_future(source: AnyFuture, destination: AnyFuture) -> None:
    """두 future를 연결하여 하나가 완료되면 다른 하나도 완료되도록 합니다.

    source의 결과(또는 예외)가 destination으로 복사됩니다.
    destination이 취소되면 source도 취소됩니다.
    asyncio.Future와 concurrent.futures.Future 모두와 호환됩니다.
    """
    if not asyncio.isfuture(source) and not isinstance(
        source, concurrent.futures.Future
    ):
        raise TypeError("A future is required for source argument")
    if not asyncio.isfuture(destination) and not isinstance(
        destination, concurrent.futures.Future
    ):
        raise TypeError("A future is required for destination argument")
    source_loop = _get_loop(source) if asyncio.isfuture(source) else None
    dest_loop = _get_loop(destination) if asyncio.isfuture(destination) else None

    def _set_state(future: AnyFuture, other: AnyFuture) -> None:
        if asyncio.isfuture(future):
            _copy_future_state(other, future)
        else:
            _set_concurrent_future_state(future, other)

    def _call_check_cancel(destination: AnyFuture) -> None:
        if destination.cancelled():
            if source_loop is None or source_loop is dest_loop:
                source.cancel()
            else:
                source_loop.call_soon_threadsafe(source.cancel)

    def _call_set_state(source: AnyFuture) -> None:
        if destination.cancelled() and dest_loop is not None and dest_loop.is_closed():
            return
        if dest_loop is None or dest_loop is source_loop:
            _set_state(destination, source)
        else:
            if dest_loop.is_closed():
                return
            dest_loop.call_soon_threadsafe(_set_state, destination, source)

    destination.add_done_callback(_call_check_cancel)
    source.add_done_callback(_call_set_state)


def chain_future(source: AnyFuture, destination: AnyFuture) -> AnyFuture:
    # asyncio.run_coroutine_threadsafe에서 적응했습니다
    try:
        _chain_future(source, destination)
        return destination
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        if isinstance(destination, concurrent.futures.Future):
            if destination.set_running_or_notify_cancel():
                destination.set_exception(exc)
        else:
            destination.set_exception(exc)
        raise


def _ensure_future(
    coro_or_future: Coroutine[None, None, T] | Awaitable[T],
    *,
    loop: asyncio.AbstractEventLoop,
    name: str | None = None,
    context: contextvars.Context | None = None,
    lazy: bool = True,
) -> asyncio.Task[T]:
    called_wrap_awaitable = False
    if not asyncio.iscoroutine(coro_or_future):
        if inspect.isawaitable(coro_or_future):
            coro_or_future = cast(
                Coroutine[None, None, T], _wrap_awaitable(coro_or_future)
            )
            called_wrap_awaitable = True
        else:
            raise TypeError(
                "An asyncio.Future, a coroutine or an awaitable is required."
                f" Got {type(coro_or_future).__name__} instead."
            )

    try:
        if CONTEXT_NOT_SUPPORTED:
            return loop.create_task(coro_or_future, name=name)
        elif EAGER_NOT_SUPPORTED or lazy:
            return loop.create_task(coro_or_future, name=name, context=context)
        else:
            return asyncio.eager_task_factory(
                loop, coro_or_future, name=name, context=context
            )
    except RuntimeError:
        if not called_wrap_awaitable:
            coro_or_future.close()
        raise


@types.coroutine
def _wrap_awaitable(awaitable: Awaitable[T]) -> Generator[None, None, T]:
    """asyncio.ensure_future()를 위한 헬퍼입니다.

    awaitable(__await__이 있는 객체)을 코루틴으로 래핑하여
    나중에 ensure_future()에 의해 Task로 래핑됩니다.
    """
    return (yield from awaitable.__await__())


def run_coroutine_threadsafe(
    coro: Coroutine[None, None, T],
    loop: asyncio.AbstractEventLoop,
    *,
    lazy: bool,
    name: str | None = None,
    context: contextvars.Context | None = None,
) -> asyncio.Future[T]:
    """주어진 이벤트 루프에 코루틴 객체를 제출합니다.

    결과에 접근하기 위한 asyncio.Future를 반환합니다.
    """

    if asyncio._get_running_loop() is loop:
        return _ensure_future(coro, loop=loop, name=name, context=context, lazy=lazy)
    else:
        future: asyncio.Future[T] = asyncio.Future(loop=loop)

        def callback() -> None:
            try:
                chain_future(
                    _ensure_future(coro, loop=loop, name=name, context=context),
                    future,
                )
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:
                future.set_exception(exc)
                raise

        loop.call_soon_threadsafe(callback, context=context)
        return future
