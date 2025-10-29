"""백그라운드 작업에서 작업을 배치 처리하기 위한 유틸리티입니다."""

from __future__ import annotations

import asyncio
import functools
import weakref
from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeVar

from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    NamespacePath,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    _ensure_refresh,
    _ensure_ttl,
    _validate_namespace,
)

F = TypeVar("F", bound=Callable)


def _check_loop(func: F) -> F:
    @functools.wraps(func)
    def wrapper(store: AsyncBatchedBaseStore, *args: Any, **kwargs: Any) -> Any:
        method_name: str = func.__name__
        try:
            current_loop = asyncio.get_running_loop()
            if current_loop is store._loop:
                replacement_str = (
                    f"Specifically, replace `store.{method_name}(...)` with `await store.a{method_name}(...)"
                    if method_name
                    else "For example, replace `store.get(...)` with `await store.aget(...)`"
                )
                raise asyncio.InvalidStateError(
                    f"Synchronous calls to {store.__class__.__name__} detected in the main event loop. "
                    "This can lead to deadlocks or performance issues. "
                    "Please use the asynchronous interface for main thread operations. "
                    f"{replacement_str} "
                )
        except RuntimeError:
            pass
        return func(store, *args, **kwargs)

    return wrapper


class AsyncBatchedBaseStore(BaseStore):
    """백그라운드 작업에서 작업을 효율적으로 배치 처리합니다."""

    __slots__ = ("_loop", "_aqueue", "_task")

    def __init__(self) -> None:
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self._aqueue: asyncio.Queue[tuple[asyncio.Future, Op]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ensure_task()

    def __del__(self) -> None:
        try:
            if self._task:
                self._task.cancel()
        except RuntimeError:
            pass

    def _ensure_task(self) -> None:
        """백그라운드 처리 루프가 실행 중인지 확인합니다."""
        if self._task is None or self._task.done():
            self._task = self._loop.create_task(_run(self._aqueue, weakref.ref(self)))

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        self._ensure_task()
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                GetOp(
                    namespace,
                    key,
                    refresh_ttl=_ensure_refresh(self.ttl_config, refresh_ttl),
                ),
            )
        )
        return await fut

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        self._ensure_task()
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                SearchOp(
                    namespace_prefix,
                    filter,
                    limit,
                    offset,
                    query,
                    refresh_ttl=_ensure_refresh(self.ttl_config, refresh_ttl),
                ),
            )
        )
        return await fut

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        self._ensure_task()
        _validate_namespace(namespace)
        fut = self._loop.create_future()
        self._aqueue.put_nowait(
            (
                fut,
                PutOp(
                    namespace, key, value, index, ttl=_ensure_ttl(self.ttl_config, ttl)
                ),
            )
        )
        return await fut

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        self._ensure_task()
        fut = self._loop.create_future()
        self._aqueue.put_nowait((fut, PutOp(namespace, key, None)))
        return await fut

    async def alist_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        self._ensure_task()
        fut = self._loop.create_future()
        match_conditions = []
        if prefix:
            match_conditions.append(MatchCondition(match_type="prefix", path=prefix))
        if suffix:
            match_conditions.append(MatchCondition(match_type="suffix", path=suffix))

        op = ListNamespacesOp(
            match_conditions=tuple(match_conditions),
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
        self._aqueue.put_nowait((fut, op))
        return await fut

    @_check_loop
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return asyncio.run_coroutine_threadsafe(self.abatch(ops), self._loop).result()

    @_check_loop
    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return asyncio.run_coroutine_threadsafe(
            self.aget(namespace, key=key, refresh_ttl=refresh_ttl), self._loop
        ).result()

    @_check_loop
    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return asyncio.run_coroutine_threadsafe(
            self.asearch(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            ),
            self._loop,
        ).result()

    @_check_loop
    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        _validate_namespace(namespace)
        asyncio.run_coroutine_threadsafe(
            self.aput(
                namespace,
                key=key,
                value=value,
                index=index,
                ttl=_ensure_ttl(self.ttl_config, ttl),
            ),
            self._loop,
        ).result()

    @_check_loop
    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        asyncio.run_coroutine_threadsafe(
            self.adelete(namespace, key=key), self._loop
        ).result()

    @_check_loop
    def list_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return asyncio.run_coroutine_threadsafe(
            self.alist_namespaces(
                prefix=prefix,
                suffix=suffix,
                max_depth=max_depth,
                limit=limit,
                offset=offset,
            ),
            self._loop,
        ).result()


def _dedupe_ops(values: list[Op]) -> tuple[list[int] | None, list[Op]]:
    """결과의 순서를 유지하면서 작업을 중복 제거합니다.

    Args:
        values: 중복 제거할 작업 목록

    Returns:
        (listen 인덱스, 중복 제거된 작업)의 튜플
        여기서 listen 인덱스는 중복 제거된 작업 결과를 원래 위치로 다시 매핑합니다
    """
    if len(values) <= 1:
        return None, list(values)

    dedupped: list[Op] = []
    listen: list[int] = []
    puts: dict[tuple[tuple[str, ...], str], int] = {}

    for op in values:
        if isinstance(op, (GetOp, SearchOp, ListNamespacesOp)):
            try:
                listen.append(dedupped.index(op))
            except ValueError:
                listen.append(len(dedupped))
                dedupped.append(op)
        elif isinstance(op, PutOp):
            putkey = (op.namespace, op.key)
            if putkey in puts:
                # 이전 put 덮어쓰기
                ix = puts[putkey]
                dedupped[ix] = op
                listen.append(ix)
            else:
                puts[putkey] = len(dedupped)
                listen.append(len(dedupped))
                dedupped.append(op)

        else:  # 새로운 작업은 정기적으로 처리됩니다
            listen.append(len(dedupped))
            dedupped.append(op)

    return listen, dedupped


async def _run(
    aqueue: asyncio.Queue[tuple[asyncio.Future, Op]],
    store: weakref.ReferenceType[BaseStore],
) -> None:
    while item := await aqueue.get():
        # 저장소가 아직 살아있는지 확인
        if s := store():
            try:
                # 동일한 틱에서 예약된 작업 누적
                items = [item]
                try:
                    while item := aqueue.get_nowait():
                        items.append(item)
                except asyncio.QueueEmpty:
                    pass
                # 실행할 작업 가져오기
                futs = [item[0] for item in items]
                values = [item[1] for item in items]
                # 각 작업 실행
                try:
                    listen, dedupped = _dedupe_ops(values)
                    results = await s.abatch(dedupped)
                    if listen is not None:
                        results = [results[ix] for ix in listen]

                    # 각 작업의 결과 설정
                    for fut, result in zip(futs, results, strict=False):
                        # future가 완료되지 않았는지 확인 (예: 취소됨)
                        if not fut.done():
                            fut.set_result(result)
                except Exception as e:
                    for fut in futs:
                        # future가 완료되지 않았는지 확인 (예: 취소됨)
                        if not fut.done():
                            fut.set_exception(e)
            finally:
                # 저장소에 대한 강한 참조 제거
                del s
        else:
            break
