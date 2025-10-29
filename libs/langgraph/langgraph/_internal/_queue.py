# type: ignore
from __future__ import annotations

import asyncio
import queue
import threading
import types
from collections import deque
from time import monotonic


class AsyncQueue(asyncio.Queue):
    """wait() 메서드가 있는 비동기 무제한 FIFO 큐입니다.

    asyncio.Queue를 서브클래싱하여 wait() 메서드를 추가했습니다."""

    async def wait(self) -> None:
        """큐가 비어있으면 아이템이 사용 가능할 때까지 기다립니다.

        Queue.get()에서 복사했으며, .get_nowait() 호출을 제거했습니다.
        즉, 아이템을 소비하지 않고 단지 기다리기만 합니다.
        """
        while self.empty():
            getter = self._get_loop().create_future()
            self._getters.append(getter)
            try:
                await getter
            except:
                getter.cancel()  # getter가 아직 완료되지 않은 경우를 대비합니다.
                try:
                    # 취소된 getter를 self._getters에서 제거합니다.
                    self._getters.remove(getter)
                except ValueError:
                    # getter가 이전 put_nowait 호출에 의해
                    # self._getters에서 제거되었을 수 있습니다.
                    pass
                if not self.empty() and not getter.cancelled():
                    # put_nowait()에 의해 깨어났지만 호출을 받을 수 없습니다.
                    # 대기 중인 다음 항목을 깨웁니다.
                    self._wakeup_next(self._getters)
                raise


class Semaphore(threading.Semaphore):
    """wait() 메서드가 있는 Semaphore 서브클래스입니다."""

    def wait(self, blocking: bool = True, timeout: float | None = None):
        """세마포어를 획득할 수 있을 때까지 블록하지만, 획득하지는 않습니다."""
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        rc = False
        endtime = None
        with self._cond:
            while self._value == 0:
                if not blocking:
                    break
                if timeout is not None:
                    if endtime is None:
                        endtime = monotonic() + timeout
                    else:
                        timeout = endtime - monotonic()
                        if timeout <= 0:
                            break
                self._cond.wait(timeout)
            else:
                rc = True
        return rc


class SyncQueue:
    """wait() 메서드가 있는 무제한 FIFO 큐입니다.
    queue.SimpleQueue의 순수 Python 구현에서 적응했습니다.
    """

    def __init__(self):
        self._queue = deque()
        self._count = Semaphore(0)

    def put(self, item, block=True, timeout=None):
        """아이템을 큐에 넣습니다.

        선택적 'block'과 'timeout' 인자는 무시됩니다. 이 메서드는 절대 블록하지 않습니다.
        이들은 Queue 클래스와의 호환성을 위해 제공됩니다.
        """
        self._queue.append(item)
        self._count.release()

    def get(self, block=False, timeout=None):
        """큐에서 아이템을 제거하고 반환합니다.

        선택적 인자 'block'이 true이고 'timeout'이 None(기본값)이면,
        필요한 경우 아이템이 사용 가능할 때까지 블록합니다. 'timeout'이
        음이 아닌 숫자이면, 최대 'timeout'초 동안 블록하며 그 시간 내에
        아이템이 사용 가능하지 않으면 Empty 예외를 발생시킵니다.
        그렇지 않으면('block'이 false), 즉시 사용 가능한 아이템이 있으면
        반환하고, 그렇지 않으면 Empty 예외를 발생시킵니다('timeout'은
        이 경우 무시됩니다).
        """
        if timeout is not None and timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        if not self._count.acquire(block, timeout):
            raise queue.Empty
        try:
            return self._queue.popleft()
        except IndexError:
            raise queue.Empty

    def wait(self, block=True, timeout=None):
        """큐가 비어있으면 아이템이 사용 가능할 때까지 기다리지만,
        소비하지는 않습니다.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        self._count.wait(block, timeout)

    def empty(self):
        """큐가 비어있으면 True를 반환하고, 그렇지 않으면 False를 반환합니다 (신뢰할 수 없음!)."""
        return len(self._queue) == 0

    def qsize(self):
        """큐의 대략적인 크기를 반환합니다 (신뢰할 수 없음!)."""
        return len(self._queue)

    __class_getitem__ = classmethod(types.GenericAlias)
