"""다음에서 수정됨:
https://github.com/oittaa/uuid6-python/blob/main/src/uuid6/__init__.py#L95
uuid6 패키지의 설치 문제를 피하기 위해 번들로 제공됩니다
"""

from __future__ import annotations

import random
import time
import uuid

_last_v6_timestamp = None


class UUID(uuid.UUID):
    r"""UUID 드래프트 버전 객체"""

    __slots__ = ()

    def __init__(
        self,
        hex: str | None = None,
        bytes: bytes | None = None,
        bytes_le: bytes | None = None,
        fields: tuple[int, int, int, int, int, int] | None = None,
        int: int | None = None,
        version: int | None = None,
        *,
        is_safe: uuid.SafeUUID = uuid.SafeUUID.unknown,
    ) -> None:
        r"""UUID를 생성합니다."""

        if int is None or [hex, bytes, bytes_le, fields].count(None) != 4:
            return super().__init__(
                hex=hex,
                bytes=bytes,
                bytes_le=bytes_le,
                fields=fields,
                int=int,
                version=version,
                is_safe=is_safe,
            )
        if not 0 <= int < 1 << 128:
            raise ValueError("int가 범위를 벗어났습니다 (128비트 값이 필요합니다)")
        if version is not None:
            if not 6 <= version <= 8:
                raise ValueError("잘못된 버전 번호입니다")
            # variant를 RFC 4122로 설정합니다.
            int &= ~(0xC000 << 48)
            int |= 0x8000 << 48
            # 버전 번호를 설정합니다.
            int &= ~(0xF000 << 64)
            int |= version << 76
        super().__init__(int=int, is_safe=is_safe)

    @property
    def subsec(self) -> int:
        return ((self.int >> 64) & 0x0FFF) << 8 | ((self.int >> 54) & 0xFF)

    @property
    def time(self) -> int:
        if self.version == 6:
            return (
                (self.time_low << 28)
                | (self.time_mid << 12)
                | (self.time_hi_version & 0x0FFF)
            )
        if self.version == 7:
            return self.int >> 80
        if self.version == 8:
            return (self.int >> 80) * 10**6 + _subsec_decode(self.subsec)
        return super().time


def _subsec_decode(value: int) -> int:
    return -(-value * 10**6 // 2**20)


def uuid6(node: int | None = None, clock_seq: int | None = None) -> UUID:
    r"""UUID 버전 6은 UUIDv1의 필드 호환 버전으로, 향상된 DB 지역성을 위해
    재정렬되었습니다. UUIDv6은 주로 기존 v1 UUID가 있는 컨텍스트에서
    사용될 것으로 예상됩니다. 레거시 UUIDv1이 포함되지 않은 시스템은
    대신 UUIDv7 사용을 고려해야 합니다.

    'node'가 제공되지 않으면 무작위 48비트 숫자가 선택됩니다.

    'clock_seq'가 제공되면 시퀀스 번호로 사용됩니다.
    그렇지 않으면 무작위 14비트 시퀀스 번호가 선택됩니다."""

    global _last_v6_timestamp

    nanoseconds = time.time_ns()
    # 0x01b21dd213814000은 UUID epoch 1582-10-15 00:00:00과
    # Unix epoch 1970-01-01 00:00:00 사이의 100ns 간격 수입니다.
    timestamp = nanoseconds // 100 + 0x01B21DD213814000
    if _last_v6_timestamp is not None and timestamp <= _last_v6_timestamp:
        timestamp = _last_v6_timestamp + 1
    _last_v6_timestamp = timestamp
    if clock_seq is None:
        clock_seq = random.getrandbits(14)  # 안정적인 저장소 대신
    if node is None:
        node = random.getrandbits(48)
    time_high_and_time_mid = (timestamp >> 12) & 0xFFFFFFFFFFFF
    time_low_and_version = timestamp & 0x0FFF
    uuid_int = time_high_and_time_mid << 80
    uuid_int |= time_low_and_version << 64
    uuid_int |= (clock_seq & 0x3FFF) << 48
    uuid_int |= node & 0xFFFFFFFFFFFF
    return UUID(int=uuid_int, version=6)
