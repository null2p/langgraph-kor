from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class UntypedSerializerProtocol(Protocol):
    """객체의 직렬화 및 역직렬화를 위한 프로토콜입니다."""

    def dumps(self, obj: Any) -> bytes: ...

    def loads(self, data: bytes) -> Any: ...


@runtime_checkable
class SerializerProtocol(Protocol):
    """객체의 직렬화 및 역직렬화를 위한 프로토콜입니다.

    - `dumps`: 객체를 바이트로 직렬화합니다.
    - `dumps_typed`: 객체를 튜플 `(type, bytes)`로 직렬화합니다.
    - `loads`: 바이트에서 객체를 역직렬화합니다.
    - `loads_typed`: 튜플 `(type, bytes)`에서 객체를 역직렬화합니다.

    유효한 구현에는 `pickle`, `json` 및 `orjson` 모듈이 포함됩니다.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...


class SerializerCompat(SerializerProtocol):
    def __init__(self, serde: UntypedSerializerProtocol) -> None:
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return type(obj).__name__, self.serde.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.serde.loads(data[1])


def maybe_add_typed_methods(
    serde: SerializerProtocol | UntypedSerializerProtocol,
) -> SerializerProtocol:
    """하위 호환성을 위해 loads_typed 및 dumps_typed가 있는 클래스로 이전 serde 구현을 래핑합니다."""

    if not isinstance(serde, SerializerProtocol):
        return SerializerCompat(serde)

    return serde


class CipherProtocol(Protocol):
    """데이터 암호화 및 복호화를 위한 프로토콜입니다.
    - `encrypt`: 평문을 암호화합니다.
    - `decrypt`: 암호문을 복호화합니다.
    """

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        """평문을 암호화합니다. 튜플 (cipher name, ciphertext)을 반환합니다."""
        ...

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        """암호문을 복호화합니다. 평문을 반환합니다."""
        ...
