import os
from typing import Any

from langgraph.checkpoint.serde.base import CipherProtocol, SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class EncryptedSerializer(SerializerProtocol):
    """암호화 프로토콜을 사용하여 데이터를 암호화 및 복호화하는 시리얼라이저입니다."""

    def __init__(
        self, cipher: CipherProtocol, serde: SerializerProtocol = JsonPlusSerializer()
    ) -> None:
        self.cipher = cipher
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """객체를 튜플 `(type, bytes)`로 직렬화하고 바이트를 암호화합니다."""
        # 데이터 직렬화
        typ, data = self.serde.dumps_typed(obj)
        # 데이터 암호화
        ciphername, ciphertext = self.cipher.encrypt(data)
        # 타입에 cipher name 추가
        return f"{typ}+{ciphername}", ciphertext

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        enc_cipher, ciphertext = data
        # 암호화되지 않은 데이터
        if "+" not in enc_cipher:
            return self.serde.loads_typed(data)
        # cipher name 추출
        typ, ciphername = enc_cipher.split("+", 1)
        # 데이터 복호화
        decrypted_data = self.cipher.decrypt(ciphername, ciphertext)
        # 데이터 역직렬화
        return self.serde.loads_typed((typ, decrypted_data))

    @classmethod
    def from_pycryptodome_aes(
        cls, serde: SerializerProtocol = JsonPlusSerializer(), **kwargs: Any
    ) -> "EncryptedSerializer":
        """AES 암호화를 사용하는 `EncryptedSerializer`를 생성합니다."""
        try:
            from Crypto.Cipher import AES  # type: ignore
        except ImportError:
            raise ImportError(
                "Pycryptodome is not installed. Please install it with `pip install pycryptodome`."
            ) from None

        # AES 키가 제공되었는지 확인
        if "key" in kwargs:
            key: bytes = kwargs.pop("key")
        else:
            key_str = os.getenv("LANGGRAPH_AES_KEY")
            if key_str is None:
                raise ValueError("LANGGRAPH_AES_KEY 환경 변수가 설정되지 않았습니다.")
            key = key_str.encode()
            if len(key) not in (16, 24, 32):
                raise ValueError("LANGGRAPH_AES_KEY는 16, 24 또는 32바이트 길이여야 합니다.")

        # 제공되지 않은 경우 기본 모드를 EAX로 설정
        if kwargs.get("mode") is None:
            kwargs["mode"] = AES.MODE_EAX

        class PycryptodomeAesCipher(CipherProtocol):
            def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
                cipher = AES.new(key, **kwargs)
                ciphertext, tag = cipher.encrypt_and_digest(plaintext)
                return "aes", cipher.nonce + tag + ciphertext

            def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
                assert ciphername == "aes", f"Unsupported cipher: {ciphername}"
                nonce = ciphertext[:16]
                tag = ciphertext[16:32]
                actual_ciphertext = ciphertext[32:]

                cipher = AES.new(key, **kwargs, nonce=nonce)
                return cipher.decrypt_and_verify(actual_ciphertext, tag)

        return cls(PycryptodomeAesCipher(), serde)
