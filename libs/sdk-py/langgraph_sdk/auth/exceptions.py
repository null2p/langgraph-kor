"""auth 시스템에서 사용되는 예외입니다."""

from __future__ import annotations

import http
from collections.abc import Mapping


class HTTPException(Exception):
    """특정 HTTP 오류 응답을 반환하기 위해 발생시킬 수 있는 HTTP 예외입니다.

    이것은 auth 모듈에 정의되어 있으므로 기본적으로 401 상태 코드를 사용합니다.

    Args:
        status_code: 오류에 대한 HTTP 상태 코드입니다. 기본값은 401 "Unauthorized"입니다.
        detail: 상세한 오류 메시지입니다. `None`인 경우, 상태 코드를 기반으로
            기본 메시지를 사용합니다.
        headers: 오류 응답에 포함할 추가 HTTP 헤더입니다.

    Example:
        기본값:
        ```python
        raise HTTPException()
        # HTTPException(status_code=401, detail='Unauthorized')
        ```

        헤더 추가:
        ```python
        raise HTTPException(headers={"X-Custom-Header": "Custom Value"})
        # HTTPException(status_code=401, detail='Unauthorized', headers={"WWW-Authenticate": "Bearer"})
        ```

        커스텀 오류:
        ```python
        raise HTTPException(status_code=404, detail="Not found")
        ```
    """

    def __init__(
        self,
        status_code: int = 401,
        detail: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if detail is None:
            detail = http.HTTPStatus(status_code).phrase
        self.status_code = status_code
        self.detail = detail
        self.headers = headers

    def __str__(self) -> str:
        return f"{self.status_code}: {self.detail}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(status_code={self.status_code!r}, detail={self.detail!r})"


__all__ = ["HTTPException"]
