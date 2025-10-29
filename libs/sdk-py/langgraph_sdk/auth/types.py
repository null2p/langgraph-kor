"""LangGraph를 위한 인증 및 권한 부여 타입입니다.

이 모듈은 LangGraph에서 인증, 권한 부여 및 요청 처리에 사용되는 핵심 타입을
정의합니다. 사용자 프로토콜, 인증 컨텍스트 및 다양한 API 작업을 위한
타입 딕셔너리를 포함합니다.

Note:
    모든 typing.TypedDict 클래스는 total=False를 사용하여 기본적으로 모든 필드를 typing.Optional로 만듭니다.
"""

from __future__ import annotations

import typing
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import typing_extensions

RunStatus = typing.Literal["pending", "error", "success", "timeout", "interrupted"]
"""실행의 상태입니다.

Values:
    - pending: 실행이 대기 중이거나 진행 중
    - error: 실행이 오류로 실패함
    - success: 실행이 성공적으로 완료됨
    - timeout: 실행이 시간 제한을 초과함
    - interrupted: 실행이 수동으로 중단됨
"""

MultitaskStrategy = typing.Literal["reject", "rollback", "interrupt", "enqueue"]
"""여러 동시 작업을 처리하기 위한 전략입니다.

Values:
    - reject: 작업이 진행 중일 때 새 작업 거부
    - rollback: 현재 작업을 취소하고 새 작업 시작
    - interrupt: 현재 작업을 중단하고 새 작업 시작
    - enqueue: 현재 작업 후에 실행되도록 새 작업을 대기열에 추가
"""

OnConflictBehavior = typing.Literal["raise", "do_nothing"]
"""충돌 발생 시 동작입니다.

Values:
    - raise: 충돌 시 예외 발생
    - do_nothing: 충돌을 조용히 무시
"""

IfNotExists = typing.Literal["create", "reject"]
"""엔티티가 존재하지 않을 때의 동작입니다.

Values:
    - create: 엔티티 생성
    - reject: 작업 거부
"""

FilterType = (
    dict[
        str,
        str
        | dict[typing.Literal["$eq", "$contains"], str]
        | dict[typing.Literal["$contains"], list[str]],
    ]
    | dict[str, str]
)
"""권한 부여 핸들러를 위한 응답 타입입니다.

정확한 매치와 연산자를 지원합니다:
    - 정확한 매치 단축형: {"field": "value"}
    - 정확한 매치: {"field": {"$eq": "value"}}
    - 포함 (멤버십): {"field": {"$contains": "value"}}
    - 포함 (부분집합 포함): {"field": {"$contains": ["value1", "value2"]}}

부분집합 포함은 최신 버전의 LangGraph dev 서버에서만 지원됩니다;
이 필터 변형을 사용하려면 langgraph-runtime-inmem >= 0.14.1을 설치하세요.

???+ example "예제"

    리소스 소유자에 대한 간단한 정확한 매치 필터:

    ```python
    filter = {"owner": "user-abcd123"}
    ```

    정확한 매치 필터의 명시적 버전:

    ```python
    filter = {"owner": {"$eq": "user-abcd123"}}
    ```

    포함 (단일 요소의 멤버십):

    ```python
    filter = {"participants": {"$contains": "user-abcd123"}}
    ```

    포함 (부분집합 포함; 모든 값이 존재해야 하지만 순서는 상관없음):

    ```python
    filter = {"participants": {"$contains": ["user-abcd123", "user-efgh456"]}}
    ```

    필터 결합 (논리적 `AND`로 처리됨):

    ```python
    filter = {"owner": "user-abcd123", "participants": {"$contains": "user-efgh456"}}
    ```
"""

ThreadStatus = typing.Literal["idle", "busy", "interrupted", "error"]
"""스레드의 상태입니다.

Values:
    - idle: 스레드가 작업 가능 상태
    - busy: 스레드가 현재 처리 중
    - interrupted: 스레드가 중단됨
    - error: 스레드에서 오류 발생
"""

MetadataInput = dict[str, typing.Any]
"""엔티티에 첨부된 임의의 메타데이터 타입입니다.

모든 엔티티에 커스텀 키-값 쌍을 저장할 수 있습니다.
키는 문자열이어야 하며, 값은 JSON 직렬화 가능한 모든 타입일 수 있습니다.

???+ example "예제"

    ```python
    metadata = {
        "created_by": "user123",
        "priority": 1,
        "tags": ["important", "urgent"]
    }
    ```
"""

HandlerResult = None | bool | FilterType
"""핸들러의 결과는 다음 중 하나일 수 있습니다:
    * None | True: 요청 수락.
    * False: 403 오류로 요청 거부
    * FilterType: 적용할 필터
"""

Handler = Callable[..., Awaitable[HandlerResult]]

T = typing.TypeVar("T")


@typing.runtime_checkable
class MinimalUser(typing.Protocol):
    """사용자 객체는 최소한 identity 속성을 노출해야 합니다."""

    @property
    def identity(self) -> str:
        """사용자의 고유 식별자입니다.

        이것은 사용자 이름, 이메일 또는 시스템에서 다른 사용자를 구별하는 데
        사용되는 기타 고유 식별자일 수 있습니다.
        """
        ...


class MinimalUserDict(typing.TypedDict, total=False):
    """사용자의 딕셔너리 표현입니다."""

    identity: typing_extensions.Required[str]
    """사용자의 필수 고유 식별자입니다."""
    display_name: str
    """사용자의 typing.Optional 표시 이름입니다."""
    is_authenticated: bool
    """사용자가 인증되었는지 여부입니다. 기본값은 True입니다."""
    permissions: Sequence[str]
    """사용자와 연결된 권한 목록입니다.

    `@auth.on` 권한 부여 로직에서 이를 사용하여 다양한 리소스에 대한
    접근 권한을 결정할 수 있습니다.
    """


@typing.runtime_checkable
class BaseUser(typing.Protocol):
    """기본 ASGI 사용자 프로토콜입니다."""

    @property
    def is_authenticated(self) -> bool:
        """사용자가 인증되었는지 여부입니다."""
        ...

    @property
    def display_name(self) -> str:
        """사용자의 표시 이름입니다."""
        ...

    @property
    def identity(self) -> str:
        """사용자의 고유 식별자입니다."""
        ...

    @property
    def permissions(self) -> Sequence[str]:
        """사용자와 연결된 권한입니다."""
        ...

    def __getitem__(self, key):
        """최소 사용자 딕셔너리에서 키를 가져옵니다."""
        ...

    def __contains__(self, key):
        """속성이 존재하는지 확인합니다."""
        ...

    def __iter__(self):
        """사용자의 키를 반복합니다."""
        ...


class StudioUser:
    """LangGraph Studio의 인증된 요청으로부터 채워지는 사용자 객체입니다.

    Note: Studio 인증은 `langgraph.json` 구성에서 비활성화할 수 있습니다.

    ```json
    {
      "auth": {
        "disable_studio_auth": true
      }
    }
    ```

    권한 부여 핸들러(`@auth.on`)에서 `isinstance` 검사를 사용하여
    LangGraph Studio UI에서 인스턴스에 접근하는 개발자의 접근을 구체적으로 제어할 수 있습니다.

    ???+ example "예제"

        ```python
        @auth.on
        async def allow_developers(ctx: Auth.types.AuthContext, value: Any) -> None:
            if isinstance(ctx.user, Auth.types.StudioUser):
                return None
            ...
            return False
        ```
    """

    __slots__ = ("username", "_is_authenticated", "_permissions")

    def __init__(self, username: str, is_authenticated: bool = False) -> None:
        self.username = username
        self._is_authenticated = is_authenticated
        self._permissions = ["authenticated"] if is_authenticated else []

    @property
    def is_authenticated(self) -> bool:
        return self._is_authenticated

    @property
    def display_name(self) -> str:
        return self.username

    @property
    def identity(self) -> str:
        return self.username

    @property
    def permissions(self) -> Sequence[str]:
        return self._permissions


Authenticator = Callable[
    ...,
    Awaitable[
        MinimalUser
        | str
        | BaseUser
        | MinimalUserDict
        | typing.Mapping[str, typing.Any],
    ],
]
"""인증 함수를 위한 타입입니다.

인증자는 다음 중 하나를 반환할 수 있습니다:
1. 문자열 (user_id)
2. {"identity": str, "permissions": list[str]}을 포함하는 dict
3. identity 및 permissions 속성이 있는 객체

권한은 다양한 리소스에 대한 접근 권한을 결정하기 위해 권한 부여 로직에서
다운스트림에서 사용할 수 있습니다.

authenticate 데코레이터는 함수 시그니처에 포함된 경우 다음 매개변수 중 어느 것이든
이름으로 자동으로 주입합니다:

Parameters:
    request (Request): 원시 ASGI 요청 객체
    body (dict): 파싱된 요청 본문
    path (str): 요청 경로
    method (str): HTTP 메서드 (GET, POST 등)
    path_params (dict[str, str] | None): URL 경로 매개변수
    query_params (dict[str, str] | None): URL 쿼리 매개변수
    headers (dict[str, bytes] | None): 요청 헤더
    authorization (str | None): Authorization 헤더 값 (예: "Bearer <token>")

???+ example "예제"

    토큰을 사용한 기본 인증:

    ```python
    from langgraph_sdk import Auth

    auth = Auth()

    @auth.authenticate
    async def authenticate1(authorization: str) -> Auth.types.MinimalUserDict:
        return await get_user(authorization)
    ```

    여러 매개변수를 사용한 인증:

    ```
    @auth.authenticate
    async def authenticate2(
        method: str,
        path: str,
        headers: dict[str, bytes]
    ) -> Auth.types.MinimalUserDict:
        # 메서드, 경로 및 헤더를 사용한 커스텀 인증 로직
        user = verify_request(method, path, headers)
        return user
    ```

    원시 ASGI 요청 수락:

    ```python
    MY_SECRET = "my-secret-key"
    @auth.authenticate
    async def get_current_user(request: Request) -> Auth.types.MinimalUserDict:
        try:
            token = (request.headers.get("authorization") or "").split(" ", 1)[1]
            payload = jwt.decode(token, MY_SECRET, algorithms=["HS256"])
        except (IndexError, InvalidTokenError):
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.myauth-provider.com/auth/v1/user",
                headers={"Authorization": f"Bearer {MY_SECRET}"}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="User not found")

            user_data = response.json()
            return {
                "identity": user_data["id"],
                "display_name": user_data.get("name"),
                "permissions": user_data.get("permissions", []),
                "is_authenticated": True,
            }
    ```
"""


@dataclass(slots=True)
class BaseAuthContext:
    """인증 컨텍스트를 위한 베이스 클래스입니다.

    권한 부여 결정에 필요한 기본 인증 정보를 제공합니다.
    """

    permissions: Sequence[str]
    """인증된 사용자에게 부여된 권한입니다."""

    user: BaseUser
    """인증된 사용자입니다."""


@typing.final
@dataclass(slots=True)
class AuthContext(BaseAuthContext):
    """리소스 및 액션 정보를 포함한 완전한 인증 컨텍스트입니다.

    BaseAuthContext를 확장하여 접근 중인 특정 리소스와 액션을 포함하며,
    세밀한 접근 제어 결정을 가능하게 합니다.
    """

    resource: typing.Literal["runs", "threads", "crons", "assistants", "store"]
    """접근 중인 리소스입니다."""

    action: typing.Literal[
        "create",
        "read",
        "update",
        "delete",
        "search",
        "create_run",
        "put",
        "get",
        "list_namespaces",
    ]
    """리소스에 대해 수행되는 액션입니다.

    대부분의 리소스는 다음 액션을 지원합니다:
    - create: 새 리소스 생성
    - read: 리소스에 대한 정보 읽기
    - update: 기존 리소스 업데이트
    - delete: 리소스 삭제
    - search: 리소스 검색

    store는 다음 액션을 지원합니다:
    - put: store에 문서 추가 또는 업데이트
    - get: store에서 문서 가져오기
    - list_namespaces: store의 네임스페이스 나열
    """


class ThreadTTL(typing.TypedDict, total=False):
    """스레드의 수명(Time-to-live) 구성입니다.

    TTL이 선택적 전략과 분 단위 시간 값을 가진 객체로 표현되는
    OpenAPI 스키마와 일치합니다.
    """

    strategy: typing.Literal["delete"]
    """TTL 전략입니다. 현재 'delete'만 지원됩니다."""

    ttl: int
    """스레드가 정리될 때까지의 수명(분 단위)입니다."""


class ThreadsCreate(typing.TypedDict, total=False):
    """새 스레드를 생성하기 위한 매개변수입니다.

    ???+ example "예제"

        ```python
        create_params = {
            "thread_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "metadata": {"owner": "user123"},
            "if_exists": "do_nothing"
        }
        ```
    """

    thread_id: UUID
    """스레드의 고유 식별자입니다."""

    metadata: MetadataInput
    """스레드에 첨부할 typing.Optional 메타데이터입니다."""

    if_exists: OnConflictBehavior
    """동일한 ID를 가진 스레드가 이미 존재할 때의 동작입니다."""

    ttl: ThreadTTL
    """스레드의 선택적 TTL 구성입니다."""


class ThreadsRead(typing.TypedDict, total=False):
    """스레드 상태 또는 실행 정보를 읽기 위한 매개변수입니다.

    이 타입은 세 가지 컨텍스트에서 사용됩니다:
    1. 스레드, 스레드 버전 또는 스레드 상태 정보 읽기: thread_id만 제공됨
    2. 실행 정보 읽기: thread_id와 run_id가 모두 제공됨
    """

    thread_id: UUID
    """스레드의 고유 식별자입니다."""

    run_id: UUID | None
    """필터링할 실행 ID입니다. 스레드 내에서 실행 정보를 읽을 때만 사용됩니다."""


class ThreadsUpdate(typing.TypedDict, total=False):
    """스레드 또는 실행을 업데이트하기 위한 매개변수입니다.

    스레드, 스레드 버전 또는 실행 취소에 대한 업데이트에 호출됩니다.
    """

    thread_id: UUID
    """스레드의 고유 식별자입니다."""

    metadata: MetadataInput
    """업데이트할 typing.Optional 메타데이터입니다."""

    action: typing.Literal["interrupt", "rollback"] | None
    """스레드에 대해 수행할 typing.Optional 액션입니다."""


class ThreadsDelete(typing.TypedDict, total=False):
    """스레드를 삭제하기 위한 매개변수입니다.

    스레드, 스레드 버전 또는 실행 삭제에 호출됩니다.
    """

    thread_id: UUID
    """스레드의 고유 식별자입니다."""

    run_id: UUID | None
    """필터링할 typing.Optional 실행 ID입니다."""


class ThreadsSearch(typing.TypedDict, total=False):
    """스레드를 검색하기 위한 매개변수입니다.

    스레드 또는 실행 검색에 호출됩니다.
    """

    metadata: MetadataInput
    """필터링할 typing.Optional 메타데이터입니다."""

    values: MetadataInput
    """필터링할 typing.Optional 값입니다."""

    status: ThreadStatus | None
    """필터링할 typing.Optional 상태입니다."""

    limit: int
    """반환할 최대 결과 수입니다."""

    offset: int
    """페이지네이션을 위한 오프셋입니다."""

    ids: Sequence[UUID] | None
    """필터링할 typing.Optional 스레드 ID 목록입니다."""

    thread_id: UUID | None
    """필터링할 typing.Optional 스레드 ID입니다."""


class RunsCreate(typing.TypedDict, total=False):
    """실행을 생성하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        create_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "thread_id": UUID("123e4567-e89b-12d3-a456-426614174001"),
            "run_id": UUID("123e4567-e89b-12d3-a456-426614174002"),
            "status": "pending",
            "metadata": {"owner": "user123"},
            "prevent_insert_if_inflight": True,
            "multitask_strategy": "reject",
            "if_not_exists": "create",
            "after_seconds": 10,
            "kwargs": {"key": "value"},
            "action": "interrupt"
        }
        ```
    """

    assistant_id: UUID | None
    """이 실행에 사용할 typing.Optional 어시스턴트 ID입니다."""

    thread_id: UUID | None
    """이 실행에 사용할 typing.Optional 스레드 ID입니다."""

    run_id: UUID | None
    """이 실행에 사용할 typing.Optional 실행 ID입니다."""

    status: RunStatus | None
    """이 실행의 typing.Optional 상태입니다."""

    metadata: MetadataInput
    """실행의 typing.Optional 메타데이터입니다."""

    prevent_insert_if_inflight: bool
    """이미 진행 중인 실행이 있는 경우 새 실행 삽입을 방지합니다."""

    multitask_strategy: MultitaskStrategy
    """이 실행의 멀티태스크 전략입니다."""

    if_not_exists: IfNotExists
    """이 실행의 IfNotExists입니다."""

    after_seconds: int
    """실행을 생성하기 전에 대기할 초 수입니다."""

    kwargs: dict[str, typing.Any]
    """실행에 전달할 키워드 인수입니다."""

    action: typing.Literal["interrupt", "rollback"] | None
    """기존 실행을 업데이트하는 경우 수행할 액션입니다."""


class AssistantsCreate(typing.TypedDict, total=False):
    """어시스턴트를 생성하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        create_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "graph_id": "graph123",
            "config": {"tags": ["tag1", "tag2"]},
            "context": {"key": "value"},
            "metadata": {"owner": "user123"},
            "if_exists": "do_nothing",
            "name": "Assistant 1"
        }
        ```
    """

    assistant_id: UUID
    """어시스턴트의 고유 식별자입니다."""

    graph_id: str
    """이 어시스턴트에 사용할 그래프 ID입니다."""

    config: dict[str, typing.Any]
    """어시스턴트의 typing.Optional 구성입니다."""

    context: dict[str, typing.Any]

    metadata: MetadataInput
    """어시스턴트에 첨부할 typing.Optional 메타데이터입니다."""

    if_exists: OnConflictBehavior
    """동일한 ID를 가진 어시스턴트가 이미 존재할 때의 동작입니다."""

    name: str
    """어시스턴트의 이름입니다."""


class AssistantsRead(typing.TypedDict, total=False):
    """어시스턴트를 읽기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        read_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "metadata": {"owner": "user123"}
        }
        ```
    """

    assistant_id: UUID
    """어시스턴트의 고유 식별자입니다."""

    metadata: MetadataInput
    """필터링할 typing.Optional 메타데이터입니다."""


class AssistantsUpdate(typing.TypedDict, total=False):
    """어시스턴트를 업데이트하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        update_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "graph_id": "graph123",
            "config": {"tags": ["tag1", "tag2"]},
            "context": {"key": "value"},
            "metadata": {"owner": "user123"},
            "name": "Assistant 1",
            "version": 1
        }
        ```
    """

    assistant_id: UUID
    """어시스턴트의 고유 식별자입니다."""

    graph_id: str | None
    """업데이트할 typing.Optional 그래프 ID입니다."""

    config: dict[str, typing.Any]
    """업데이트할 typing.Optional 구성입니다."""

    context: dict[str, typing.Any]
    """어시스턴트의 정적 컨텍스트입니다."""

    metadata: MetadataInput
    """업데이트할 typing.Optional 메타데이터입니다."""

    name: str | None
    """업데이트할 typing.Optional 이름입니다."""

    version: int | None
    """업데이트할 typing.Optional 버전입니다."""


class AssistantsDelete(typing.TypedDict):
    """어시스턴트를 삭제하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        delete_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000")
        }
        ```
    """

    assistant_id: UUID
    """어시스턴트의 고유 식별자입니다."""


class AssistantsSearch(typing.TypedDict):
    """어시스턴트를 검색하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        search_params = {
            "graph_id": "graph123",
            "metadata": {"owner": "user123"},
            "limit": 10,
            "offset": 0
        }
        ```
    """

    graph_id: str | None
    """필터링할 typing.Optional 그래프 ID입니다."""

    metadata: MetadataInput
    """필터링할 typing.Optional 메타데이터입니다."""

    limit: int
    """반환할 최대 결과 수입니다."""

    offset: int
    """페이지네이션을 위한 오프셋입니다."""


class CronsCreate(typing.TypedDict, total=False):
    """크론 작업을 생성하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        create_params = {
            "payload": {"key": "value"},
            "schedule": "0 0 * * *",
            "cron_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "thread_id": UUID("123e4567-e89b-12d3-a456-426614174001"),
            "user_id": "user123",
            "end_time": datetime(2024, 3, 16, 10, 0, 0)
        }
        ```
    """

    payload: dict[str, typing.Any]
    """크론 작업의 페이로드입니다."""

    schedule: str
    """크론 작업의 스케줄입니다."""

    cron_id: UUID | None
    """크론 작업의 typing.Optional 고유 식별자입니다."""

    thread_id: UUID | None
    """이 크론 작업에 사용할 typing.Optional 스레드 ID입니다."""

    user_id: str | None
    """이 크론 작업에 사용할 typing.Optional 사용자 ID입니다."""

    end_time: datetime | None
    """크론 작업의 typing.Optional 종료 시간입니다."""


class CronsDelete(typing.TypedDict):
    """크론 작업을 삭제하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        delete_params = {
            "cron_id": UUID("123e4567-e89b-12d3-a456-426614174000")
        }
        ```
    """

    cron_id: UUID
    """크론 작업의 고유 식별자입니다."""


class CronsRead(typing.TypedDict):
    """크론 작업을 읽기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        read_params = {
            "cron_id": UUID("123e4567-e89b-12d3-a456-426614174000")
        }
        ```
    """

    cron_id: UUID
    """크론 작업의 고유 식별자입니다."""


class CronsUpdate(typing.TypedDict, total=False):
    """크론 작업을 업데이트하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        update_params = {
            "cron_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "payload": {"key": "value"},
            "schedule": "0 0 * * *"
        }
        ```
    """

    cron_id: UUID
    """크론 작업의 고유 식별자입니다."""

    payload: dict[str, typing.Any] | None
    """업데이트할 typing.Optional 페이로드입니다."""

    schedule: str | None
    """업데이트할 typing.Optional 스케줄입니다."""


class CronsSearch(typing.TypedDict, total=False):
    """크론 작업을 검색하기 위한 페이로드입니다.

    ???+ example "예제"

        ```python
        search_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "thread_id": UUID("123e4567-e89b-12d3-a456-426614174001"),
            "limit": 10,
            "offset": 0
        }
        ```
    """

    assistant_id: UUID | None
    """필터링할 typing.Optional 어시스턴트 ID입니다."""

    thread_id: UUID | None
    """필터링할 typing.Optional 스레드 ID입니다."""

    limit: int
    """반환할 최대 결과 수입니다."""

    offset: int
    """페이지네이션을 위한 오프셋입니다."""


class StoreGet(typing.TypedDict):
    """네임스페이스와 키로 특정 아이템을 검색하는 작업입니다."""

    namespace: tuple[str, ...]
    """아이템의 위치를 고유하게 식별하는 계층적 경로입니다."""

    key: str
    """특정 네임스페이스 내에서 아이템의 고유 식별자입니다."""


class StoreSearch(typing.TypedDict):
    """지정된 네임스페이스 계층 내에서 아이템을 검색하는 작업입니다."""

    namespace: tuple[str, ...]
    """검색 범위를 정의하기 위한 접두사 필터입니다."""

    filter: dict[str, typing.Any] | None
    """정확한 매치 또는 비교 연산자를 기반으로 결과를 필터링하기 위한 키-값 쌍입니다."""

    limit: int
    """검색 결과에서 반환할 최대 아이템 수입니다."""

    offset: int
    """페이지네이션을 위해 건너뛸 일치하는 아이템 수입니다."""

    query: str | None
    """시맨틱 검색 기능을 위한 자연어 검색 쿼리입니다."""


class StoreListNamespaces(typing.TypedDict):
    """store의 네임스페이스를 나열하고 필터링하는 작업입니다."""

    namespace: tuple[str, ...] | None
    """네임스페이스를 필터링하는 접두사입니다."""

    suffix: tuple[str, ...] | None
    """네임스페이스를 필터링하기 위한 선택적 조건입니다."""

    max_depth: int | None
    """반환할 네임스페이스 계층의 최대 깊이입니다.

    Note:
        이 레벨보다 깊은 네임스페이스는 잘립니다.
    """

    limit: int
    """반환할 최대 네임스페이스 수입니다."""

    offset: int
    """페이지네이션을 위해 건너뛸 네임스페이스 수입니다."""


class StorePut(typing.TypedDict):
    """store에 아이템을 저장, 업데이트 또는 삭제하는 작업입니다."""

    namespace: tuple[str, ...]
    """아이템의 위치를 식별하는 계층적 경로입니다."""

    key: str
    """네임스페이스 내에서 아이템의 고유 식별자입니다."""

    value: dict[str, typing.Any] | None
    """저장할 데이터이거나, 삭제를 위해 표시하려면 `None`입니다."""

    index: typing.Literal[False] | list[str] | None
    """전체 텍스트 검색을 위한 선택적 인덱스 구성입니다."""


class StoreDelete(typing.TypedDict):
    """store에서 아이템을 삭제하는 작업입니다."""

    namespace: tuple[str, ...]
    """아이템의 위치를 고유하게 식별하는 계층적 경로입니다."""

    key: str
    """특정 네임스페이스 내에서 아이템의 고유 식별자입니다."""


class on:
    """다양한 API 작업의 타입 정의를 위한 네임스페이스입니다.

    이 클래스는 다양한 리소스(threads, assistants, crons)에 걸쳐 생성, 읽기, 업데이트, 삭제
    및 검색 작업에 대한 타입 정의를 구성합니다.

    ???+ note "사용법"
        ```python
        from langgraph_sdk import Auth

        auth = Auth()

        @auth.on
        def handle_all(params: Auth.on.value):
            raise Exception("Not authorized")

        @auth.on.threads.create
        def handle_thread_create(params: Auth.on.threads.create.value):
            # 스레드 생성 처리
            pass

        @auth.on.assistants.search
        def handle_assistant_search(params: Auth.on.assistants.search.value):
            # 어시스턴트 검색 처리
            pass
        ```
    """

    value = dict[str, typing.Any]

    class threads:
        """스레드 관련 작업을 위한 타입입니다."""

        value = (
            ThreadsCreate | ThreadsRead | ThreadsUpdate | ThreadsDelete | ThreadsSearch
        )

        class create:
            """스레드 생성 매개변수를 위한 타입입니다."""

            value = ThreadsCreate

        class create_run:
            """실행 생성 또는 스트리밍을 위한 타입입니다."""

            value = RunsCreate

        class read:
            """스레드 읽기 매개변수를 위한 타입입니다."""

            value = ThreadsRead

        class update:
            """스레드 업데이트 매개변수를 위한 타입입니다."""

            value = ThreadsUpdate

        class delete:
            """스레드 삭제 매개변수를 위한 타입입니다."""

            value = ThreadsDelete

        class search:
            """스레드 검색 매개변수를 위한 타입입니다."""

            value = ThreadsSearch

    class assistants:
        """어시스턴트 관련 작업을 위한 타입입니다."""

        value = (
            AssistantsCreate
            | AssistantsRead
            | AssistantsUpdate
            | AssistantsDelete
            | AssistantsSearch
        )

        class create:
            """어시스턴트 생성 매개변수를 위한 타입입니다."""

            value = AssistantsCreate

        class read:
            """어시스턴트 읽기 매개변수를 위한 타입입니다."""

            value = AssistantsRead

        class update:
            """어시스턴트 업데이트 매개변수를 위한 타입입니다."""

            value = AssistantsUpdate

        class delete:
            """어시스턴트 삭제 매개변수를 위한 타입입니다."""

            value = AssistantsDelete

        class search:
            """어시스턴트 검색 매개변수를 위한 타입입니다."""

            value = AssistantsSearch

    class crons:
        """크론 관련 작업을 위한 타입입니다."""

        value = CronsCreate | CronsRead | CronsUpdate | CronsDelete | CronsSearch

        class create:
            """크론 생성 매개변수를 위한 타입입니다."""

            value = CronsCreate

        class read:
            """크론 읽기 매개변수를 위한 타입입니다."""

            value = CronsRead

        class update:
            """크론 업데이트 매개변수를 위한 타입입니다."""

            value = CronsUpdate

        class delete:
            """크론 삭제 매개변수를 위한 타입입니다."""

            value = CronsDelete

        class search:
            """크론 검색 매개변수를 위한 타입입니다."""

            value = CronsSearch

    class store:
        """store 관련 작업을 위한 타입입니다."""

        value = StoreGet | StoreSearch | StoreListNamespaces | StorePut | StoreDelete

        class put:
            """store put 매개변수를 위한 타입입니다."""

            value = StorePut

        class get:
            """store get 매개변수를 위한 타입입니다."""

            value = StoreGet

        class search:
            """store 검색 매개변수를 위한 타입입니다."""

            value = StoreSearch

        class delete:
            """store 삭제 매개변수를 위한 타입입니다."""

            value = StoreDelete

        class list_namespaces:
            """store 네임스페이스 나열 매개변수를 위한 타입입니다."""

            value = StoreListNamespaces


__all__ = [
    "on",
    "MetadataInput",
    "RunsCreate",
    "ThreadsCreate",
    "ThreadsRead",
    "ThreadsUpdate",
    "ThreadsDelete",
    "ThreadsSearch",
    "AssistantsCreate",
    "AssistantsRead",
    "AssistantsUpdate",
    "AssistantsDelete",
    "AssistantsSearch",
    "StoreGet",
    "StoreSearch",
    "StoreListNamespaces",
    "StorePut",
    "StoreDelete",
]
