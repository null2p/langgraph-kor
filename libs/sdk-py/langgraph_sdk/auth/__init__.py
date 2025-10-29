from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Sequence

from langgraph_sdk.auth import exceptions, types

TH = typing.TypeVar("TH", bound=types.Handler)
AH = typing.TypeVar("AH", bound=types.Authenticator)


class Auth:
    """LangGraph 애플리케이션에 커스텀 인증 및 권한 부여 관리를 추가합니다.

    Auth 클래스는 LangGraph 애플리케이션에서 인증 및 권한 부여를 처리하기 위한
    통합 시스템을 제공합니다. 커스텀 사용자 인증 프로토콜과 다양한 리소스 및
    작업에 대한 세밀한 권한 부여 규칙을 지원합니다.

    사용하려면 별도의 python 파일을 생성하고 해당 파일의 경로를 LangGraph API
    구성 파일(`langgraph.json`)에 추가하세요. 해당 파일 내에서 Auth 클래스의
    인스턴스를 생성하고 필요에 따라 인증 및 권한 부여 핸들러를 등록합니다.

    Example `langgraph.json` file:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "agent": "./my_agent/agent.py:graph"
      },
      "env": ".env",
      "auth": {
        "path": "./auth.py:my_auth"
      }
    ```

    그러면 LangGraph 서버가 auth 파일을 로드하고 요청이 들어올 때마다 서버 측에서 실행합니다.

    ???+ example "Basic Usage"

        ```python
        from langgraph_sdk import Auth

        my_auth = Auth()

        async def verify_token(token: str) -> str:
            # Verify token and return user_id
            # This would typically be a call to your auth server
            return "user_id"

        @auth.authenticate
        async def authenticate(authorization: str) -> str:
            # Verify token and return user_id
            result = await verify_token(authorization)
            if result != "user_id":
                raise Auth.exceptions.HTTPException(
                    status_code=401, detail="Unauthorized"
                )
            return result

        # Global fallback handler
        @auth.on
        async def authorize_default(params: Auth.on.value):
            return False # Reject all requests (default behavior)

        @auth.on.threads.create
        async def authorize_thread_create(params: Auth.on.threads.create.value):
            # Allow the allowed user to create a thread
            assert params.get("metadata", {}).get("owner") == "allowed_user"

        @auth.on.store
        async def authorize_store(ctx: Auth.types.AuthContext, value: Auth.types.on):
            assert ctx.user.identity in value["namespace"], "Not authorized"
        ```

    ???+ note "요청 처리 흐름"

        1. 인증(`@auth.authenticate` 핸들러)이 **모든 요청**에 대해 먼저 수행됩니다
        2. 권한 부여의 경우 가장 구체적으로 일치하는 핸들러가 호출됩니다:
            * 정확한 리소스와 작업에 대한 핸들러가 존재하면 사용됩니다 (예: `@auth.on.threads.create`)
            * 그렇지 않으면 모든 작업에 대한 리소스 핸들러가 존재하면 사용됩니다 (예: `@auth.on.threads`)
            * 마지막으로 특정 핸들러가 일치하지 않으면 전역 핸들러가 사용됩니다 (예: `@auth.on`)
            * 전역 핸들러가 설정되지 않은 경우 요청이 수락됩니다

        이를 통해 전역 핸들러로 기본 동작을 설정하면서
        필요에 따라 특정 경로를 재정의할 수 있습니다.
    """

    __slots__ = (
        "on",
        "_handlers",
        "_global_handlers",
        "_authenticate_handler",
        "_handler_cache",
    )
    types = types
    """auth 타입 정의에 대한 참조입니다.

    ThreadsCreate, AssistantsRead 등과 같은 auth 시스템에서 사용되는
    모든 타입 정의에 대한 액세스를 제공합니다."""

    exceptions = exceptions
    """auth 예외 정의에 대한 참조입니다.

    HTTPException 등과 같은 auth 시스템에서 사용되는
    모든 예외 정의에 대한 액세스를 제공합니다.
    """

    def __init__(self) -> None:
        self.on = _On(self)
        """특정 리소스에 대한 액세스를 제어하는 권한 부여 핸들러의 진입점입니다.

        on 클래스는 애플리케이션의 다양한 리소스 및 작업에 대한 권한 부여 규칙을
        정의하는 유연한 방법을 제공합니다. 세 가지 주요 사용 패턴을 지원합니다:

        1. 모든 리소스 및 작업에 대해 실행되는 전역 핸들러
        2. 리소스의 모든 작업에 대해 실행되는 리소스별 핸들러
        3. 세밀한 제어를 위한 리소스 및 작업별 핸들러

        각 핸들러는 두 개의 매개변수를 받는 비동기 함수여야 합니다:
            - ctx (AuthContext): 요청 컨텍스트 및 인증된 사용자 정보를 포함합니다
            - value: 권한 부여되는 데이터 (엔드포인트에 따라 타입이 다름)

        핸들러는 다음 중 하나를 반환해야 합니다:

            - None 또는 True: 요청을 수락합니다
            - False: 403 오류로 거부합니다
            - FilterType: 응답에 필터링 규칙을 적용합니다
        
        ???+ example "Examples"

            Global handler for all requests:

            ```python
            @auth.on
            async def reject_unhandled_requests(ctx: AuthContext, value: Any) -> None:
                print(f"Request to {ctx.path} by {ctx.user.identity}")
                return False
            ```

            Resource-specific handler. This would take precedence over the global handler
            for all actions on the `threads` resource:
            
            ```python
            @auth.on.threads
            async def check_thread_access(ctx: AuthContext, value: Any) -> bool:
                # Allow access only to threads created by the user
                return value.get("created_by") == ctx.user.identity
            ```

            Resource and action specific handler:

            ```python
            @auth.on.threads.delete
            async def prevent_thread_deletion(ctx: AuthContext, value: Any) -> bool:
                # Only admins can delete threads
                return "admin" in ctx.user.permissions
            ```

            Multiple resources or actions:

            ```python
            @auth.on(resources=["threads", "runs"], actions=["create", "update"])
            async def rate_limit_writes(ctx: AuthContext, value: Any) -> bool:
                # Implement rate limiting for write operations
                return await check_rate_limit(ctx.user.identity)
            ```

            Auth for the `store` resource is a bit different since its structure is developer defined.
            You typically want to enforce user creds in the namespace.

            ```python
            @auth.on.store
            async def check_store_access(ctx: AuthContext, value: Auth.types.on) -> bool:
                # Assuming you structure your store like (store.aput((user_id, application_context), key, value))
                assert value["namespace"][0] == ctx.user.identity
            ```
        """
        # These are accessed by the API. Changes to their names or types is
        # will be considered a breaking change.
        self._handlers: dict[tuple[str, str], list[types.Handler]] = {}
        self._global_handlers: list[types.Handler] = []
        self._authenticate_handler: types.Authenticator | None = None
        self._handler_cache: dict[tuple[str, str], types.Handler] = {}

    def authenticate(self, fn: AH) -> AH:
        """인증 핸들러 함수를 등록합니다.

        인증 핸들러는 자격 증명을 확인하고 사용자 범위를 반환하는 역할을 합니다.
        다음 매개변수 중 하나를 이름으로 받을 수 있습니다:

            - request (Request): 원시 ASGI 요청 객체
            - path (str): 요청 경로, 예: "/threads/abcd-1234-abcd-1234/runs/abcd-1234-abcd-1234/stream"
            - method (str): HTTP 메서드, 예: "GET"
            - path_params (dict[str, str]): URL 경로 매개변수, 예: {"thread_id": "abcd-1234-abcd-1234", "run_id": "abcd-1234-abcd-1234"}
            - query_params (dict[str, str]): URL 쿼리 매개변수, 예: {"stream": "true"}
            - headers (dict[bytes, bytes]): 요청 헤더
            - authorization (str | None): Authorization 헤더 값 (예: "Bearer <token>")

        Args:
            fn: 등록할 인증 핸들러 함수입니다.
                사용자의 표현을 반환해야 합니다. 다음 중 하나일 수 있습니다:
                    - 문자열 (사용자 id)
                    - {"identity": str, "permissions": list[str]}를 포함하는 dict
                    - 또는 identity 및 permissions 속성이 있는 객체
                권한은 다운스트림 핸들러에서 선택적으로 사용할 수 있습니다.

        Returns:
            등록된 핸들러 함수입니다.

        Raises:
            ValueError: 인증 핸들러가 이미 등록된 경우.

        ???+ example "예제"

            기본 토큰 인증:

            ```python
            @auth.authenticate
            async def authenticate(authorization: str) -> str:
                user_id = verify_token(authorization)
                return user_id
            ```

            전체 요청 컨텍스트 받기:

            ```python
            @auth.authenticate
            async def authenticate(
                method: str,
                path: str,
                headers: dict[str, bytes]
            ) -> str:
                user = await verify_request(method, path, headers)
                return user
            ```

            사용자 이름과 권한 반환:

            ```python
            @auth.authenticate
            async def authenticate(
                method: str,
                path: str,
                headers: dict[str, bytes]
            ) -> Auth.types.MinimalUserDict:
                permissions, user = await verify_request(method, path, headers)
                # 권한은 다음과 같을 수 있습니다: ["runs:read", "runs:write", "threads:read", "threads:write"]
                return {
                    "identity": user["id"],
                    "permissions": permissions,
                    "display_name": user["name"],
                }
            ```
        """
        if self._authenticate_handler is not None:
            raise ValueError(
                f"인증 핸들러가 이미 {self._authenticate_handler}로 설정되어 있습니다."
            )
        self._authenticate_handler = fn
        return fn


## 헬퍼 타입 및 유틸리티

V = typing.TypeVar("V", contravariant=True)


class _ActionHandler(typing.Protocol[V]):
    async def __call__(
        self, *, ctx: types.AuthContext, value: V
    ) -> types.HandlerResult: ...


T = typing.TypeVar("T", covariant=True)


class _ResourceActionOn(typing.Generic[T]):
    def __init__(
        self,
        auth: Auth,
        resource: typing.Literal["threads", "crons", "assistants"],
        action: typing.Literal[
            "create", "read", "update", "delete", "search", "create_run"
        ],
        value: type[T],
    ) -> None:
        self.auth = auth
        self.resource = resource
        self.action = action
        self.value = value

    def __call__(self, fn: _ActionHandler[T]) -> _ActionHandler[T]:
        _validate_handler(fn)
        _register_handler(self.auth, self.resource, self.action, fn)
        return fn


VCreate = typing.TypeVar("VCreate", covariant=True)
VUpdate = typing.TypeVar("VUpdate", covariant=True)
VRead = typing.TypeVar("VRead", covariant=True)
VDelete = typing.TypeVar("VDelete", covariant=True)
VSearch = typing.TypeVar("VSearch", covariant=True)


class _ResourceOn(typing.Generic[VCreate, VRead, VUpdate, VDelete, VSearch]):
    """
    리소스별 핸들러를 위한 제네릭 베이스 클래스입니다.
    """

    value: type[VCreate | VUpdate | VRead | VDelete | VSearch]

    Create: type[VCreate]
    Read: type[VRead]
    Update: type[VUpdate]
    Delete: type[VDelete]
    Search: type[VSearch]

    def __init__(
        self,
        auth: Auth,
        resource: typing.Literal["threads", "crons", "assistants"],
    ) -> None:
        self.auth = auth
        self.resource = resource
        self.create: _ResourceActionOn[VCreate] = _ResourceActionOn(
            auth, resource, "create", self.Create
        )
        self.read: _ResourceActionOn[VRead] = _ResourceActionOn(
            auth, resource, "read", self.Read
        )
        self.update: _ResourceActionOn[VUpdate] = _ResourceActionOn(
            auth, resource, "update", self.Update
        )
        self.delete: _ResourceActionOn[VDelete] = _ResourceActionOn(
            auth, resource, "delete", self.Delete
        )
        self.search: _ResourceActionOn[VSearch] = _ResourceActionOn(
            auth, resource, "search", self.Search
        )

    @typing.overload
    def __call__(
        self,
        fn: (
            _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch]
            | _ActionHandler[dict[str, typing.Any]]
        ),
    ) -> _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch]: ...

    @typing.overload
    def __call__(
        self,
        *,
        resources: str | Sequence[str],
        actions: str | Sequence[str] | None = None,
    ) -> Callable[
        [_ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch]],
        _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch],
    ]: ...

    def __call__(
        self,
        fn: (
            _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch]
            | _ActionHandler[dict[str, typing.Any]]
            | None
        ) = None,
        *,
        resources: str | Sequence[str] | None = None,
        actions: str | Sequence[str] | None = None,
    ) -> (
        _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch]
        | Callable[
            [_ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch]],
            _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch],
        ]
    ):
        if fn is not None:
            _validate_handler(fn)
            return typing.cast(
                _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch],
                _register_handler(self.auth, self.resource, "*", fn),
            )

        def decorator(
            handler: _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch],
        ) -> _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch]:
            _validate_handler(handler)
            return typing.cast(
                _ActionHandler[VCreate | VUpdate | VRead | VDelete | VSearch],
                _register_handler(self.auth, self.resource, "*", handler),
            )

        # 향후 필터링 동작을 위한 키워드 전용 매개변수를 받아들임; 린터를 만족시키기 위해 참조됨.
        _ = resources, actions
        return decorator


class _AssistantsOn(
    _ResourceOn[
        types.AssistantsCreate,
        types.AssistantsRead,
        types.AssistantsUpdate,
        types.AssistantsDelete,
        types.AssistantsSearch,
    ]
):
    value = (
        types.AssistantsCreate
        | types.AssistantsRead
        | types.AssistantsUpdate
        | types.AssistantsDelete
        | types.AssistantsSearch
    )
    Create = types.AssistantsCreate
    Read = types.AssistantsRead
    Update = types.AssistantsUpdate
    Delete = types.AssistantsDelete
    Search = types.AssistantsSearch


class _ThreadsOn(
    _ResourceOn[
        types.ThreadsCreate,
        types.ThreadsRead,
        types.ThreadsUpdate,
        types.ThreadsDelete,
        types.ThreadsSearch,
    ]
):
    value = (
        types.ThreadsCreate
        | types.ThreadsRead
        | types.ThreadsUpdate
        | types.ThreadsDelete
        | types.ThreadsSearch
        | types.RunsCreate
    )
    Create = types.ThreadsCreate
    Read = types.ThreadsRead
    Update = types.ThreadsUpdate
    Delete = types.ThreadsDelete
    Search = types.ThreadsSearch
    CreateRun = types.RunsCreate

    def __init__(
        self,
        auth: Auth,
        resource: typing.Literal["threads", "crons", "assistants"],
    ) -> None:
        super().__init__(auth, resource)
        self.create_run: _ResourceActionOn[types.RunsCreate] = _ResourceActionOn(
            auth, resource, "create_run", self.CreateRun
        )


class _CronsOn(
    _ResourceOn[
        types.CronsCreate,
        types.CronsRead,
        types.CronsUpdate,
        types.CronsDelete,
        types.CronsSearch,
    ]
):
    value = type[
        types.CronsCreate
        | types.CronsRead
        | types.CronsUpdate
        | types.CronsDelete
        | types.CronsSearch
    ]

    Create = types.CronsCreate
    Read = types.CronsRead
    Update = types.CronsUpdate
    Delete = types.CronsDelete
    Search = types.CronsSearch


class _StoreOn:
    def __init__(self, auth: Auth) -> None:
        self._auth = auth

    @typing.overload
    def __call__(
        self,
        *,
        actions: (
            typing.Literal["put", "get", "search", "list_namespaces", "delete"]
            | Sequence[
                typing.Literal["put", "get", "search", "list_namespaces", "delete"]
            ]
            | None
        ) = None,
    ) -> Callable[[AHO], AHO]: ...

    @typing.overload
    def __call__(self, fn: AHO) -> AHO: ...

    def __call__(
        self,
        fn: AHO | None = None,
        *,
        actions: (
            typing.Literal["put", "get", "search", "list_namespaces", "delete"]
            | Sequence[
                typing.Literal["put", "get", "search", "list_namespaces", "delete"]
            ]
            | None
        ) = None,
    ) -> AHO | Callable[[AHO], AHO]:
        """특정 리소스와 액션에 대한 핸들러를 등록합니다.

        데코레이터로 사용하거나 명시적인 리소스/액션 매개변수와 함께 사용할 수 있습니다:

        @auth.on.store
        async def handler(): ... # 모든 store 작업 처리

        @auth.on.store(actions=("put", "get", "search", "delete"))
        async def handler(): ... # 특정 store 작업 처리

        @auth.on.store.put
        async def handler(): ... # store.put 작업 처리
        """
        if fn is not None:
            # 일반 데코레이터로 사용됨
            _register_handler(self._auth, "store", None, fn)
            return fn

        # 매개변수와 함께 사용됨, 데코레이터 반환
        def decorator(
            handler: AHO,
        ) -> AHO:
            if isinstance(actions, str):
                action_list = [actions]
            else:
                action_list = list(actions) if actions is not None else ["*"]
            for action in action_list:
                _register_handler(self._auth, "store", action, handler)
            return handler

        return decorator


AHO = typing.TypeVar("AHO", bound=_ActionHandler[dict[str, typing.Any]])


class _On:
    """특정 리소스에 대한 접근을 제어하는 권한 부여 핸들러의 진입점입니다.

    _On 클래스는 애플리케이션에서 다양한 리소스와 액션에 대한 권한 부여 규칙을
    정의하는 유연한 방법을 제공합니다. 세 가지 주요 사용 패턴을 지원합니다:

    1. 모든 리소스와 액션에 대해 실행되는 전역 핸들러
    2. 리소스의 모든 액션에 대해 실행되는 리소스별 핸들러
    3. 세밀한 제어를 위한 리소스 및 액션별 핸들러

    각 핸들러는 두 개의 매개변수를 받는 비동기 함수여야 합니다:
    - ctx (AuthContext): 요청 컨텍스트와 인증된 사용자 정보를 포함
    - value: 권한 부여되는 데이터 (타입은 엔드포인트에 따라 다름)

    핸들러는 다음 중 하나를 반환해야 합니다:
        - None 또는 True: 요청 수락
        - False: 403 오류로 거부
        - FilterType: 응답에 필터링 규칙 적용

    ???+ example "예제"

        모든 요청에 대한 전역 핸들러:

        ```python
        @auth.on
        async def log_all_requests(ctx: AuthContext, value: Any) -> None:
            print(f"Request to {ctx.path} by {ctx.user.identity}")
            return True
        ```

        리소스별 핸들러:

        ```python
        @auth.on.threads
        async def check_thread_access(ctx: AuthContext, value: Any) -> bool:
            # 사용자가 생성한 스레드에만 접근 허용
            return value.get("created_by") == ctx.user.identity
        ```

        리소스 및 액션별 핸들러:

        ```python
        @auth.on.threads.delete
        async def prevent_thread_deletion(ctx: AuthContext, value: Any) -> bool:
            # 관리자만 스레드를 삭제할 수 있음
            return "admin" in ctx.user.permissions
        ```

        여러 리소스 또는 액션:

        ```python
        @auth.on(resources=["threads", "runs"], actions=["create", "update"])
        async def rate_limit_writes(ctx: AuthContext, value: Any) -> bool:
            # 쓰기 작업에 대한 속도 제한 구현
            return await check_rate_limit(ctx.user.identity)
        ```
    """

    __slots__ = (
        "_auth",
        "assistants",
        "threads",
        "runs",
        "crons",
        "store",
        "value",
    )

    def __init__(self, auth: Auth) -> None:
        self._auth = auth
        self.assistants = _AssistantsOn(auth, "assistants")
        self.threads = _ThreadsOn(auth, "threads")
        self.crons = _CronsOn(auth, "crons")
        self.store = _StoreOn(auth)
        self.value = dict[str, typing.Any]

    @typing.overload
    def __call__(
        self,
        *,
        resources: str | Sequence[str],
        actions: str | Sequence[str] | None = None,
    ) -> Callable[[AHO], AHO]: ...

    @typing.overload
    def __call__(self, fn: AHO) -> AHO: ...

    def __call__(
        self,
        fn: AHO | None = None,
        *,
        resources: str | Sequence[str] | None = None,
        actions: str | Sequence[str] | None = None,
    ) -> AHO | Callable[[AHO], AHO]:
        """특정 리소스와 액션에 대한 핸들러를 등록합니다.

        데코레이터로 사용하거나 명시적인 리소스/액션 매개변수와 함께 사용할 수 있습니다:

        @auth.on
        async def handler(): ...  # 전역 핸들러

        @auth.on(resources="threads")
        async def handler(): ...  # 모든 스레드 액션에 대한 types.Handler

        @auth.on(resources="threads", actions="create")
        async def handler(): ...  # 스레드 생성에 대한 types.Handler
        """
        if fn is not None:
            # 일반 데코레이터로 사용됨
            _register_handler(self._auth, None, None, fn)
            return fn

        # 매개변수와 함께 사용됨, 데코레이터 반환
        def decorator(
            handler: AHO,
        ) -> AHO:
            if isinstance(resources, str):
                resource_list = [resources]
            else:
                resource_list = list(resources) if resources is not None else ["*"]

            if isinstance(actions, str):
                action_list = [actions]
            else:
                action_list = list(actions) if actions is not None else ["*"]
            for resource in resource_list:
                for action in action_list:
                    _register_handler(self._auth, resource, action, handler)
            return handler

        return decorator


def _register_handler(
    auth: Auth,
    resource: str | None,
    action: str | None,
    fn: types.Handler,
) -> types.Handler:
    _validate_handler(fn)
    resource = resource or "*"
    action = action or "*"
    if resource == "*" and action == "*":
        if auth._global_handlers:
            raise ValueError("전역 핸들러가 이미 설정되어 있습니다.")
        auth._global_handlers.append(fn)
    else:
        r = resource if resource is not None else "*"
        a = action if action is not None else "*"
        if (r, a) in auth._handlers:
            raise ValueError(f"{r}, {a}에 대한 types.Handler가 이미 설정되어 있습니다.")
        auth._handlers[(r, a)] = [fn]
    return fn


def _validate_handler(fn: Callable[..., typing.Any]) -> None:
    """auth 핸들러 함수가 필요한 시그니처를 충족하는지 검증합니다.

    Auth 핸들러는 다음을 만족해야 합니다:
    1. 비동기 함수여야 합니다
    2. AuthContext 타입의 ctx 매개변수를 받아야 합니다
    3. 권한 부여되는 데이터를 위한 value 매개변수를 받아야 합니다
    """
    if not inspect.iscoroutinefunction(fn):
        raise ValueError(
            f"Auth 핸들러 '{getattr(fn, '__name__', fn)}'는 비동기 함수여야 합니다. "
            "'def' 앞에 'async'를 추가하여 비동기로 만들고 "
            "모든 IO 작업이 논블로킹 방식인지 확인하세요."
        )

    sig = inspect.signature(fn)
    if "ctx" not in sig.parameters:
        raise ValueError(
            f"Auth 핸들러 '{getattr(fn, '__name__', fn)}'는 'ctx: AuthContext' 매개변수를 가져야 합니다. "
            "이 필수 매개변수를 포함하도록 함수 시그니처를 업데이트하세요."
        )
    if "value" not in sig.parameters:
        raise ValueError(
            f"Auth 핸들러 '{getattr(fn, '__name__', fn)}'는 'value' 매개변수를 가져야 합니다. "
            "value는 엔드포인트로 전송되는 가변 데이터를 포함합니다. "
            "이 필수 매개변수를 포함하도록 함수 시그니처를 업데이트하세요."
        )


def is_studio_user(
    user: types.MinimalUser | types.BaseUser | types.MinimalUserDict,
) -> bool:
    return (
        isinstance(user, types.StudioUser)
        or isinstance(user, dict)
        and user.get("kind") == "StudioUser"  # ty: ignore[invalid-argument-type]
    )


__all__ = ["Auth", "types", "exceptions"]
