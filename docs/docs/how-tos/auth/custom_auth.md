# 커스텀 인증 추가

!!! tip "사전 요구사항"

    이 가이드는 다음 개념에 익숙하다고 가정합니다:

      *  [**인증 및 접근 제어**](../../concepts/auth.md)
      *  [**LangGraph Platform**](../../concepts/langgraph_platform.md)

    보다 자세한 안내는 [**커스텀 인증 설정**](../../tutorials/auth/getting_started.md) 튜토리얼을 참조하세요.

???+ note "배포 유형별 지원"

    커스텀 인증은 **관리형 LangGraph Platform**의 모든 배포와 **Enterprise** 셀프 호스팅 플랜에서 지원됩니다.

이 가이드는 LangGraph Platform 애플리케이션에 커스텀 인증을 추가하는 방법을 보여줍니다. 이 가이드는 LangGraph Platform과 셀프 호스팅 배포 모두에 적용됩니다. 자체 커스텀 서버에서 LangGraph 오픈 소스 라이브러리를 독립적으로 사용하는 경우에는 적용되지 않습니다.

!!! note

    커스텀 인증은 모든 **관리형 LangGraph Platform** 배포와 **Enterprise** 셀프 호스팅 플랜에서 지원됩니다.

## 배포에 커스텀 인증 추가

배포에서 커스텀 인증을 활용하고 사용자 레벨 메타데이터에 접근하려면 커스텀 인증 핸들러를 통해 `config["configurable"]["langgraph_auth_user"]` 객체를 자동으로 채우도록 커스텀 인증을 설정하세요. 그런 다음 `langgraph_auth_user` 키로 그래프에서 이 객체에 접근하여 [에이전트가 사용자를 대신하여 인증된 작업을 수행](#enable-agent-authentication)할 수 있도록 할 수 있습니다.

:::python

1.  Implement authentication:

    !!! note

        Without a custom `@auth.authenticate` handler, LangGraph sees only the API-key owner (usually the developer), so requests aren’t scoped to individual end-users. To propagate custom tokens, you must implement your own handler.

    ```python
    from langgraph_sdk import Auth
    import requests

    auth = Auth()

    def is_valid_key(api_key: str) -> bool:
        is_valid = # your API key validation logic
        return is_valid

    @auth.authenticate # (1)!
    async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
        api_key = headers.get("x-api-key")
        if not api_key or not is_valid_key(api_key):
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid API key")

        # Fetch user-specific tokens from your secret store
        user_tokens = await fetch_user_tokens(api_key)

        return { # (2)!
            "identity": api_key,  #  fetch user ID from LangSmith
            "github_token" : user_tokens.github_token
            "jira_token" : user_tokens.jira_token
            # ... custom fields/secrets here
        }
    ```

    1. This handler receives the request (headers, etc.), validates the user, and returns a dictionary with at least an identity field.
    2. You can add any custom fields you want (e.g., OAuth tokens, roles, org IDs, etc.).

2.  In your `langgraph.json`, add the path to your auth file:

    ```json hl_lines="7-9"
    {
      "dependencies": ["."],
      "graphs": {
        "agent": "./agent.py:graph"
      },
      "env": ".env",
      "auth": {
        "path": "./auth.py:my_auth"
      }
    }
    ```

3.  Once you've set up authentication in your server, requests must include the required authorization information based on your chosen scheme. Assuming you are using JWT token authentication, you could access your deployments using any of the following methods:

    === "Python Client"

        ```python
        from langgraph_sdk import get_client

        my_token = "your-token" # In practice, you would generate a signed token with your auth provider
        client = get_client(
            url="http://localhost:2024",
            headers={"Authorization": f"Bearer {my_token}"}
        )
        threads = await client.threads.search()
        ```

    === "Python RemoteGraph"

        ```python
        from langgraph.pregel.remote import RemoteGraph

        my_token = "your-token" # In practice, you would generate a signed token with your auth provider
        remote_graph = RemoteGraph(
            "agent",
            url="http://localhost:2024",
            headers={"Authorization": f"Bearer {my_token}"}
        )
        threads = await remote_graph.ainvoke(...)
        ```
        ```python
        from langgraph.pregel.remote import RemoteGraph

        my_token = "your-token" # In practice, you would generate a signed token with your auth provider
        remote_graph = RemoteGraph(
            "agent",
            url="http://localhost:2024",
            headers={"Authorization": f"Bearer {my_token}"}
        )
        threads = await remote_graph.ainvoke(...)
        ```

    === "CURL"

        ```bash
        curl -H "Authorization: Bearer ${your-token}" http://localhost:2024/threads
        ```

## 에이전트 인증 활성화

[인증](#add-custom-authentication-to-your-deployment) 후, 플랫폼은 LangGraph Platform 배포에 전달되는 특별한 구성 객체(`config`)를 생성합니다. 이 객체에는 `@auth.authenticate` 핸들러에서 반환하는 커스텀 필드를 포함하여 현재 사용자에 대한 정보가 포함됩니다.

에이전트가 사용자를 대신하여 인증된 작업을 수행할 수 있도록 하려면 `langgraph_auth_user` 키로 그래프에서 이 객체에 접근하세요:

```python
def my_node(state, config):
    user_config = config["configurable"].get("langgraph_auth_user")
    # token was resolved during the @auth.authenticate function
    token = user_config.get("github_token","")
    ...
```

!!! note

    안전한 비밀 저장소에서 사용자 자격 증명을 가져오세요. 그래프 상태에 비밀을 저장하는 것은 권장되지 않습니다.

### Studio 사용자 권한 부여

기본적으로 리소스에 커스텀 권한 부여를 추가하면 Studio에서 수행되는 상호작용에도 적용됩니다. 원하는 경우 [is_studio_user()](../../reference/functions/sdk_auth.isStudioUser.html)를 확인하여 로그인한 Studio 사용자를 다르게 처리할 수 있습니다.

!!! note
    `is_studio_user`는 langgraph-sdk 버전 0.1.73에서 추가되었습니다. 이전 버전을 사용하는 경우에도 `isinstance(ctx.user, StudioUser)`를 확인할 수 있습니다.

```python
from langgraph_sdk.auth import is_studio_user, Auth
auth = Auth()

# ... Setup authenticate, etc.

@auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,
    value: dict  # The payload being sent to this access method
) -> dict:  # Returns a filter dict that restricts access to resources
    if is_studio_user(ctx.user):
        return {}

    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
```

관리형 LangGraph Platform SaaS에 배포된 그래프에 대한 개발자 접근을 허용하려는 경우에만 사용하세요.

:::

:::js

1.  Implement authentication:

    !!! note

        Without a custom `authenticate` handler, LangGraph sees only the API-key owner (usually the developer), so requests aren’t scoped to individual end-users. To propagate custom tokens, you must implement your own handler.

    ```typescript
    import { Auth, HTTPException } from "@langchain/langgraph-sdk/auth";

    const auth = new Auth()
      .authenticate(async (request) => {
        const authorization = request.headers.get("Authorization");
        const token = authorization?.split(" ")[1]; // "Bearer <token>"
        if (!token) {
          throw new HTTPException(401, "No token provided");
        }
        try {
          const user = await verifyToken(token);
          return user;
        } catch (error) {
          throw new HTTPException(401, "Invalid token");
        }
      })
      // Add authorization rules to actually control access to resources
      .on("*", async ({ user, value }) => {
        const filters = { owner: user.identity };
        const metadata = value.metadata ?? {};
        metadata.update(filters);
        return filters;
      })
      // Assumes you organize information in store like (user_id, resource_type, resource_id)
      .on("store", async ({ user, value }) => {
        const namespace = value.namespace;
        if (namespace[0] !== user.identity) {
          throw new HTTPException(403, "Not authorized");
        }
      });
    ```

    1. This handler receives the request (headers, etc.), validates the user, and returns an object with at least an identity field.
    2. You can add any custom fields you want (e.g., OAuth tokens, roles, org IDs, etc.).

2.  In your `langgraph.json`, add the path to your auth file:

    ```json hl_lines="7-9"
    {
      "dependencies": ["."],
      "graphs": {
        "agent": "./agent.ts:graph"
      },
      "env": ".env",
      "auth": {
        "path": "./auth.ts:my_auth"
      }
    }
    ```

3.  Once you've set up authentication in your server, requests must include the required authorization information based on your chosen scheme. Assuming you are using JWT token authentication, you could access your deployments using any of the following methods:

    === "SDK Client"

        ```javascript
        import { Client } from "@langchain/langgraph-sdk";

        const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
        const client = new Client({
          apiUrl: "http://localhost:2024",
          defaultHeaders: { Authorization: `Bearer ${my_token}` },
        });
        const threads = await client.threads.search();
        ```

    === "RemoteGraph"

        ```javascript
        import { RemoteGraph } from "@langchain/langgraph/remote";

        const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
        const remoteGraph = new RemoteGraph({
        graphId: "agent",
          url: "http://localhost:2024",
          headers: { Authorization: `Bearer ${my_token}` },
        });
        const threads = await remoteGraph.invoke(...);
        ```

    === "CURL"

        ```bash
        curl -H "Authorization: Bearer ${your-token}" http://localhost:2024/threads
        ```

## Enable agent authentication

After [authentication](#add-custom-authentication-to-your-deployment), the platform creates a special configuration object (`config`) that is passed to LangGraph Platform deployment. This object contains information about the current user, including any custom fields you return from your `authenticate` handler.

To allow an agent to perform authenticated actions on behalf of the user, access this object in your graph with the `langgraph_auth_user` key:

```ts
async function myNode(state, config) {
  const userConfig = config["configurable"]["langgraph_auth_user"];
  // token was resolved during the authenticate function
  const token = userConfig["github_token"];
  ...
}
```

!!! note

    Fetch user credentials from a secure secret store. Storing secrets in graph state is not recommended.

:::

## Learn more

- [Authentication & Access Control](../../concepts/auth.md)
- [LangGraph Platform](../../concepts/langgraph_platform.md)
- [Setting up custom authentication tutorial](../../tutorials/auth/getting_started.md)
