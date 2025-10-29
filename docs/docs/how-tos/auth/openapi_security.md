# OpenAPI에서 API 인증 문서화

이 가이드는 LangGraph Platform API 문서에 대한 OpenAPI 보안 스키마를 커스터마이징하는 방법을 보여줍니다. 잘 문서화된 보안 스키마는 API 사용자가 API로 인증하는 방법을 이해하는 데 도움이 되며 자동 클라이언트 생성도 가능하게 합니다. LangGraph의 인증 시스템에 대한 자세한 내용은 [Authentication & Access Control 개념 가이드](../../concepts/auth.md)를 참조하세요.

!!! note "구현 vs 문서화"

    이 가이드는 OpenAPI에서 보안 요구 사항을 문서화하는 방법만 다룹니다. 실제 인증 로직을 구현하려면 [커스텀 인증 추가 방법](./custom_auth.md)을 참조하세요.

이 가이드는 모든 LangGraph Platform 배포(Cloud 및 self-hosted)에 적용됩니다. LangGraph Platform을 사용하지 않는 경우 LangGraph 오픈소스 라이브러리 사용에는 적용되지 않습니다.

## 기본 스키마

기본 보안 스키마는 배포 유형에 따라 다릅니다:

=== "LangGraph Platform"

기본적으로 LangGraph Platform은 `x-api-key` 헤더에 LangSmith API 키가 필요합니다:

```yaml
components:
  securitySchemes:
    apiKeyAuth:
      type: apiKey
      in: header
      name: x-api-key
security:
  - apiKeyAuth: []
```

LangGraph SDK 중 하나를 사용할 때 이는 환경 변수에서 추론될 수 있습니다.

=== "Self-hosted"

기본적으로 self-hosted 배포에는 보안 스키마가 없습니다. 즉, 보안 네트워크에만 배포하거나 인증과 함께 배포해야 합니다. 커스텀 인증을 추가하려면 [커스텀 인증 추가 방법](./custom_auth.md)을 참조하세요.

## 커스텀 보안 스키마

OpenAPI 문서에서 보안 스키마를 커스터마이징하려면 `langgraph.json`의 `auth` 구성에 `openapi` 필드를 추가합니다. 이는 API 문서만 업데이트한다는 점을 기억하세요 - [커스텀 인증 추가 방법](./custom_auth.md)에 표시된 대로 해당 인증 로직도 구현해야 합니다.

LangGraph Platform은 인증 엔드포인트를 제공하지 않습니다 - 클라이언트 애플리케이션에서 사용자 인증을 처리하고 결과 자격 증명을 LangGraph API에 전달해야 합니다.

:::python
=== "OAuth2 with Bearer Token"

    ```json
    {
      "auth": {
        "path": "./auth.py:my_auth",  // 여기에 인증 로직 구현
        "openapi": {
          "securitySchemes": {
            "OAuth2": {
              "type": "oauth2",
              "flows": {
                "implicit": {
                  "authorizationUrl": "https://your-auth-server.com/oauth/authorize",
                  "scopes": {
                    "me": "Read information about the current user",
                    "threads": "Access to create and manage threads"
                  }
                }
              }
            }
          },
          "security": [
            {"OAuth2": ["me", "threads"]}
          ]
        }
      }
    }
    ```

=== "API Key"

    ```json
    {
      "auth": {
        "path": "./auth.py:my_auth",  // 여기에 인증 로직 구현
        "openapi": {
          "securitySchemes": {
            "apiKeyAuth": {
              "type": "apiKey",
              "in": "header",
              "name": "X-API-Key"
            }
          },
          "security": [
            {"apiKeyAuth": []}
          ]
        }
      }
    }
    ```

:::

:::js
=== "OAuth2 with Bearer Token"

    ```json
    {
      "auth": {
        "path": "./auth.ts:my_auth",  // 여기에 인증 로직 구현
        "openapi": {
          "securitySchemes": {
            "OAuth2": {
              "type": "oauth2",
              "flows": {
                "implicit": {
                  "authorizationUrl": "https://your-auth-server.com/oauth/authorize",
                  "scopes": {
                    "me": "Read information about the current user",
                    "threads": "Access to create and manage threads"
                  }
                }
              }
            }
          },
          "security": [
            {"OAuth2": ["me", "threads"]}
          ]
        }
      }
    }
    ```

=== "API Key"

    ```json
    {
      "auth": {
        "path": "./auth.ts:my_auth",  // 여기에 인증 로직 구현
        "openapi": {
          "securitySchemes": {
            "apiKeyAuth": {
              "type": "apiKey",
              "in": "header",
              "name": "X-API-Key"
            }
          },
          "security": [
            {"apiKeyAuth": []}
          ]
        }
      }
    }
    ```

:::

## 테스트

구성을 업데이트한 후:

1. 애플리케이션을 배포합니다
2. `/docs`를 방문하여 업데이트된 OpenAPI 문서를 확인합니다
3. 인증 서버의 자격 증명을 사용하여 엔드포인트를 시도합니다 (먼저 인증 로직을 구현했는지 확인하세요)
