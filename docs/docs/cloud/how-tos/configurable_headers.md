# Configurable Headers

LangGraph는 에이전트 동작과 권한을 동적으로 수정하기 위한 런타임 구성을 허용합니다. [LangGraph Platform](../quick_start.md)을 사용할 때 이 구성을 요청 본문(`config`)이나 특정 요청 헤더로 전달할 수 있습니다. 이를 통해 사용자 ID나 기타 요청 데이터를 기반으로 조정할 수 있습니다.

프라이버시를 위해 `langgraph.json` 파일의 `http.configurable_headers` 섹션을 통해 런타임 구성에 전달되는 헤더를 제어합니다.

다음은 포함 및 제외 헤더를 커스터마이징하는 방법입니다:

```json
{
  "http": {
    "configurable_headers": {
      "include": ["x-user-id", "x-organization-id", "my-prefix-*"],
      "exclude": ["authorization", "x-api-key"]
    }
  }
}
```


`include` 및 `exclude` 목록은 정확한 헤더 이름 또는 `*`를 사용하여 임의의 문자 수와 일치하는 패턴을 허용합니다. 보안을 위해 다른 정규식 패턴은 지원되지 않습니다.

## 그래프 내에서 사용하기

그래프의 모든 노드의 `config` 인수를 사용하여 포함된 헤더에 액세스할 수 있습니다.

```python
def my_node(state, config):
  organization_id = config["configurable"].get("x-organization-id")
  ...
```

또는 컨텍스트에서 가져올 수 있습니다(도구 내부나 다른 중첩 함수 내에서 유용합니다).

```python
from langgraph.config import get_config

def search_everything(query: str):
  organization_id = get_config()["configurable"].get("x-organization-id")
  ...
```


이를 사용하여 그래프를 동적으로 컴파일할 수도 있습니다.

```python
# my_graph.py.
import contextlib

@contextlib.asynccontextmanager
async def generate_agent(config):
  organization_id = config["configurable"].get("x-organization-id")
  if organization_id == "org1":
    graph = ...
    yield graph
  else:
    graph = ...
    yield graph

```

```json
{
  "graphs": {"agent": "my_grph.py:generate_agent"}
}
```

### Configurable headers 옵트아웃

configurable headers를 옵트아웃하려면 `exclude` 목록에 와일드카드 패턴을 설정하기만 하면 됩니다:

```json
{
  "http": {
    "configurable_headers": {
      "exclude": ["*"]
    }
  }
}
```

이렇게 하면 모든 헤더가 run의 구성에 추가되지 않습니다.

제외가 포함보다 우선한다는 점에 유의하세요.
