# 프롬프트 반복

## 개요

LangGraph Studio는 그래프에서 프롬프트를 수정하는 두 가지 방법을 지원합니다: 직접 노드 편집 및 LangSmith Playground 인터페이스.

## 직접 노드 편집

Studio를 사용하면 그래프 인터페이스에서 직접 개별 노드 내부에서 사용되는 프롬프트를 편집할 수 있습니다.

!!! info "사전 요구사항"

    - [어시스턴트 개요](../../concepts/assistants.md)

### 그래프 구성

`langgraph_nodes` 및 `langgraph_type` 키를 사용하여 프롬프트 필드와 관련 노드를 지정하도록 [구성](https://langchain-ai.github.io/langgraph/how-tos/configuration/)을 정의합니다.

#### 구성 레퍼런스

##### `langgraph_nodes`

- **설명**: 구성 필드가 연결된 그래프의 노드를 지정합니다.
- **값 타입**: 문자열 배열, 각 문자열은 그래프의 노드 이름입니다.
- **사용 컨텍스트**: Pydantic 모델의 경우 `json_schema_extra` 딕셔너리에, 데이터클래스의 경우 `metadata["json_schema_extra"]` 딕셔너리에 포함합니다.
- **예제**:
  ```python
  system_prompt: str = Field(
      default="You are a helpful AI assistant.",
      json_schema_extra={"langgraph_nodes": ["call_model", "other_node"]},
  )
  ```

##### `langgraph_type`

- **설명**: 구성 필드의 타입을 지정하며, UI에서 처리되는 방식을 결정합니다.
- **값 타입**: 문자열
- **지원되는 값**:
  - `"prompt"`: 필드가 UI에서 특별히 처리되어야 하는 프롬프트 텍스트를 포함함을 나타냅니다.
- **사용 컨텍스트**: Pydantic 모델의 경우 `json_schema_extra` 딕셔너리에, 데이터클래스의 경우 `metadata["json_schema_extra"]` 딕셔너리에 포함합니다.
- **예제**:
  ```python
  system_prompt: str = Field(
      default="You are a helpful AI assistant.",
      json_schema_extra={
          "langgraph_nodes": ["call_model"],
          "langgraph_type": "prompt",
      },
  )
  ```

#### 구성 예제

```python
## Pydantic 사용
from pydantic import BaseModel, Field
from typing import Annotated, Literal

class Configuration(BaseModel):
    """The configuration for the agent."""

    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="The system prompt to use for the agent's interactions. "
        "This prompt sets the context and behavior for the agent.",
        json_schema_extra={
            "langgraph_nodes": ["call_model"],
            "langgraph_type": "prompt",
        },
    )

    model: Annotated[
        Literal[
            "anthropic/claude-3-7-sonnet-latest",
            "anthropic/claude-3-5-haiku-latest",
            "openai/o1",
            "openai/gpt-4o-mini",
            "openai/o1-mini",
            "openai/o3-mini",
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = Field(
        default="openai/gpt-4o-mini",
        description="The name of the language model to use for the agent's main interactions. "
        "Should be in the form: provider/model-name.",
        json_schema_extra={"langgraph_nodes": ["call_model"]},
    )

## 데이터클래스 사용
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default="You are a helpful AI assistant.",
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

```

### UI에서 프롬프트 편집

1. 연결된 구성 필드가 있는 노드에서 기어 아이콘을 찾습니다
2. 클릭하여 구성 모달을 엽니다
3. 값을 편집합니다
4. 저장하여 현재 어시스턴트 버전을 업데이트하거나 새 버전을 생성합니다

## LangSmith Playground

[LangSmith Playground](https://docs.smith.langchain.com/prompt_engineering/how_to_guides#playground) 인터페이스를 사용하면 전체 그래프를 실행하지 않고도 개별 LLM 호출을 테스트할 수 있습니다:

1. thread를 선택합니다
2. 노드에서 "View LLM Runs"를 클릭합니다. 노드 내부에서 수행된 모든 LLM 호출(있는 경우)이 나열됩니다.
3. Playground에서 열 LLM run을 선택합니다
4. 프롬프트를 수정하고 다양한 모델 및 도구 설정을 테스트합니다
5. 업데이트된 프롬프트를 그래프에 다시 복사합니다

고급 Playground 기능을 사용하려면 오른쪽 상단의 확장 버튼을 클릭하세요.
