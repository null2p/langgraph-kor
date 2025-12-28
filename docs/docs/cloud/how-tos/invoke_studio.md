# 애플리케이션 실행

!!!info  "사전 요구사항"
    - [에이전트 실행하기](../../agents/run_agents.md#running-agents)

이 가이드는 애플리케이션에 [run](../../concepts/assistants.md#execution)을 제출하는 방법을 보여줍니다.

## 그래프 모드 {#graph-mode}

### 입력 지정
먼저 페이지 왼쪽의 그래프 인터페이스 아래 "Input" 섹션에서 그래프에 대한 입력을 정의합니다.

Studio는 그래프에 정의된 [state schema](../../concepts/low_level.md/#schema)를 기반으로 입력 폼을 렌더링하려고 시도합니다. 이를 비활성화하려면 "View Raw" 버튼을 클릭하면 JSON 에디터가 제공됩니다.

"Input" 섹션 상단의 위/아래 화살표를 클릭하면 이전에 제출한 입력들을 토글하고 사용할 수 있습니다.

### Run 설정

#### Assistant

Run에 사용할 [assistant](../../concepts/assistants.md)를 지정하려면 왼쪽 하단의 설정 버튼을 클릭합니다. assistant가 현재 선택되어 있으면 버튼에 assistant 이름도 표시됩니다. assistant가 선택되지 않은 경우 "Manage Assistants"라고 표시됩니다.

실행할 assistant를 선택하고 모달 상단의 "Active" 토글을 클릭하여 활성화합니다. assistant 관리에 대한 자세한 내용은 [여기](./studio/manage_assistants.md)를 참조하세요.

#### Streaming
"Submit" 옆의 드롭다운을 클릭하고 토글을 클릭하여 스트리밍을 활성화/비활성화합니다.

#### Breakpoints
breakpoint와 함께 그래프를 실행하려면 "Interrupt" 버튼을 클릭합니다. 노드를 선택하고 해당 노드가 실행되기 전 및/또는 후에 일시 중지할지 여부를 선택합니다. 실행을 재개하려면 thread 로그에서 "Continue"를 클릭합니다.


breakpoint에 대한 자세한 내용은 [여기](../../concepts/human_in_the_loop.md)를 참조하세요.

### Run 제출

지정한 입력과 run 설정으로 run을 제출하려면 "Submit" 버튼을 클릭합니다. 이렇게 하면 기존에 선택한 [thread](../../concepts/persistence.md#threads)에 [run](../../concepts/assistants.md#execution)이 추가됩니다. 현재 선택된 thread가 없으면 새 thread가 생성됩니다.

진행 중인 run을 취소하려면 "Cancel" 버튼을 클릭합니다.


## 챗 모드 {#chat-mode}
대화 패널 하단에서 챗 애플리케이션에 대한 입력을 지정합니다. "Send message" 버튼을 클릭하면 입력이 사람 메시지로 제출되고 응답이 스트리밍됩니다.

진행 중인 run을 취소하려면 "Cancel" 버튼을 클릭합니다. "Show tool calls" 토글을 클릭하면 대화에서 tool call을 숨기거나 표시할 수 있습니다.

## 더 알아보기

기존 thread의 특정 checkpoint에서 애플리케이션을 실행하는 방법은 [이 가이드](./threads_studio.md#edit-thread-history)를 참조하세요.
