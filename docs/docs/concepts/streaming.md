---
search:
boost: 2
---

# 스트리밍

LangGraph는 실시간 업데이트를 표면화하여 반응형이고 투명한 사용자 경험을 가능하게 하는 스트리밍 시스템을 구현합니다.

LangGraph의 스트리밍 시스템을 사용하면 그래프 실행에서 앱으로 실시간 피드백을 표시할 수 있습니다.
스트리밍할 수 있는 데이터에는 세 가지 주요 범주가 있습니다:

1. **워크플로우 진행 상황** — 각 그래프 노드가 실행된 후 상태 업데이트를 가져옵니다.
2. **LLM 토큰** — 언어 모델 토큰이 생성되는 대로 스트리밍합니다.
3. **사용자 정의 업데이트** — 사용자 정의 신호를 발행합니다(예: "100개 중 10개 레코드 가져옴").

## LangGraph 스트리밍으로 가능한 것

- [**LLM 토큰 스트리밍**](../how-tos/streaming.md#messages) — 노드, 서브그래프 또는 도구 내부 어디에서나 토큰 스트림을 캡처합니다.
- [**도구에서 진행 상황 알림 발행**](../how-tos/streaming.md#stream-custom-data) — 도구 함수에서 직접 사용자 정의 업데이트 또는 진행 상황 신호를 보냅니다.
- [**서브그래프에서 스트리밍**](../how-tos/streaming.md#stream-subgraph-outputs) — 부모 그래프와 중첩된 서브그래프 모두의 출력을 포함합니다.
- [**모든 LLM 사용**](../how-tos/streaming.md#use-with-any-llm) — `custom` 스트리밍 모드를 사용하여 LangChain 모델이 아니더라도 모든 LLM에서 토큰을 스트리밍합니다.
- [**여러 스트리밍 모드 사용**](../how-tos/streaming.md#stream-multiple-modes) — `values`(전체 상태), `updates`(상태 델타), `messages`(LLM 토큰 + 메타데이터), `custom`(임의 사용자 데이터) 또는 `debug`(상세 추적) 중에서 선택합니다.
