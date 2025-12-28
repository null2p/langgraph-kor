# SQL 에이전트 구축

이 튜토리얼에서는 SQL 데이터베이스에 대한 질문에 답할 수 있는 에이전트를 구축하는 방법을 안내합니다.

높은 수준에서 에이전트는 다음을 수행합니다:

1. 데이터베이스에서 사용 가능한 테이블 가져오기
2. 질문과 관련된 테이블 결정하기
3. 관련 테이블의 스키마 가져오기
4. 질문과 스키마 정보를 기반으로 쿼리 생성하기
5. LLM을 사용하여 일반적인 실수가 있는지 쿼리 재확인하기
6. 쿼리를 실행하고 결과 반환하기
7. 쿼리가 성공할 때까지 데이터베이스 엔진에서 발견된 오류 수정하기
8. 결과를 기반으로 응답 작성하기

!!! warning "보안 참고사항"
    SQL 데이터베이스의 Q&A 시스템을 구축하려면 모델이 생성한 SQL 쿼리를 실행해야 합니다. 이에는 내재적인 위험이 있습니다. 데이터베이스 연결 권한이 항상 에이전트의 필요에 맞게 가능한 한 좁게 범위가 지정되어 있는지 확인하세요. 이렇게 하면 모델 기반 시스템 구축의 위험을 완화할 수 있지만 제거할 수는 없습니다.

## 1. 설정

먼저 몇 가지 의존성을 설치해봅시다. 이 튜토리얼은 [langchain-community](https://python.langchain.com/docs/concepts/architecture/#langchain-community)의 SQL 데이터베이스 및 도구 추상화를 사용합니다. 또한 LangChain [채팅 모델](https://python.langchain.com/docs/concepts/chat_models/)도 필요합니다.

```python
%%capture --no-stderr
%pip install -U langgraph langchain_community "langchain[openai]"
```

!!! tip
    LangSmith에 가입하여 LangGraph 프로젝트의 문제를 신속하게 발견하고 성능을 개선하세요. [LangSmith](https://docs.smith.langchain.com)를 사용하면 추적 데이터를 활용하여 LangGraph로 구축한 LLM 앱을 디버그, 테스트 및 모니터링할 수 있습니다.

### LLM 선택

먼저 [LLM을 초기화](https://python.langchain.com/docs/how_to/chat_models_universal_init/)합니다. [도구 호출](https://python.langchain.com/docs/integrations/chat/#featured-providers)을 지원하는 모든 모델이 작동해야 합니다. 아래에서는 OpenAI를 사용합니다.

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4.1")
```

### 데이터베이스 구성

이 튜토리얼을 위해 SQLite 데이터베이스를 생성합니다. SQLite는 설정하고 사용하기 쉬운 경량 데이터베이스입니다. 디지털 미디어 스토어를 나타내는 샘플 데이터베이스인 `chinook` 데이터베이스를 로드합니다.
데이터베이스에 대한 자세한 정보는 [여기](https://www.sqlitetutorial.net/sqlite-sample-database/)에서 확인할 수 있습니다.

편의를 위해 공개 GCS 버킷에 데이터베이스(`Chinook.db`)를 호스팅했습니다.

```python
import requests

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

response = requests.get(url)

if response.status_code == 200:
    # 바이너리 쓰기 모드로 로컬 파일 열기
    with open("Chinook.db", "wb") as file:
        # 응답의 내용(파일)을 로컬 파일에 작성
        file.write(response.content)
    print("File downloaded and saved as Chinook.db")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")
```

데이터베이스와 상호작용하기 위해 `langchain_community` 패키지에서 사용할 수 있는 편리한 SQL 데이터베이스 래퍼를 사용합니다. 이 래퍼는 SQL 쿼리를 실행하고 결과를 가져오는 간단한 인터페이스를 제공합니다:

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')
```

**출력:**
```
Dialect: sqlite
Available tables: ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']
Sample output: [(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), (4, 'Alanis Morissette'), (5, 'Alice In Chains')]
```

### 데이터베이스 상호작용을 위한 도구

`langchain-community`는 우리의 `SQLDatabase`와 상호작용하기 위한 몇 가지 내장 도구를 구현하고 있으며, 테이블 나열, 테이블 스키마 읽기, 쿼리 확인 및 실행을 위한 도구가 포함됩니다:

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")
```

**출력:**
```
sql_db_query: 이 도구의 입력은 상세하고 올바른 SQL 쿼리이며, 출력은 데이터베이스의 결과입니다. 쿼리가 올바르지 않으면 오류 메시지가 반환됩니다. 오류가 반환되면 쿼리를 다시 작성하고 확인한 후 다시 시도하세요. 'field list'에 알 수 없는 컬럼 'xxxx'가 있는 문제가 발생하면 sql_db_schema를 사용하여 올바른 테이블 필드를 쿼리하세요.

sql_db_schema: 이 도구의 입력은 쉼표로 구분된 테이블 목록이며, 출력은 해당 테이블의 스키마와 샘플 행입니다. 먼저 sql_db_list_tables를 호출하여 테이블이 실제로 존재하는지 확인하세요! 예제 입력: table1, table2, table3

sql_db_list_tables: 입력은 빈 문자열이며, 출력은 데이터베이스의 테이블을 쉼표로 구분한 목록입니다.

sql_db_query_checker: 쿼리를 실행하기 전에 쿼리가 올바른지 다시 확인하려면 이 도구를 사용하세요. sql_db_query로 쿼리를 실행하기 전에 항상 이 도구를 사용하세요!

```

## 2. 사전 구축된 에이전트 사용

이러한 도구들이 주어지면 한 줄로 사전 구축된 에이전트를 초기화할 수 있습니다. 에이전트의 동작을 커스터마이즈하기 위해 설명적인 시스템 프롬프트를 작성합니다.

```python
from langgraph.prebuilt import create_react_agent

system_prompt = """
당신은 SQL 데이터베이스와 상호작용하도록 설계된 에이전트입니다.
입력 질문이 주어지면 실행할 구문적으로 올바른 {dialect} 쿼리를 생성한 다음,
쿼리 결과를 보고 답변을 반환하세요. 사용자가 얻고자 하는 특정 예제 수를
지정하지 않는 한 항상 쿼리를 최대 {top_k}개 결과로 제한하세요.

관련 컬럼을 기준으로 결과를 정렬하여 데이터베이스에서 가장 흥미로운
예제를 반환할 수 있습니다. 특정 테이블의 모든 컬럼을 쿼리하지 말고
질문에 따라 관련 컬럼만 요청하세요.

쿼리를 실행하기 전에 반드시 쿼리를 다시 확인해야 합니다. 쿼리를 실행하는 중에
오류가 발생하면 쿼리를 다시 작성하고 다시 시도하세요.

데이터베이스에 DML 문(INSERT, UPDATE, DELETE, DROP 등)을 절대 작성하지 마세요.

시작하려면 항상 먼저 데이터베이스의 테이블을 확인하여 쿼리할 수 있는 것이
무엇인지 확인해야 합니다. 이 단계를 건너뛰지 마세요.

그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)
```

!!! note
    이 시스템 프롬프트는 특정 도구를 항상 다른 도구보다 먼저 또는 나중에 실행하는 것과 같은 여러 지침을 포함합니다. [다음 섹션](#customizing-the-agent)에서는 그래프의 구조를 통해 이러한 동작을 강제하여 더 큰 제어 수준을 제공하고 프롬프트를 단순화할 수 있습니다.

샘플 쿼리에 대해 이 에이전트를 실행하고 동작을 관찰해봅시다:

```python
question = "Which genre on average has the longest tracks?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

**출력:**
```
================================ Human Message =================================

Which genre on average has the longest tracks?
================================== Ai Message ==================================
Tool Calls:
  sql_db_list_tables (call_d8lCgywSroCgpVl558nmXKwA)
 Call ID: call_d8lCgywSroCgpVl558nmXKwA
  Args:
================================= Tool Message =================================
Name: sql_db_list_tables

Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track
================================== Ai Message ==================================
Tool Calls:
  sql_db_schema (call_nNf6IIUcwMYLIkE0l6uWkZHe)
 Call ID: call_nNf6IIUcwMYLIkE0l6uWkZHe
  Args:
    table_names: Genre, Track
================================= Tool Message =================================
Name: sql_db_schema


CREATE TABLE "Genre" (
    "GenreId" INTEGER NOT NULL, 
    "Name" NVARCHAR(120), 
    PRIMARY KEY ("GenreId")
)

/*
3 rows from Genre table:
GenreId Name
1   Rock
2   Jazz
3   Metal
*/


CREATE TABLE "Track" (
    "TrackId" INTEGER NOT NULL, 
    "Name" NVARCHAR(200) NOT NULL, 
    "AlbumId" INTEGER, 
    "MediaTypeId" INTEGER NOT NULL, 
    "GenreId" INTEGER, 
    "Composer" NVARCHAR(220), 
    "Milliseconds" INTEGER NOT NULL, 
    "Bytes" INTEGER, 
    "UnitPrice" NUMERIC(10, 2) NOT NULL, 
    PRIMARY KEY ("TrackId"), 
    FOREIGN KEY("MediaTypeId") REFERENCES "MediaType" ("MediaTypeId"), 
    FOREIGN KEY("GenreId") REFERENCES "Genre" ("GenreId"), 
    FOREIGN KEY("AlbumId") REFERENCES "Album" ("AlbumId")
)

/*
3 rows from Track table:
TrackId Name    AlbumId MediaTypeId GenreId Composer    Milliseconds    Bytes   UnitPrice
1   For Those About To Rock (We Salute You) 1   1   1   Angus Young, Malcolm Young, Brian Johnson   343719  11170334    0.99
2   Balls to the Wall   2   2   1   None    342562  5510424 0.99
3   Fast As a Shark 3   2   1   F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman 230619  3990994 0.99
*/
================================== Ai Message ==================================
Tool Calls:
  sql_db_query_checker (call_urTRmtiGtTxkwHtscec7Fd2K)
 Call ID: call_urTRmtiGtTxkwHtscec7Fd2K
  Args:
    query: SELECT Genre.Name, AVG(Track.Milliseconds) AS AvgMilliseconds
FROM Track
JOIN Genre ON Track.GenreId = Genre.GenreId
GROUP BY Genre.Name
ORDER BY AvgMilliseconds DESC
LIMIT 1;
================================= Tool Message =================================
Name: sql_db_query_checker

\`\`\`sql
SELECT Genre.Name, AVG(Track.Milliseconds) AS AvgMilliseconds
FROM Track
JOIN Genre ON Track.GenreId = Genre.GenreId
GROUP BY Genre.Name
ORDER BY AvgMilliseconds DESC
LIMIT 1;
\`\`\`
================================== Ai Message ==================================
Tool Calls:
  sql_db_query (call_RNMqyUEMv0rvy0UxSwrXY2AV)
 Call ID: call_RNMqyUEMv0rvy0UxSwrXY2AV
  Args:
    query: SELECT Genre.Name, AVG(Track.Milliseconds) AS AvgMilliseconds
FROM Track
JOIN Genre ON Track.GenreId = Genre.GenreId
GROUP BY Genre.Name
ORDER BY AvgMilliseconds DESC
LIMIT 1;
================================= Tool Message =================================
Name: sql_db_query

[('Sci Fi & Fantasy', 2911783.0384615385)]
================================== Ai Message ==================================

평균 트랙 길이가 가장 긴 장르는 "Sci Fi & Fantasy"이며, 트랙당 평균 약 2,911,783밀리초(약 48.5분)입니다.
```

충분히 잘 작동했습니다: 에이전트가 테이블을 올바르게 나열하고, 스키마를 얻고, 쿼리를 작성하고, 쿼리를 확인하고, 최종 응답을 위해 실행했습니다.

!!! tip
    수행한 단계, 호출된 도구, LLM이 본 프롬프트 등을 포함하여 위 실행의 모든 측면을 [LangSmith 추적](https://smith.langchain.com/public/bd594960-73e3-474b-b6f2-db039d7c713a/r)에서 검사할 수 있습니다.

## 3. 에이전트 커스터마이징 { #customizing-the-agent }

사전 구축된 에이전트를 사용하면 빠르게 시작할 수 있지만, 각 단계에서 에이전트는 전체 도구 세트에 액세스할 수 있습니다. 위에서는 시스템 프롬프트를 사용하여 동작을 제한했습니다. 예를 들어, 항상 "list tables" 도구로 시작하고 쿼리를 실행하기 전에 항상 query-checker 도구를 실행하도록 에이전트에 지시했습니다.

에이전트를 커스터마이징하여 LangGraph에서 더 높은 수준의 제어를 강제할 수 있습니다. 아래에서는 특정 도구 호출을 위한 전용 노드가 있는 간단한 ReAct-에이전트 설정을 구현합니다. 사전 구축된 에이전트와 동일한 [상태](../../concepts/low_level.md#state)를 사용합니다.

다음 단계를 위한 전용 노드를 구성합니다:

- DB 테이블 나열
- "get schema" 도구 호출
- 쿼리 생성
- 쿼리 확인

이러한 단계를 전용 노드에 배치하면 (1) 필요할 때 도구 호출을 강제하고 (2) 각 단계와 관련된 프롬프트를 커스터마이즈할 수 있습니다.

```python
from typing import Literal
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")


# 예제: 미리 결정된 도구 호출 생성
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}


# 예제: 모델이 도구 호출을 생성하도록 강제
def call_get_schema(state: MessagesState):
    # LangChain은 모든 모델이 `tool_choice="any"`와
    # `tool_choice=<도구 이름 문자열>`을 받아들이도록 강제합니다.
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


generate_query_system_prompt = """
당신은 SQL 데이터베이스와 상호작용하도록 설계된 에이전트입니다.
입력 질문이 주어지면 실행할 구문적으로 올바른 {dialect} 쿼리를 생성한 다음,
쿼리 결과를 보고 답변을 반환하세요. 사용자가 얻고자 하는 특정 예제 수를
지정하지 않는 한 항상 쿼리를 최대 {top_k}개 결과로 제한하세요.

관련 컬럼을 기준으로 결과를 정렬하여 데이터베이스에서 가장 흥미로운
예제를 반환할 수 있습니다. 특정 테이블의 모든 컬럼을 쿼리하지 말고
질문에 따라 관련 컬럼만 요청하세요.

데이터베이스에 DML 문(INSERT, UPDATE, DELETE, DROP 등)을 절대 작성하지 마세요.
""".format(
    dialect=db.dialect,
    top_k=5,
)


def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # 여기서는 도구 호출을 강제하지 않아 모델이 솔루션을 얻었을 때
    # 자연스럽게 응답할 수 있도록 합니다.
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


check_query_system_prompt = """
당신은 세부 사항에 주의를 기울이는 SQL 전문가입니다.
다음을 포함한 일반적인 실수가 있는지 {dialect} 쿼리를 다시 확인하세요:
- NULL 값과 함께 NOT IN 사용
- UNION ALL을 사용해야 하는데 UNION 사용
- 배타적 범위에 BETWEEN 사용
- 술어의 데이터 타입 불일치
- 식별자를 적절하게 인용
- 함수에 올바른 수의 인수 사용
- 올바른 데이터 타입으로 캐스팅
- 조인에 적절한 컬럼 사용

위의 실수가 있으면 쿼리를 다시 작성하세요. 실수가 없으면
원래 쿼리를 그대로 재현하세요.

이 확인을 실행한 후 적절한 도구를 호출하여 쿼리를 실행합니다.
""".format(dialect=db.dialect)


def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # 확인할 인공 사용자 메시지 생성
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}
```

마지막으로 Graph API를 사용하여 이러한 단계를 워크플로로 조립합니다. 쿼리가 생성되면 쿼리 검사기로 라우팅하고 도구 호출이 없으면 종료하도록 쿼리 생성 단계에서 [조건부 엣지](../../concepts/low_level.md#conditional-edges)를 정의합니다. 이는 LLM이 쿼리에 대한 응답을 전달했음을 의미합니다.

```python
def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"


builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

agent = builder.compile()
```

아래에 애플리케이션을 시각화합니다:

```python
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(Image(agent.get_graph().draw_mermaid_png()))
```

![Graph](./output.png)

**참고:** 이 코드를 실행하면 서로 다른 노드 간의 흐름(list_tables → call_get_schema → get_schema → generate_query → check_query → run_query)을 보여주는 SQL 에이전트 그래프의 시각적 표현이 생성되고 표시됩니다.

이제 이전과 똑같이 그래프를 호출할 수 있습니다:

```python
question = "Which genre on average has the longest tracks?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

**출력:**
```
================================ Human Message =================================

Which genre on average has the longest tracks?
================================== Ai Message ==================================

Available tables: Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track
================================== Ai Message ==================================
Tool Calls:
  sql_db_schema (call_qxKtYiHgf93AiTDin9ez5wFp)
 Call ID: call_qxKtYiHgf93AiTDin9ez5wFp
  Args:
    table_names: Genre,Track
================================= Tool Message =================================
Name: sql_db_schema


CREATE TABLE "Genre" (
    "GenreId" INTEGER NOT NULL, 
    "Name" NVARCHAR(120), 
    PRIMARY KEY ("GenreId")
)

/*
3 rows from Genre table:
GenreId Name
1   Rock
2   Jazz
3   Metal
*/


CREATE TABLE "Track" (
    "TrackId" INTEGER NOT NULL, 
    "Name" NVARCHAR(200) NOT NULL, 
    "AlbumId" INTEGER, 
    "MediaTypeId" INTEGER NOT NULL, 
    "GenreId" INTEGER, 
    "Composer" NVARCHAR(220), 
    "Milliseconds" INTEGER NOT NULL, 
    "Bytes" INTEGER, 
    "UnitPrice" NUMERIC(10, 2) NOT NULL, 
    PRIMARY KEY ("TrackId"), 
    FOREIGN KEY("MediaTypeId") REFERENCES "MediaType" ("MediaTypeId"), 
    FOREIGN KEY("GenreId") REFERENCES "Genre" ("GenreId"), 
    FOREIGN KEY("AlbumId") REFERENCES "Album" ("AlbumId")
)

/*
3 rows from Track table:
TrackId Name    AlbumId MediaTypeId GenreId Composer    Milliseconds    Bytes   UnitPrice
1   For Those About To Rock (We Salute You) 1   1   1   Angus Young, Malcolm Young, Brian Johnson   343719  11170334    0.99
2   Balls to the Wall   2   2   1   None    342562  5510424 0.99
3   Fast As a Shark 3   2   1   F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman 230619  3990994 0.99
*/
================================== Ai Message ==================================
Tool Calls:
  sql_db_query (call_RPN3GABMfb6DTaFTLlwnZxVN)
 Call ID: call_RPN3GABMfb6DTaFTLlwnZxVN
  Args:
    query: SELECT Genre.Name, AVG(Track.Milliseconds) AS AvgTrackLength
FROM Track
JOIN Genre ON Track.GenreId = Genre.GenreId
GROUP BY Genre.GenreId
ORDER BY AvgTrackLength DESC
LIMIT 1;
================================== Ai Message ==================================
Tool Calls:
  sql_db_query (call_PR4s8ymiF3ZQLaoZADXtdqcl)
 Call ID: call_PR4s8ymiF3ZQLaoZADXtdqcl
  Args:
    query: SELECT Genre.Name, AVG(Track.Milliseconds) AS AvgTrackLength
FROM Track
JOIN Genre ON Track.GenreId = Genre.GenreId
GROUP BY Genre.GenreId
ORDER BY AvgTrackLength DESC
LIMIT 1;
================================= Tool Message =================================
Name: sql_db_query

[('Sci Fi & Fantasy', 2911783.0384615385)]
================================== Ai Message ==================================

평균 트랙 길이가 가장 긴 장르는 "Sci Fi & Fantasy"이며, 평균 트랙 길이는 약 2,911,783밀리초입니다.
```

!!! tip
    위 실행에 대한 [LangSmith 추적](https://smith.langchain.com/public/94b8c9ac-12f7-4692-8706-836a1f30f1ea/r)을 참조하세요.

## 다음 단계

LangSmith를 사용하여 이와 같은 SQL 에이전트를 포함한 LangGraph 애플리케이션을 평가하려면 [이 가이드](https://docs.smith.langchain.com/evaluation/how_to_guides/langgraph)를 확인하세요. 