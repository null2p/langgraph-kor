# MCP

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction)은 애플리케이션이 언어 모델에 도구와 컨텍스트를 제공하는 방법을 표준화하는 개방형 프로토콜입니다. LangGraph 에이전트는 `langchain-mcp-adapters` 라이브러리를 통해 MCP 서버에 정의된 도구를 사용할 수 있습니다.

![MCP](../agents/assets/mcp.png)

LangGraph에서 MCP 도구를 사용하려면 `langchain-mcp-adapters` 라이브러리를 설치하세요:

:::python
```bash
pip install langchain-mcp-adapters
```
:::

:::js
```bash
npm install @langchain/mcp-adapters
```
:::
