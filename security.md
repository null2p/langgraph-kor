# 보안 정책

## OSS 취약점 신고

LangChain은 오픈 소스 프로젝트를 위한 보상 프로그램을 제공하기 위해 [huntr by Protect AI](https://huntr.com/)와 파트너십을 맺고 있습니다.

다음 링크를 방문하여 LangChain 오픈 소스 프로젝트와 관련된 보안 취약점을 신고해주세요:

[https://huntr.com/bounties/disclose/](https://huntr.com/bounties/disclose/?target=https%3A%2F%2Fgithub.com%2Flangchain-ai%2Flangchain&validSearch=true)

취약점을 신고하기 전에 다음을 검토하세요:

1) 아래의 범위 내 대상 및 범위 외 대상.
2) [langchain-ai/langchain](https://python.langchain.com/docs/contributing/repo_structure) 모노레포 구조.
3) 보안 취약점과 개발자 책임으로 간주하는 것을 이해하기 위한 LangChain [보안 가이드라인](https://python.langchain.com/docs/security).

### 범위 내 대상

다음 패키지와 저장소는 버그 바운티 대상입니다:

- langchain-core
- langchain (예외 사항 참조)
- langchain-community (예외 사항 참조)
- langgraph
- langserve

### 범위 외 대상

huntr에서 정의한 모든 범위 외 대상 및 다음:

- **langchain-experimental**: 이 저장소는 실험적 코드를 위한 것이며 버그 바운티 대상이 아닙니다. 이에 대한 버그 보고서는 흥미롭거나 시간 낭비로 표시되며 바운티 없이 게시됩니다.
- **tools**: langchain 또는 langchain-community의 도구는 버그 바운티 대상이 아닙니다. 여기에는 다음 디렉토리가 포함됩니다
  - langchain/tools
  - langchain-community/tools
  - 자세한 내용은 [보안 가이드라인](https://python.langchain.com/docs/security)을 검토하세요. 일반적으로 도구는 실제 세계와 상호작용합니다. 개발자는 코드의 보안 영향을 이해하고 도구의 보안에 책임을 져야 합니다.
- 보안 공지가 문서화된 코드. 이는 사례별로 결정되지만, 코드가 이미 애플리케이션을 안전하게 만들기 위해 따라야 할 개발자 가이드라인으로 문서화되어 있으므로 바운티 대상이 아닐 가능성이 높습니다.
- 아래를 참조하는 모든 LangSmith 관련 저장소 또는 API.

## LangSmith 취약점 신고

LangSmith와 관련된 보안 취약점은 `security@langchain.dev`로 이메일로 신고해주세요.

- LangSmith 사이트: https://smith.langchain.com
- SDK 클라이언트: https://github.com/langchain-ai/langsmith-sdk

### 기타 보안 문제

기타 보안 문제는 `security@langchain.dev`로 문의해주세요.
