# LangGraph 문서

문서 기여에 대한 자세한 내용은 [기여 가이드](../CONTRIBUTING.md)를 참조하세요.

## 구조

주요 문서는 `docs/` 디렉토리에 있습니다. 이 디렉토리에는 메인 문서의 소스 파일과 API 레퍼런스 문서 빌드 프로세스가 모두 포함되어 있습니다.

### 메인 문서

메인 문서 파일은 `docs/docs/`에 있으며 Markdown 형식으로 작성됩니다. 사이트는 [Material 테마](https://squidfunk.github.io/mkdocs-material/)가 적용된 [**MkDocs**](https://www.mkdocs.org/)를 사용하며 다음을 포함합니다:

- **개념**: 핵심 LangGraph 개념 및 설명
- **튜토리얼**: 단계별 학습 가이드
- **How-tos**: 특정 사용 사례를 위한 작업 중심 가이드
- **예제**: 실제 애플리케이션 및 사용 사례
- **Jupyter 노트북**: 마크다운으로 자동 변환되는 대화형 튜토리얼

### API 레퍼런스

API 레퍼런스 문서는 `docs/docs/reference/`에 정의되어 있습니다. 각 `.md` 파일은 각 페이지가 구축되는 "템플릿"을 설명합니다. 레퍼런스 콘텐츠는 **mkdocstrings** 플러그인을 사용하여 코드베이스의 docstring에서 자동으로 생성됩니다. 생성되면 콘텐츠는 수동 지시문을 사용하여 문서화할 클래스 및/또는 함수를 지정하여 참조되는 해당 마크다운 파일에 연결됩니다:

```markdown
::: langgraph.graph.state.StateGraph
    options:
      show_if_no_docstring: true
      show_root_heading: true
      show_root_full_path: false
      members:
        - add_node
        - add_edge
        - add_conditional_edges
        - add_sequence
        - compile
```

## 빌드 프로세스

문서는 다음 단계에 따라 빌드됩니다:

1. **콘텐츠 처리:**
   - `_scripts/notebook_hooks.py` - 다음을 수행하는 메인 처리 파이프라인:
     - `notebook_convert.py`를 사용하여 how-tos/tutorial Jupyter 노트북을 마크다운으로 변환
     - `generate_api_reference_links.py`를 사용하여 코드 블록에 자동 API 레퍼런스 링크 추가
     - Python/JS 버전에 대한 조건부 렌더링 처리
     - 하이라이트 주석 및 사용자 정의 구문 처리

2. **API 레퍼런스 생성:**
   - **mkdocstrings** 플러그인이 Python 소스 코드에서 docstring을 추출
   - 레퍼런스 페이지(`/docs/docs/*`)의 수동 `::: module.Class` 지시문이 문서화할 내용을 지정
   - 문서와 API 간의 상호 참조가 자동으로 생성됨

3. **사이트 생성:**
   - **MkDocs**가 모든 마크다운 파일을 처리하고 정적 HTML을 생성
   - 사용자 정의 훅이 리디렉션을 처리하고 추가 기능을 주입

4. **배포:**
   - 사이트는 Vercel로 배포됨
   - `make build-docs`는 프로덕션 빌드를 생성함(로컬 테스트에도 사용 가능)
   - 자동 리디렉션이 버전 간 URL 변경을 처리

### 로컬 개발

로컬 개발의 경우 Makefile 타겟을 사용하세요:

```bash
# 핫 리로딩으로 문서를 로컬로 제공
make serve-docs

# 프로덕션 테스트를 위한 클린 빌드
make build-docs

# 클린 빌드로 제공
make serve-clean-docs
```

`serve-docs` 명령:

- 소스 파일의 변경 사항을 감시
- 더 빠른 반복을 위한 더티 빌드 포함
- [http://127.0.0.1:8000/langgraph/](http://127.0.0.1:8000/langgraph/)에서 제공

## 표준

**Docstring 형식:**
API 레퍼런스는 Markdown 마크업을 사용하는 **Google 스타일 docstring**을 사용합니다. `mkdocstrings` 플러그인이 이를 처리하여 문서를 생성합니다.

**필수 형식:**

```python
def example_function(param1: str, param2: int = 5) -> bool:
    """함수에 대한 간략한 설명.

    더 긴 설명은 여기에 들어갈 수 있습니다. **굵게** 및 *기울임꼴*과 같은
    풍부한 서식을 위해 Markdown 구문을 사용하세요.

    Args:
        param1: 첫 번째 매개변수에 대한 설명.
        param2: 기본값이 있는 두 번째 매개변수에 대한 설명.

    Returns:
        반환 값에 대한 설명.

    Raises:
        ValueError: param1이 비어 있을 때.
        TypeError: param2가 정수가 아닐 때.

    !!! warning
        이 함수는 실험적이며 변경될 수 있습니다.

    !!! version-added "버전 0.2.0에 추가됨"
    """
```

**특수 마커:**

- **MkDocs 주의사항**: `!!! warning`, `!!! note`, `!!! version-added`
- **코드 블록**: 표준 마크다운 ``` 구문
- **상호 참조**: `generate_api_reference_links.py`를 통한 자동 링크

## 노트북 실행

모든 노트북을 자동으로 실행하여 "Run notebooks" GitHub 액션을 모방하려면 다음을 실행할 수 있습니다:

```bash
python _scripts/prepare_notebooks_for_ci.py
./_scripts/execute_notebooks.sh
```

**참고**: `%pip install` 셀 없이 노트북을 실행하려면 다음을 실행할 수 있습니다:

```bash
python _scripts/prepare_notebooks_for_ci.py --comment-install-cells
./_scripts/execute_notebooks.sh
```

`prepare_notebooks_for_ci.py` 스크립트는 노트북의 각 셀에 대해 VCR 카세트 컨텍스트 매니저를 추가하여 다음을 수행합니다:

- 노트북이 처음 실행될 때 네트워크 요청이 있는 셀이 VCR 카세트 파일에 기록됨
- 노트북이 이후에 실행될 때 네트워크 요청이 있는 셀이 카세트에서 재생됨

## 새 노트북 추가

API 요청이 있는 노트북을 추가하는 경우, 이후에 재생할 수 있도록 네트워크 요청을 기록하는 것이 **권장**됩니다. 이렇게 하지 않으면 노트북 러너가 노트북을 실행할 때마다 API 요청을 하게 되어 비용이 많이 들고 느릴 수 있습니다.

네트워크 요청을 기록하려면 먼저 `prepare_notebooks_for_ci.py` 스크립트를 실행하세요.

그런 다음 다음을 실행하세요

```bash
jupyter execute <path_to_notebook>
```

노트북이 실행되면 `cassettes` 디렉토리에 새로운 VCR 카세트가 기록된 것을 확인할 수 있으며 업데이트된 노트북은 폐기하세요.

## 기존 노트북 업데이트

기존 노트북을 업데이트하는 경우, `cassettes` 디렉토리에서 노트북에 대한 기존 카세트를 제거한 다음(각 카세트는 노트북 이름으로 접두사가 붙음), 위의 "새 노트북 추가" 섹션의 단계를 실행하세요.

노트북에 대한 카세트를 삭제하려면 다음을 실행할 수 있습니다:

```bash
rm cassettes/<notebook_name>*
```
