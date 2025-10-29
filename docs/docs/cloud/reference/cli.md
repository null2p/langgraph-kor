# LangGraph CLI

LangGraph 명령줄 인터페이스는 [Docker](https://www.docker.com/)에서 로컬로 LangGraph Platform API 서버를 빌드하고 실행하는 명령을 포함합니다. 개발 및 테스트를 위해 CLI를 사용하여 로컬 API 서버를 배포할 수 있습니다.

## 설치

1.  Docker가 설치되어 있는지 확인합니다 (예: `docker --version`).
2.  CLI 패키지를 설치합니다:

    === "Python"
        ```bash
        pip install langgraph-cli
        ```

    === "JS"
        ```bash
        npx @langchain/langgraph-cli

        # Install globally, will be available as `langgraphjs`
        npm install -g @langchain/langgraph-cli
        ```

3.  `langgraph --help` 또는 `npx @langchain/langgraph-cli --help` 명령을 실행하여 CLI가 올바르게 작동하는지 확인합니다.

[](){#langgraph.json}

## 구성 파일 {#configuration-file}

LangGraph CLI는 이 [스키마](https://raw.githubusercontent.com/langchain-ai/langgraph/refs/heads/main/libs/cli/schemas/schema.json)를 따르는 JSON 구성 파일이 필요합니다. 다음 속성을 포함합니다:

<div class="admonition tip">
    <p class="admonition-title">참고</p>
    <p>
        LangGraph CLI는 기본적으로 현재 디렉토리의 구성 파일 <strong>langgraph.json</strong>을 사용합니다.
    </p>
</div>

=== "Python"

    | Key                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`dependencies`</span>     | **Required**. Array of dependencies for LangGraph Platform API server. Dependencies can be one of the following: <ul><li>A single period (`"."`), which will look for local Python packages.</li><li>The directory path where `pyproject.toml`, `setup.py` or `requirements.txt` is located.</br></br>For example, if `requirements.txt` is located in the root of the project directory, specify `"./"`. If it's located in a subdirectory called `local_package`, specify `"./local_package"`. Do not specify the string `"requirements.txt"` itself.</li><li>A Python package name.</li></ul> |
    | <span style="white-space: nowrap;">`graphs`</span>           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./your_package/your_file.py:variable`, where `variable` is an instance of `langgraph.graph.state.CompiledStateGraph`</li><li>`./your_package/your_file.py:make_graph`, where `make_graph` is a function that takes a config dictionary (`langchain_core.runnables.RunnableConfig`) and returns an instance of `langgraph.graph.state.StateGraph` or `langgraph.graph.state.CompiledStateGraph`. See [how to rebuild a graph at runtime](../../cloud/deployment/graph_rebuild.md) for more details.</li></ul>                                    |
    | <span style="white-space: nowrap;">`auth`</span>             | _(Added in v0.0.11)_ Auth configuration containing the path to your authentication handler. Example: `./your_package/auth.py:auth`, where `auth` is an instance of `langgraph_sdk.Auth`. See [authentication guide](../../concepts/auth.md) for details.                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`base_image`</span>       | Optional. Base image to use for the LangGraph API server. Defaults to `langchain/langgraph-api` or `langchain/langgraphjs-api`. Use this to pin your builds to a particular version of the langgraph API, such as `"langchain/langgraph-server:0.2"`. See https://hub.docker.com/r/langchain/langgraph-server/tags for more details. (added in `langgraph-cli==0.2.8`) |
    | <span style="white-space: nowrap;">`image_distro`</span>     | Optional. Linux distribution for the base image. Must be either `"debian"` or `"wolfi"`. If omitted, defaults to `"debian"`. Available in `langgraph-cli>=0.2.11`.|
    | <span style="white-space: nowrap;">`env`</span>              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`store`</span>            | Configuration for adding semantic search and/or time-to-live (TTL) to the BaseStore. Contains the following fields: <ul><li>`index` (optional): Configuration for semantic search indexing with fields `embed`, `dims`, and optional `fields`.</li><li>`ttl` (optional): Configuration for item expiration. An object with optional fields: `refresh_on_read` (boolean, defaults to `true`), `default_ttl` (float, lifespan in **minutes**, defaults to no expiration), and `sweep_interval_minutes` (integer, how often to check for expired items, defaults to no sweeping).</li></ul> |
    | <span style="white-space: nowrap;">`ui`</span>               | Optional. Named definitions of UI components emitted by the agent, each pointing to a JS/TS file. (added in `langgraph-cli==0.1.84`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
    | <span style="white-space: nowrap;">`python_version`</span>   | `3.11`, `3.12`, or `3.13`. Defaults to `3.11`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
    | <span style="white-space: nowrap;">`node_version`</span>     | Specify `node_version: 20` to use LangGraph.js.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
    | <span style="white-space: nowrap;">`pip_config_file`</span>  | Path to `pip` config file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
    | <span style="white-space: nowrap;">`pip_installer`</span> | _(Added in v0.3)_ Optional. Python package installer selector. It can be set to `"auto"`, `"pip"`, or `"uv"`. From version&nbsp;0.3 onward the default strategy is to run `uv pip`, which typically delivers faster builds while remaining a drop-in replacement. In the uncommon situation where `uv` cannot handle your dependency graph or the structure of your `pyproject.toml`, specify `"pip"` here to revert to the earlier behaviour. |
    | <span style="white-space: nowrap;">`keep_pkg_tools`</span> | _(Added in v0.3.4)_ Optional. Control whether to retain Python packaging tools (`pip`, `setuptools`, `wheel`) in the final image. Accepted values: <ul><li><code>true</code> : Keep all three tools (skip uninstall).</li><li><code>false</code> / omitted : Uninstall all three tools (default behaviour).</li><li><code>list[str]</code> : Names of tools <strong>to retain</strong>. Each value must be one of "pip", "setuptools", "wheel".</li></ul>. By default, all three tools are uninstalled. |
    | <span style="white-space: nowrap;">`dockerfile_lines`</span> | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
    | <span style="white-space: nowrap;">`checkpointer`</span>   | Configuration for the checkpointer. Contains a `ttl` field which is an object with the following keys: <ul><li>`strategy`: How to handle expired checkpoints (e.g., `"delete"`).</li><li>`sweep_interval_minutes`: How often to check for expired checkpoints (integer).</li><li>`default_ttl`: Default time-to-live for checkpoints in **minutes** (integer). Defines how long checkpoints are kept before the specified strategy is applied.</li></ul> |
    | <span style="white-space: nowrap;">`http`</span>            | HTTP server configuration with the following fields: <ul><li>`app`: Path to custom Starlette/FastAPI app (e.g., `"./src/agent/webapp.py:app"`). See [custom routes guide](../../how-tos/http/custom_routes.md).</li><li>`cors`: CORS configuration with fields for `allow_origins`, `allow_methods`, `allow_headers`, etc.</li><li>`configurable_headers`: Define which request headers to exclude or include as a run's configurable values.</li><li>`disable_assistants`: Disable `/assistants` routes</li><li>`disable_mcp`: Disable `/mcp` routes</li><li>`disable_meta`: Disable `/ok`, `/info`, `/metrics`, and `/docs` routes</li><li>`disable_runs`: Disable `/runs` routes</li><li>`disable_store`: Disable `/store` routes</li><li>`disable_threads`: Disable `/threads` routes</li><li>`disable_ui`: Disable `/ui` routes</li><li>`disable_webhooks`: Disable webhooks calls on run completion in all routes</li><li>`mount_prefix`: Prefix for mounted routes (e.g., "/my-deployment/api")</li></ul> |

=== "JS"

    | Key                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`graphs`</span>           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./src/graph.ts:variable`, where `variable` is an instance of `CompiledStateGraph`</li><li>`./src/graph.ts:makeGraph`, where `makeGraph` is a function that takes a config dictionary (`LangGraphRunnableConfig`) and returns an instance of `StateGraph` or `CompiledStateGraph`. See [how to rebuild a graph at runtime](../../cloud/deployment/graph_rebuild.md) for more details.</li></ul>                                    |
    | <span style="white-space: nowrap;">`env`</span>              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`store`</span>            | Configuration for adding semantic search and/or time-to-live (TTL) to the BaseStore. Contains the following fields: <ul><li>`index` (optional): Configuration for semantic search indexing with fields `embed`, `dims`, and optional `fields`.</li><li>`ttl` (optional): Configuration for item expiration. An object with optional fields: `refresh_on_read` (boolean, defaults to `true`), `default_ttl` (float, lifespan in **minutes**, defaults to no expiration), and `sweep_interval_minutes` (integer, how often to check for expired items, defaults to no sweeping).</li></ul> |
    | <span style="white-space: nowrap;">`node_version`</span>     | Specify `node_version: 20` to use LangGraph.js.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
    | <span style="white-space: nowrap;">`dockerfile_lines`</span> | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
    | <span style="white-space: nowrap;">`checkpointer`</span>   | Configuration for the checkpointer. Contains a `ttl` field which is an object with the following keys: <ul><li>`strategy`: How to handle expired checkpoints (e.g., `"delete"`).</li><li>`sweep_interval_minutes`: How often to check for expired checkpoints (integer).</li><li>`default_ttl`: Default time-to-live for checkpoints in **minutes** (integer). Defines how long checkpoints are kept before the specified strategy is applied.</li></ul> |

### Examples

=== "Python"
    
    #### Basic Configuration

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "chat": "./chat/graph.py:graph"
      }
    }
    ```

    #### Using Wolfi Base Images

    You can specify the Linux distribution for your base image using the `image_distro` field. Valid options are `debian` or `wolfi`. Wolfi is the recommended option as it provides smaller and more secure images. This is available in `langgraph-cli>=0.2.11`.

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "chat": "./chat/graph.py:graph"
      },
      "image_distro": "wolfi"
    }
    ```

    #### Adding semantic search to the store

    All deployments come with a DB-backed BaseStore. Adding an "index" configuration to your `langgraph.json` will enable [semantic search](../deployment/semantic_search.md) within the BaseStore of your deployment.

    The `index.fields` configuration determines which parts of your documents to embed:

    - If omitted or set to `["$"]`, the entire document will be embedded
    - To embed specific fields, use JSON path notation: `["metadata.title", "content.text"]`
    - Documents missing specified fields will still be stored but won't have embeddings for those fields
    - You can still override which fields to embed on a specific item at `put` time using the `index` parameter

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "memory_agent": "./agent/graph.py:graph"
      },
      "store": {
        "index": {
          "embed": "openai:text-embedding-3-small",
          "dims": 1536,
          "fields": ["$"]
        }
      }
    }
    ```

    !!! note "Common model dimensions" 
        - `openai:text-embedding-3-large`: 3072 
        - `openai:text-embedding-3-small`: 1536 
        - `openai:text-embedding-ada-002`: 1536 
        - `cohere:embed-english-v3.0`: 1024 
        - `cohere:embed-english-light-v3.0`: 384 
        - `cohere:embed-multilingual-v3.0`: 1024 
        - `cohere:embed-multilingual-light-v3.0`: 384 

    #### Semantic search with a custom embedding function

    If you want to use semantic search with a custom embedding function, you can pass a path to a custom embedding function:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "memory_agent": "./agent/graph.py:graph"
      },
      "store": {
        "index": {
          "embed": "./embeddings.py:embed_texts",
          "dims": 768,
          "fields": ["text", "summary"]
        }
      }
    }
    ```

    The `embed` field in store configuration can reference a custom function that takes a list of strings and returns a list of embeddings. Example implementation:

    ```python
    # embeddings.py
    def embed_texts(texts: list[str]) -> list[list[float]]:
        """Custom embedding function for semantic search."""
        # Implementation using your preferred embedding model
        return [[0.1, 0.2, ...] for _ in texts]  # dims-dimensional vectors
    ```

    #### Adding custom authentication

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "chat": "./chat/graph.py:graph"
      },
      "auth": {
        "path": "./auth.py:auth",
        "openapi": {
          "securitySchemes": {
            "apiKeyAuth": {
              "type": "apiKey",
              "in": "header",
              "name": "X-API-Key"
            }
          },
          "security": [{ "apiKeyAuth": [] }]
        },
        "disable_studio_auth": false
      }
    }
    ```

    See the [authentication conceptual guide](../../concepts/auth.md) for details, and the [setting up custom authentication](../../tutorials/auth/getting_started.md) guide for a practical walk through of the process.

    #### Configuring Store Item Time-to-Live (TTL)

    You can configure default data expiration for items/memories in the BaseStore using the `store.ttl` key. This determines how long items are retained after they are last accessed (with reads potentially refreshing the timer based on `refresh_on_read`). Note that these defaults can be overwritten on a per-call basis by modifying the corresponding arguments in `get`, `search`, etc.
    
    The `ttl` configuration is an object containing optional fields:

    - `refresh_on_read`: If `true` (the default), accessing an item via `get` or `search` resets its expiration timer. Set to `false` to only refresh TTL on writes (`put`).
    - `default_ttl`: The default lifespan of an item in **minutes**. If not set, items do not expire by default.
    - `sweep_interval_minutes`: How frequently (in minutes) the system should run a background process to delete expired items. If not set, sweeping does not occur automatically.

    Here is an example enabling a 7-day TTL (10080 minutes), refreshing on reads, and sweeping every hour:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "memory_agent": "./agent/graph.py:graph"
      },
      "store": {
        "ttl": {
          "refresh_on_read": true,
          "sweep_interval_minutes": 60,
          "default_ttl": 10080 
        }
      }
    }
    ```

    #### Configuring Checkpoint Time-to-Live (TTL)

    You can configure the time-to-live (TTL) for checkpoints using the `checkpointer` key. This determines how long checkpoint data is retained before being automatically handled according to the specified strategy (e.g., deletion). The `ttl` configuration is an object containing:

    - `strategy`: The action to take on expired checkpoints (currently `"delete"` is the only accepted option).
    - `sweep_interval_minutes`: How frequently (in minutes) the system checks for expired checkpoints.
    - `default_ttl`: The default lifespan of a checkpoint in **minutes**.

    Here's an example setting a default TTL of 30 days (43200 minutes):

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "chat": "./chat/graph.py:graph"
      },
      "checkpointer": {
        "ttl": {
          "strategy": "delete",
          "sweep_interval_minutes": 10,
          "default_ttl": 43200
        }
      }
    }
    ```

    In this example, checkpoints older than 30 days will be deleted, and the check runs every 10 minutes.


=== "JS"
    
    #### Basic Configuration

    ```json
    {
      "graphs": {
        "chat": "./src/graph.ts:graph"
      }
    }
    ```


## 명령

**사용법**

=== "Python"

    LangGraph CLI의 기본 명령은 `langgraph`입니다.

    ```
    langgraph [OPTIONS] COMMAND [ARGS]
    ```
=== "JS"

    LangGraph.js CLI의 기본 명령은 `langgraphjs`입니다.

    ```
    npx @langchain/langgraph-cli [OPTIONS] COMMAND [ARGS]
    ```

    항상 최신 버전의 CLI를 사용하려면 `npx`를 사용하는 것이 좋습니다.

### `dev`

=== "Python"

    핫 리로딩 및 디버깅 기능을 갖춘 개발 모드에서 LangGraph API 서버를 실행합니다. 이 경량 서버는 Docker 설치가 필요하지 않으며 개발 및 테스트에 적합합니다. 상태는 로컬 디렉토리에 유지됩니다.

    !!! note

        현재 CLI는 Python >= 3.11만 지원합니다.

    **설치**

    이 명령에는 "inmem" 추가 항목을 설치해야 합니다:

    ```bash
    pip install -U "langgraph-cli[inmem]"
    ```

    **사용법**

    ```
    langgraph dev [OPTIONS]
    ```

    **옵션**

    | 옵션                          | 기본값           | 설명                                                                         |
    | ----------------------------- | ---------------- | ----------------------------------------------------------------------------------- |
    | `-c, --config FILE`           | `langgraph.json` | 종속성, 그래프 및 환경 변수를 선언하는 구성 파일 경로 |
    | `--host TEXT`                 | `127.0.0.1`      | 서버를 바인딩할 호스트                                                          |
    | `--port INTEGER`              | `2024`           | 서버를 바인딩할 포트                                                          |
    | `--no-reload`                 |                  | 자동 리로드 비활성화                                                                 |
    | `--n-jobs-per-worker INTEGER` |                  | 워커당 작업 수. 기본값은 10                                            |
    | `--debug-port INTEGER`        |                  | 디버거가 수신 대기할 포트                                                      |
    | `--wait-for-client`           | `False`          | 서버를 시작하기 전에 디버거 클라이언트가 디버그 포트에 연결될 때까지 대기   |
    | `--no-browser`                |                  | 서버 시작 시 브라우저를 자동으로 열지 않음                       |
    | `--studio-url TEXT`           |                  | 연결할 LangGraph Studio 인스턴스의 URL. 기본값은 https://smith.langchain.com |
    | `--allow-blocking`            | `False`          | 코드에서 동기 I/O 차단 작업에 대한 오류를 발생시키지 않음 (`0.2.6`에서 추가)           |
    | `--tunnel`                    | `False`          | 원격 프론트엔드 액세스를 위해 공개 터널(Cloudflare)을 통해 로컬 서버를 노출합니다. 이렇게 하면 Safari와 같은 브라우저나 localhost 연결을 차단하는 네트워크의 문제를 피할 수 있습니다        |
    | `--help`                      |                  | 명령 문서 표시                                                       |


=== "JS"

    핫 리로딩 기능을 갖춘 개발 모드에서 LangGraph API 서버를 실행합니다. 이 경량 서버는 Docker 설치가 필요하지 않으며 개발 및 테스트에 적합합니다. 상태는 로컬 디렉토리에 유지됩니다.

    **사용법**

    ```
    npx @langchain/langgraph-cli dev [OPTIONS]
    ```

    **옵션**

    | 옵션                          | 기본값           | 설명                                                                         |
    | ----------------------------- | ---------------- | ----------------------------------------------------------------------------------- |
    | `-c, --config FILE`           | `langgraph.json` | 종속성, 그래프 및 환경 변수를 선언하는 구성 파일 경로 |
    | `--host TEXT`                 | `127.0.0.1`      | 서버를 바인딩할 호스트                                                          |
    | `--port INTEGER`              | `2024`           | 서버를 바인딩할 포트                                                          |
    | `--no-reload`                 |                  | 자동 리로드 비활성화                                                                 |
    | `--n-jobs-per-worker INTEGER` |                  | 워커당 작업 수. 기본값은 10                                            |
    | `--debug-port INTEGER`        |                  | 디버거가 수신 대기할 포트                                                      |
    | `--wait-for-client`           | `False`          | 서버를 시작하기 전에 디버거 클라이언트가 디버그 포트에 연결될 때까지 대기   |
    | `--no-browser`                |                  | 서버 시작 시 브라우저를 자동으로 열지 않음                       |
    | `--studio-url TEXT`           |                  | 연결할 LangGraph Studio 인스턴스의 URL. 기본값은 https://smith.langchain.com |
    | `--allow-blocking`            | `False`          | 코드에서 동기 I/O 차단 작업에 대한 오류를 발생시키지 않음            |
    | `--tunnel`                    | `False`          | 원격 프론트엔드 액세스를 위해 공개 터널(Cloudflare)을 통해 로컬 서버를 노출합니다. 이렇게 하면 브라우저나 localhost 연결을 차단하는 네트워크의 문제를 피할 수 있습니다        |
    | `--help`                      |                  | 명령 문서 표시                                                       |

### `build`

=== "Python"

    LangGraph Platform API 서버 Docker 이미지를 빌드합니다.

    **사용법**

    ```
    langgraph build [OPTIONS]
    ```

    **옵션**

    | 옵션                 | 기본값           | 설명                                                                                                     |
    | -------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
    | `--platform TEXT`    |                  | Docker 이미지를 빌드할 대상 플랫폼. 예: `langgraph build --platform linux/amd64,linux/arm64`              |
    | `-t, --tag TEXT`     |                  | **필수**. Docker 이미지의 태그. 예: `langgraph build -t my-image`                                               |
    | `--pull / --no-pull` | `--pull`         | 최신 원격 Docker 이미지로 빌드. 로컬에서 빌드한 이미지로 LangGraph Platform API 서버를 실행하려면 `--no-pull`을 사용하세요. |
    | `-c, --config FILE`  | `langgraph.json` | 종속성, 그래프 및 환경 변수를 선언하는 구성 파일 경로.                                         |
    | `--help`             |                  | 명령 문서 표시.                                                                                               |

=== "JS"

    LangGraph Platform API 서버 Docker 이미지를 빌드합니다.

    **사용법**

    ```
    npx @langchain/langgraph-cli build [OPTIONS]
    ```

    **옵션**

    | 옵션                 | 기본값           | 설명                                                                                                     |
    | -------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
    | `--platform TEXT`    |                  | Docker 이미지를 빌드할 대상 플랫폼. 예: `langgraph build --platform linux/amd64,linux/arm64`              |
    | `-t, --tag TEXT`     |                  | **필수**. Docker 이미지의 태그. 예: `langgraph build -t my-image`                                               |
    | `--no-pull`          |                  | 로컬에서 빌드한 이미지를 사용합니다. 기본값은 `false`로 최신 원격 Docker 이미지로 빌드합니다.                                      |
    | `-c, --config FILE`  | `langgraph.json` | 종속성, 그래프 및 환경 변수를 선언하는 구성 파일 경로.                                         |
    | `--help`             |                  | 명령 문서 표시.                                                                                               |


### `up`

=== "Python"

    LangGraph API 서버를 시작합니다. 로컬 테스트의 경우 LangGraph Platform에 액세스할 수 있는 LangSmith API 키가 필요합니다. 프로덕션 사용에는 라이선스 키가 필요합니다.

    **사용법**

    ```
    langgraph up [OPTIONS]
    ```

    **옵션**

    | 옵션                         | 기본값                    | 설명                                                                                                             |
    | ---------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
    | `--wait`                     |                           | 반환하기 전에 서비스가 시작될 때까지 대기합니다. --detach를 의미합니다                                                           |
    | `--base-image TEXT`          | `langchain/langgraph-api`  | LangGraph API 서버에 사용할 기본 이미지. 버전 태그를 사용하여 특정 버전을 고정합니다.                            |
    | `--image TEXT`               |                           | langgraph-api 서비스에 사용할 Docker 이미지. 지정하면 빌드를 건너뛰고 이 이미지를 직접 사용합니다.           |
    | `--postgres-uri TEXT`        | 로컬 데이터베이스            | 데이터베이스에 사용할 Postgres URI.                                                                                   |
    | `--watch`                    |                           | 파일 변경 시 재시작                                                                                                 |
    | `--debugger-base-url TEXT`   | `http://127.0.0.1:[PORT]` | 디버거가 LangGraph API에 액세스하는 데 사용하는 URL.                                                                       |
    | `--debugger-port INTEGER`    |                           | 디버거 이미지를 로컬로 가져와 지정된 포트에서 UI 제공                                                      |
    | `--verbose`                  |                           | 서버 로그에서 더 많은 출력을 표시합니다.                                                                                  |
    | `-c, --config FILE`          | `langgraph.json`          | 종속성, 그래프 및 환경 변수를 선언하는 구성 파일 경로.                                    |
    | `-d, --docker-compose FILE`  |                           | 시작할 추가 서비스가 포함된 docker-compose.yml 파일 경로.                                                     |
    | `-p, --port INTEGER`         | `8123`                    | 노출할 포트. 예: `langgraph up --port 8000`                                                                     |
    | `--pull / --no-pull`         | `pull`                    | 최신 이미지를 가져옵니다. 로컬에서 빌드한 이미지로 서버를 실행하려면 `--no-pull`을 사용하세요. 예: `langgraph up --no-pull` |
    | `--recreate / --no-recreate` | `no-recreate`             | 구성 및 이미지가 변경되지 않았더라도 컨테이너를 다시 생성합니다                                               |
    | `--help`                     |                           | 명령 문서 표시.                                                                                          |

=== "JS"

    LangGraph API 서버를 시작합니다. 로컬 테스트의 경우 LangGraph Platform에 액세스할 수 있는 LangSmith API 키가 필요합니다. 프로덕션 사용의 경우 라이선스 키가 필요합니다.

    **사용법**

    ```
    npx @langchain/langgraph-cli up [OPTIONS]
    ```

    **옵션**

    | 옵션                                                                 | 기본값                   | 설명                                                                                                             |
    | ---------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`--wait`</span>                     |                           | 반환하기 전에 서비스가 시작될 때까지 기다립니다. --detach를 의미합니다                                                           |
    | <span style="white-space: nowrap;">`--base-image TEXT`</span>          | <span style="white-space: nowrap;">`langchain/langgraph-api`</span> | LangGraph API 서버에 사용할 기본 이미지. 버전 태그를 사용하여 특정 버전에 고정합니다. |
    | <span style="white-space: nowrap;">`--image TEXT`</span>               |                           | langgraph-api 서비스에 사용할 Docker 이미지. 지정된 경우 빌드를 건너뛰고 이 이미지를 직접 사용합니다. |
    | <span style="white-space: nowrap;">`--postgres-uri TEXT`</span>        | 로컬 데이터베이스            | 데이터베이스에 사용할 Postgres URI.                                                                                   |
    | <span style="white-space: nowrap;">`--watch`</span>                    |                           | 파일 변경 시 재시작                                                                                                 |
    | <span style="white-space: nowrap;">`-c, --config FILE`</span>          | `langgraph.json`          | 종속성, 그래프 및 환경 변수를 선언하는 구성 파일 경로.                                    |
    | <span style="white-space: nowrap;">`-d, --docker-compose FILE`</span>  |                           | 시작할 추가 서비스가 포함된 docker-compose.yml 파일 경로.                                                     |
    | <span style="white-space: nowrap;">`-p, --port INTEGER`</span>         | `8123`                    | 노출할 포트. 예: `langgraph up --port 8000`                                                                     |
    | <span style="white-space: nowrap;">`--no-pull`</span>                  |                           | 로컬에서 빌드한 이미지를 사용합니다. 최신 원격 Docker 이미지로 빌드하려면 기본값 `false`입니다.                                 |
    | <span style="white-space: nowrap;">`--recreate`</span>                 |                           | 구성 및 이미지가 변경되지 않았더라도 컨테이너를 다시 생성합니다                                               |
    | <span style="white-space: nowrap;">`--help`</span>                     |                           | 명령 문서 표시.                                                                                          |

### `dockerfile`

=== "Python"

    LangGraph Platform API 서버 Docker 이미지를 빌드하기 위한 Dockerfile을 생성합니다.

    **사용법**

    ```
    langgraph dockerfile [OPTIONS] SAVE_PATH
    ```

    **옵션**

    | 옵션              | 기본값          | 설명                                                                                                     |
    | ------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
    | `-c, --config FILE` | `langgraph.json` | 종속성, 그래프 및 환경 변수를 선언하는 [구성 파일](#configuration-file) 경로. |
    | `--help`            |                  | 이 메시지를 표시하고 종료합니다.                                                                                     |

    예제:

    ```bash
    langgraph dockerfile -c langgraph.json Dockerfile
    ```

    다음과 유사한 Dockerfile을 생성합니다:

    ```dockerfile
    FROM langchain/langgraph-api:3.11

    ADD ./pipconf.txt /pipconfig.txt

    RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt langchain_community langchain_anthropic langchain_openai wikipedia scikit-learn

    ADD ./graphs /deps/outer-graphs/src
    RUN set -ex && \
        for line in '[project]' \
                    'name = "graphs"' \
                    'version = "0.1"' \
                    '[tool.setuptools.package-data]' \
                    '"*" = ["**/*"]'; do \
            echo "$line" >> /deps/outer-graphs/pyproject.toml; \
        done

    RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

    ENV LANGSERVE_GRAPHS='{"agent": "/deps/outer-graphs/src/agent.py:graph", "storm": "/deps/outer-graphs/src/storm.py:graph"}'
    ```

    ???+ note "langgraph.json 파일 업데이트"
         `langgraph dockerfile` 명령은 `langgraph.json` 파일의 모든 구성을 Dockerfile 명령으로 변환합니다. 이 명령을 사용할 때 `langgraph.json` 파일을 업데이트할 때마다 다시 실행해야 합니다. 그렇지 않으면 dockerfile을 빌드하거나 실행할 때 변경 사항이 반영되지 않습니다.

=== "JS"

    LangGraph Platform API 서버 Docker 이미지를 빌드하기 위한 Dockerfile을 생성합니다.

    **사용법**

    ```
    npx @langchain/langgraph-cli dockerfile [OPTIONS] SAVE_PATH
    ```

    **옵션**

    | 옵션              | 기본값          | 설명                                                                                                     |
    | ------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
    | `-c, --config FILE` | `langgraph.json` | 종속성, 그래프 및 환경 변수를 선언하는 [구성 파일](#configuration-file) 경로. |
    | `--help`            |                  | 이 메시지를 표시하고 종료합니다.                                                                                     |

    예제:

    ```bash
    npx @langchain/langgraph-cli dockerfile -c langgraph.json Dockerfile
    ```

    다음과 유사한 Dockerfile을 생성합니다:

    ```dockerfile
    FROM langchain/langgraphjs-api:20

    ADD . /deps/agent

    RUN cd /deps/agent && yarn install

    ENV LANGSERVE_GRAPHS='{"agent":"./src/react_agent/graph.ts:graph"}'

    WORKDIR /deps/agent

    RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts
    ```

    ???+ note "langgraph.json 파일 업데이트"
         `npx @langchain/langgraph-cli dockerfile` 명령은 `langgraph.json` 파일의 모든 구성을 Dockerfile 명령으로 변환합니다. 이 명령을 사용할 때 `langgraph.json` 파일을 업데이트할 때마다 다시 실행해야 합니다. 그렇지 않으면 dockerfile을 빌드하거나 실행할 때 변경 사항이 반영되지 않습니다.
