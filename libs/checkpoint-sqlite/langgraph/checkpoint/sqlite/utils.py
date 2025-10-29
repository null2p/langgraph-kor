from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import get_checkpoint_id


def _metadata_predicate(
    metadata_filter: dict[str, Any],
) -> tuple[Sequence[str], Sequence[Any]]:
    """메타데이터 필터가 주어졌을 때 (a)search()에 대한 WHERE 절 술어를 반환합니다.

    이 메서드는 문자열과 값의 튜플로 이루어진 튜플을 반환합니다. 문자열은
    매개변수화된 WHERE 절 술어입니다(WHERE 키워드 제외):
    "column1 = ? AND column2 IS ?". 값의 튜플에는 해당 매개변수 각각에 대한
    값이 포함됩니다.
    """

    def _where_value(query_value: Any) -> tuple[str, Any]:
        """WHERE 절 술어에 대한 연산자와 값의 튜플을 반환합니다."""
        if query_value is None:
            return ("IS ?", None)
        elif (
            isinstance(query_value, str)
            or isinstance(query_value, int)
            or isinstance(query_value, float)
        ):
            return ("= ?", query_value)
        elif isinstance(query_value, bool):
            return ("= ?", 1 if query_value else 0)
        elif isinstance(query_value, dict) or isinstance(query_value, list):
            # JSON 객체의 쿼리 값은 구분 기호(, :) 뒤에 공백이 없어야 함
            # SQLite json_extract()는 공백 없이 JSON 문자열을 반환함
            return ("= ?", json.dumps(query_value, separators=(",", ":")))
        else:
            return ("= ?", str(query_value))

    predicates = []
    param_values = []

    # 메타데이터 쿼리 처리
    for query_key, query_value in metadata_filter.items():
        operator, param_value = _where_value(query_value)
        predicates.append(
            f"json_extract(CAST(metadata AS TEXT), '$.{query_key}') {operator}"
        )
        param_values.append(param_value)

    return (predicates, param_values)


def search_where(
    config: RunnableConfig | None,
    filter: dict[str, Any] | None,
    before: RunnableConfig | None = None,
) -> tuple[str, Sequence[Any]]:
    """메타데이터 필터와 `before` config가 주어졌을 때 (a)search()에 대한
    WHERE 절 술어를 반환합니다.

    이 메서드는 문자열과 값의 튜플로 이루어진 튜플을 반환합니다. 문자열은
    매개변수화된 WHERE 절 술어입니다(WHERE 키워드 포함):
    "WHERE column1 = ? AND column2 IS ?". 값의 튜플에는 해당 매개변수 각각에 대한
    값이 포함됩니다.
    """
    wheres = []
    param_values = []

    # config 필터에 대한 술어 구성
    if config is not None:
        wheres.append("thread_id = ?")
        param_values.append(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns")
        if checkpoint_ns is not None:
            wheres.append("checkpoint_ns = ?")
            param_values.append(checkpoint_ns)

        if checkpoint_id := get_checkpoint_id(config):
            wheres.append("checkpoint_id = ?")
            param_values.append(checkpoint_id)

    # 메타데이터 필터에 대한 술어 구성
    if filter:
        metadata_predicates, metadata_values = _metadata_predicate(filter)
        wheres.extend(metadata_predicates)
        param_values.extend(metadata_values)

    # `before`에 대한 술어 구성
    if before is not None:
        wheres.append("checkpoint_id < ?")
        param_values.append(get_checkpoint_id(before))

    return ("WHERE " + " AND ".join(wheres) if wheres else "", param_values)
