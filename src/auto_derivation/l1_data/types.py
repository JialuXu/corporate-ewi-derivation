"""L1 type system. Mirrors the columns of 原子指标清单.csv."""
from __future__ import annotations

from enum import StrEnum


class Dtype(StrEnum):
    AMOUNT = "金额"
    COUNT = "计数"
    DAYS = "天数"
    RATIO = "比率"
    CATEGORY = "分类"
    BOOL = "布尔"
    TEXT = "文本"
    DATE = "日期"


# Logical types used by L2 operator signatures. Several Dtype values collapse
# to the same logical type (e.g. AMOUNT/COUNT/DAYS/RATIO are all NUMERIC).
class LogicalType(StrEnum):
    NUMERIC = "numeric"
    INT = "int"
    RATIO = "ratio"
    BOOL = "bool"
    EVENT = "event"
    CATEGORY = "category"
    DATE = "date"


DTYPE_TO_LOGICAL: dict[Dtype, LogicalType] = {
    Dtype.AMOUNT: LogicalType.NUMERIC,
    Dtype.COUNT: LogicalType.INT,
    Dtype.DAYS: LogicalType.NUMERIC,
    Dtype.RATIO: LogicalType.RATIO,
    Dtype.CATEGORY: LogicalType.CATEGORY,
    Dtype.BOOL: LogicalType.BOOL,
    Dtype.TEXT: LogicalType.CATEGORY,
    Dtype.DATE: LogicalType.DATE,
}


class TimeGrain(StrEnum):
    DAY = "日"
    QUARTER = "季"
    YEAR = "年"
    EVENT = "事件型"


class NullSemantics(StrEnum):
    DISTINGUISH = "未维护、未授权查询和不适用需区分"
    DISTINGUISH_ZERO = "无记录、0值和缺失需区分"
    UNAVAILABLE_OR_NA = "未维护或不适用"
    PRIMARY_KEY = "主键缺失表示数据不可用"
    OTHER = "其他"

    @classmethod
    def parse(cls, raw: str) -> NullSemantics:
        raw = (raw or "").strip()
        for m in cls:
            if m.value == raw:
                return m
        return cls.OTHER
