from dataclasses import dataclass


@dataclass(frozen=True)
class TPInfo:
    rank: int
    size: int


TP1 = TPInfo(rank=0, size=1)
