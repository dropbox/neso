"""Data models shared by shader lowering passes."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Op


class UnsupportedOperationError(RuntimeError):
    """Raised when an IR operation cannot be lowered without changing semantics."""

    def __init__(self, op: Op, detail: str | None = None):
        message = f"unsupported Triton operation: {op.opname}"
        if detail:
            message += f" ({detail})"
        if op.raw_text:
            message += f"\n  {op.raw_text.strip()}"
        super().__init__(message)
        self.op = op


@dataclass
class TileInfo:
    """A tensor value stored in threadgroup memory or a per-thread register."""

    shared_name: str
    shape: list[int]
    dtype: str
    broadcast_src: TileInfo | None = None
    transposed_from: TileInfo | None = None
    pending_scale: str | None = None
    binop_view: tuple[TileInfo, TileInfo, str] | None = None
    is_register: bool = False

    @property
    def total(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def cols(self) -> int:
        return self.shape[-1] if self.shape else 1

    @property
    def rows(self) -> int:
        return self.shape[0] if len(self.shape) >= 2 else 1

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass
class RegAccInfo:
    """A matrix accumulator kept in simdgroup registers."""

    reg_name: str
    shape: list[int]
    dtype: str
    blocks_per_sg: int
    num_blocks_n: int
    num_blocks_total: int
    sg_var: str
