"""
TTIR intermediate representation: data structures and type parser.

This module is backend-agnostic — it defines Op/FuncArg/TType objects that any
backend code generator can consume. The MLIR walker (mlir_walker.py) populates
these structures from the typed MLIR Python bindings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TType:
    """Represents a parsed MLIR type."""
    dtype: str  # base scalar type: f32, i32, etc.
    shape: list[int] | None = None  # None for scalar, [N] for 1D, [M,N] for 2D
    is_ptr: bool = False  # tt.ptr<...>
    encoding: str | None = None  # TTGIR layout encoding (e.g. '#blocked', '#ttg.dot_op<...>')

    @property
    def is_tensor(self) -> bool:
        return self.shape is not None

    @property
    def is_scalar(self) -> bool:
        return self.shape is None and not self.is_ptr

    @property
    def numel(self) -> int:
        if self.shape is None:
            return 1
        r = 1
        for s in self.shape:
            r *= s
        return r

    @property
    def rank(self) -> int:
        if self.shape is None:
            return 0
        return len(self.shape)


@dataclass
class SSAValue:
    name: str
    ttype: TType
    expr: str = ""  # the target-language expression that computes this value


@dataclass
class FuncArg:
    name: str  # %arg0, %arg1, ...
    ttype: TType
    index: int


@dataclass
class Op:
    results: list[str]  # SSA result names
    opname: str  # e.g. 'tt.get_program_id', 'arith.addi'
    operands: list[str]  # SSA operand names
    attrs: dict[str, str]  # parsed attributes
    type_str: str  # raw type string from the MLIR text
    result_types: list[TType]  # parsed result types
    raw_text: str = ""  # full text after opname, before type annotation
    body_ops: list[Op] | None = None  # nested ops (for scf.for, scf.if)
    else_ops: list[Op] | None = None  # else block (for scf.if)
    # scf.for specific fields
    loop_var: str = ""
    loop_start: str = ""
    loop_end: str = ""
    loop_step: str = ""
    iter_arg_names: list[str] = field(default_factory=list)
    iter_arg_inits: list[str] = field(default_factory=list)
    yield_operands: list[str] = field(default_factory=list)
    else_yield_operands: list[str] = field(default_factory=list)


def parse_type(s: str) -> TType:
    """Parse an MLIR type string like 'f32', 'tensor<256xf32>', '!tt.ptr<f32>'.

    Also handles TTGIR encoding attributes:
      'tensor<32x16xf16, #blocked>'
      'tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>'
    """
    s = s.strip()
    m = re.match(r"!tt\.ptr<(.+?)(?:,\s*\d+)?>", s)
    if m:
        inner = parse_type(m.group(1))
        return TType(dtype=inner.dtype, is_ptr=True)
    # Handle TTGIR memdesc types: !ttg.memdesc<32x32xf32, #shared, #smem>
    m = re.match(r"!ttg\.memdesc<(.+)>", s)
    if m:
        # Parse like tensor<...> — extract shape and dtype, ignore encoding attrs
        inner = m.group(1)
        parts = []
        depth = 0
        last = 0
        for i, c in enumerate(inner):
            if c in '<({':
                depth += 1
            elif c in '>)}':
                depth -= 1
            elif c == 'x' and depth == 0:
                parts.append(inner[last:i])
                last = i + 1
        parts.append(inner[last:])
        elem_type_str = parts[-1]
        comma_idx = _find_encoding_comma(elem_type_str)
        if comma_idx >= 0:
            elem_type_str = elem_type_str[:comma_idx].strip()
        shape = [int(p) for p in parts[:-1]]
        elem_type = parse_type(elem_type_str)
        return TType(dtype=elem_type.dtype, shape=shape, is_ptr=elem_type.is_ptr,
                     encoding='#memdesc')
    m = re.match(r"tensor<(.+)>", s)
    if m:
        inner = m.group(1)
        # Split on 'x' at depth 0, tracking angle brackets and braces
        parts = []
        depth = 0
        last = 0
        for i, c in enumerate(inner):
            if c in '<({':
                depth += 1
            elif c in '>)}':
                depth -= 1
            elif c == 'x' and depth == 0:
                parts.append(inner[last:i])
                last = i + 1
        parts.append(inner[last:])
        # The last part is the element type, possibly followed by ', #encoding'
        elem_type_str = parts[-1]
        encoding = None
        # Strip TTGIR encoding attribute (starts with '#' after a comma)
        # e.g. "f16, #blocked" or "f32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>"
        comma_idx = _find_encoding_comma(elem_type_str)
        if comma_idx >= 0:
            encoding = elem_type_str[comma_idx + 1:].strip()
            elem_type_str = elem_type_str[:comma_idx].strip()
        shape = [int(p) for p in parts[:-1]]
        elem_type = parse_type(elem_type_str)
        return TType(dtype=elem_type.dtype, shape=shape, is_ptr=elem_type.is_ptr,
                     encoding=encoding)
    return TType(dtype=s)


def _find_encoding_comma(s: str) -> int:
    """Find the comma separating element type from TTGIR encoding in a tensor type.

    Returns the index of the comma, or -1 if not found.
    The comma must be at depth 0 and followed by a '#'.
    """
    depth = 0
    for i, c in enumerate(s):
        if c in '<({':
            depth += 1
        elif c in '>)}':
            depth -= 1
        elif c == ',' and depth == 0:
            # Check if what follows (after whitespace) starts with '#'
            rest = s[i + 1:].lstrip()
            if rest.startswith('#'):
                return i
    return -1
