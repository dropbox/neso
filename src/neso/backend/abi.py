"""Language-neutral AOT signature and kernel ABI models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedSignature:
    triton_signature: dict[str, str]
    constants: dict[str, Any]
    attributes: dict[tuple[int, ...], list[list[Any]]]
    source_types: dict[str, str]


@dataclass(frozen=True)
class KernelParameter:
    index: int
    name: str
    type: str
    metal_type: str
    is_pointer: bool

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _parse_literal(value: str) -> int | float | None:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return None


def parse_signature(
    arg_names: Sequence[str],
    signature: str,
    namespace: Mapping[str, Any] | None = None,
) -> ParsedSignature:
    """Parse Triton's AOT signature syntax into compiler-ready structures."""
    parts = [part.strip() for part in signature.split(",")]
    if len(parts) != len(arg_names):
        raise ValueError(
            f"signature has {len(parts)} entries but kernel has {len(arg_names)} "
            f"arguments: {list(arg_names)}"
        )

    constants: dict[str, Any] = {}
    hints: dict[tuple[int, ...], int] = {}
    triton_signature: dict[str, str] = {}
    source_types: dict[str, str] = {}
    namespace = namespace or {}

    for index, (name, entry) in enumerate(zip(arg_names, parts)):
        if ":" in entry:
            entry, hint_text = entry.split(":", 1)
            hint = _parse_literal(hint_text)
            if hint not in (1, 16):
                raise ValueError(f"only specialization hints 1 and 16 are supported, got {hint_text!r}")
            hints[(index,)] = int(hint)
            if hint == 1:
                constants[name] = 1

        literal = _parse_literal(entry)
        if literal is not None:
            constants[name] = literal
            source_types[name] = "constexpr"
        elif entry == "None":
            constants[name] = None
            source_types[name] = "constexpr"
        elif entry in namespace:
            constants[name] = namespace[entry]
            source_types[name] = "constexpr"
        else:
            source_types[name] = entry

        triton_signature[name] = "constexpr" if name in constants else entry

    attributes = {
        path: [["tt.divisibility", 16]]
        for path, hint in hints.items()
        if hint == 16
    }
    return ParsedSignature(triton_signature, constants, attributes, source_types)


def build_parameters(
    parameter_names: Sequence[str],
    source_types: Mapping[str, str],
) -> list[KernelParameter]:
    # Imported lazily to keep the ABI model independent of code generation.
    from .codegen.msl_emitter import msl_type_for_triton

    parameters = []
    for index, name in enumerate(parameter_names):
        triton_type = source_types.get(name, "i32")
        if triton_type == "constexpr":
            continue
        parameters.append(KernelParameter(
            index=index,
            name=name,
            type=triton_type,
            metal_type=msl_type_for_triton(triton_type),
            is_pointer=triton_type.startswith("*"),
        ))
    return parameters
