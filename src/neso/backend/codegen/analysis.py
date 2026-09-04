"""Target-independent analyses over the lightweight Triton IR model."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .ir import Op


def iter_ops(ops: Sequence[Op]) -> Iterator[Op]:
    """Yield operations recursively in source order."""
    for op in ops:
        yield op
        if op.body_ops:
            yield from iter_ops(op.body_ops)


def flatten_ops(ops: Sequence[Op]) -> list[Op]:
    return list(iter_ops(ops))


def build_op_map(ops: Sequence[Op]) -> dict[str, Op]:
    """Map every SSA result name to its producing operation."""
    return {result: op for op in iter_ops(ops) for result in op.results}


def compute_liveness(ops: Sequence[Op]) -> dict[str, int]:
    """Compute the final use index of each SSA value, including loop bodies."""
    last_use: dict[str, int] = {}
    counter = [0]

    def walk(scope_ops: Sequence[Op], outer_defs: set[str]) -> None:
        scope_defs = set(outer_defs)
        for op in scope_ops:
            scope_defs.update(op.results)
            scope_defs.update(op.iter_arg_names)

        for op in scope_ops:
            index = counter[0]
            counter[0] += 1
            for operand in (*op.operands, *op.iter_arg_inits):
                last_use[operand] = max(last_use.get(operand, 0), index)

            if op.body_ops:
                body_defs = set(op.iter_arg_names)
                if op.loop_var:
                    body_defs.add(op.loop_var)
                walk(op.body_ops, scope_defs | body_defs)

                all_body_defs = set(body_defs)
                for body_op in iter_ops(op.body_ops):
                    all_body_defs.update(body_op.results)
                post_body_index = counter[0]
                for body_op in iter_ops(op.body_ops):
                    for operand in body_op.operands:
                        if operand not in all_body_defs:
                            last_use[operand] = max(last_use.get(operand, 0), post_body_index)

            yield_index = counter[0]
            for operand in op.yield_operands:
                last_use[operand] = max(last_use.get(operand, 0), yield_index)

    walk(ops, set())
    return last_use
