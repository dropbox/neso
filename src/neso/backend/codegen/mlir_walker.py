"""
MLIR walker: extract IR structures from a triton._C.libtriton.ir.module
using the typed MLIR Python bindings.

Replaces the regex-based MLIRParser for operation extraction. Uses the typed
API for operation names, operands, results, attributes, and type information.

Float constant values and dense tensor constants are extracted from the module's
text representation (the Python bindings don't expose get_float_attr).
"""
from __future__ import annotations

import os
import re

from .ir import FuncArg, Op, TType, parse_type

# MLIR arith.cmpi predicate integer -> string name
_CMPI_PREDICATES = {
    0: 'eq', 1: 'ne',
    2: 'slt', 3: 'sle', 4: 'sgt', 5: 'sge',
    6: 'ult', 7: 'ule', 8: 'ugt', 9: 'uge',
}

# MLIR arith.cmpf predicate integer -> string name
_CMPF_PREDICATES = {
    0: 'false', 1: 'oeq', 2: 'ogt', 3: 'oge', 4: 'olt', 5: 'ole',
    6: 'one', 7: 'ord', 8: 'ueq', 9: 'ugt', 10: 'uge', 11: 'ult',
    12: 'ule', 13: 'une', 14: 'uno', 15: 'true',
}

# MLIR tt.atomic_rmw kind integer -> string name
_ATOMIC_RMW_KINDS = {
    1: 'and', 2: 'or', 3: 'xor', 4: 'add', 5: 'fadd',
    6: 'max', 7: 'min', 8: 'umax', 9: 'umin', 10: 'xchg',
}

# tt.get_program_id / tt.get_num_programs axis integer -> string
_AXIS_NAMES = {0: 'x', 1: 'y', 2: 'z'}


def _parse_float_constants_ordered(mlir_text: str) -> list:
    """Extract float constant values from MLIR text in order.

    Returns a list of value strings for arith.constant ops with float types,
    in text order (which matches walk order). The Python bindings don't expose
    get_float_attr, so we must extract these from text.
    """
    values = []
    for line in mlir_text.split('\n'):
        line = line.strip()
        m = re.match(r'%\S+\s*=\s*arith\.constant\s+(.+?)\s*:\s*(f16|f32|f64|bf16)', line)
        if m:
            values.append(m.group(1).strip())
    return values


def _parse_dense_constants_ordered(mlir_text: str) -> list:
    """Extract dense<...> constant values from MLIR text in order.

    Returns a list of raw dense strings (e.g. 'dense<0xFF800000>'), in text
    order (which matches walk order).
    """
    values = []
    for line in mlir_text.split('\n'):
        line = line.strip()
        m = re.match(r'%\S+\s*=\s*arith\.constant\s+(dense<[^>]+>)', line)
        if m:
            values.append(m.group(1))
    return values


class _WalkState:
    """Accumulates data during a module walk."""

    def __init__(self):
        self.ops_by_block = {}     # block_id -> [(op_data, ...)]
        self.value_names = {}      # value_id -> SSA name string
        self.name_counter = 0
        self.func_block_id = None  # block ID of the function body
        self.region_to_blocks = {} # region_id -> [block_id, ...]

    def fresh_name(self) -> str:
        name = f"%v{self.name_counter}"
        self.name_counter += 1
        return name

    def register_value(self, value_id: int, name: str | None = None) -> str:
        """Register a value ID with an SSA name. Auto-generates if name is None."""
        if value_id in self.value_names:
            return self.value_names[value_id]
        if name is None:
            name = self.fresh_name()
        self.value_names[value_id] = name
        return name

    def get_name(self, value_id: int) -> str:
        """Look up the SSA name for a value ID."""
        return self.value_names.get(value_id, f"%unknown_{value_id}")


def walk_module(mod):
    """Extract function name, args, and ops from an MLIR module.

    Uses the typed MLIR Python bindings for all operation metadata extraction.
    Float constants and dense constants are extracted from the text representation
    as a fallback (the bindings don't expose get_float_attr).

    Args:
        mod: A triton._C.libtriton.ir.module (parsed and optimized).

    Returns:
        (func_name, func_args, ops) — same types as MLIRParser.parse_function().
    """
    # Get function name
    func_name = mod.get_entry_func_name()
    if not func_name:
        return None, [], []

    # Get function and extract typed argument info
    func = mod.get_function(func_name)
    n_args = func.get_num_args()

    state = _WalkState()

    # Register function arguments
    func_args = []
    for i in range(n_args):
        arg = func.args(i)
        arg_name = f"%arg{i}"
        state.register_value(arg.id(), arg_name)
        ttype = parse_type(str(arg.get_type()))
        func_args.append(FuncArg(name=arg_name, ttype=ttype, index=i))

    # Extract float/dense constants from text (bindings don't expose float attrs)
    # These are ordered lists matching walk order (post-order = text order for constants)
    mlir_text = mod.str_nodebug()
    float_const_values = _parse_float_constants_ordered(mlir_text)
    dense_const_values = _parse_dense_constants_ordered(mlir_text)

    # Walk all ops, collecting by parent block
    raw_ops = []
    # Track float/dense constant value_ids in walk order for matching
    float_const_vids = []
    dense_const_vids = []

    def walk_cb(op_handle):
        name = op_handle.get_name()

        # Skip structural ops
        if name in ('builtin.module', 'tt.func'):
            return

        block = op_handle.get_block()
        block_id = block.id() if block else 0

        n_results = op_handle.get_num_results()
        n_operands = op_handle.get_num_operands()
        n_regions = op_handle.get_num_regions()

        # Collect result info
        result_ids = []
        result_type_strs = []
        for i in range(n_results):
            r = op_handle.get_result(i)
            result_ids.append(r.id())
            result_type_strs.append(str(r.get_type()))

        # Collect operand IDs
        operand_ids = [op_handle.get_operand(i).id() for i in range(n_operands)]

        # Collect block arguments (for scf.for body blocks)
        block_arg_ids = []
        if block:
            for bi in range(block.get_num_arguments()):
                ba = block.get_argument(bi)
                block_arg_ids.append((ba.id(), str(ba.get_type())))

        # Extract attributes
        int_attrs = {}
        for attr_name in ('value', 'axis', 'start', 'end', 'predicate',
                          'atomic_rmw_op', 'sem', 'scope'):
            v = op_handle.get_int_attr(attr_name)
            if v is not None:
                int_attrs[attr_name] = v

        str_attrs = {}
        for attr_name in ('sym_name', 'symbol', 'libname', 'libpath',
                          'pure', 'reduce_op', 'scan_op'):
            v = op_handle.get_str_attr(attr_name)
            if v is not None:
                str_attrs[attr_name] = v

        # Record region IDs for this op (used to find child blocks)
        region_ids = []
        for ri in range(n_regions):
            region = op_handle.get_region(ri)
            region_ids.append(region.id())

        # Record parent region for this block (used to match blocks to ops)
        parent_region_id = None
        if block:
            parent_region = block.get_parent()
            if parent_region:
                parent_region_id = parent_region.id()

        # Track float/dense constants by value_id for text matching
        if name == 'arith.constant' and result_ids and 'value' not in int_attrs:
            # Not an int constant — either float or dense
            rtype = result_type_strs[0] if result_type_strs else ''
            if 'tensor' in rtype:
                dense_const_vids.append(result_ids[0])
            else:
                float_const_vids.append(result_ids[0])

        raw_ops.append({
            'name': name,
            'block_id': block_id,
            'block_arg_ids': block_arg_ids,
            'result_ids': result_ids,
            'result_type_strs': result_type_strs,
            'operand_ids': operand_ids,
            'n_regions': n_regions,
            'int_attrs': int_attrs,
            'str_attrs': str_attrs,
            'region_ids': region_ids,
            'parent_region_id': parent_region_id,
        })

    mod.walk(walk_cb)

    # Build value_id -> float/dense value dicts by matching walk order to text order
    float_consts = {}  # value_id -> float value string
    for vid, val in zip(float_const_vids, float_const_values):
        float_consts[vid] = val
    dense_consts = {}  # value_id -> dense string
    for vid, val in zip(dense_const_vids, dense_const_values):
        dense_consts[vid] = val

    # Group ops by block
    ops_by_block = {}
    for raw in raw_ops:
        bid = raw['block_id']
        if bid not in ops_by_block:
            ops_by_block[bid] = []
        ops_by_block[bid].append(raw)

    # Build region_id -> block_id mapping from parent_region_id metadata.
    # Each block knows its parent region, so we invert to get region -> blocks.
    region_to_blocks = {}
    for raw in raw_ops:
        bid = raw['block_id']
        prid = raw.get('parent_region_id')
        if prid is not None:
            if prid not in region_to_blocks:
                region_to_blocks[prid] = []
            if bid not in region_to_blocks[prid]:
                region_to_blocks[prid].append(bid)
    state.region_to_blocks = region_to_blocks

    # Find the function body blocks.
    # For functions with cf.cond_br (early return), there are multiple blocks
    # in the function's entry region: ^bb0 (entry), ^bb1 (return), ^bb2 (body).
    # We need ops from ALL blocks in the function region, not just one.
    func_region_id = None
    for raw in raw_ops:
        if raw['name'] == 'tt.return':
            func_region_id = raw.get('parent_region_id')
            state.func_block_id = raw['block_id']
            break

    if state.func_block_id is None:
        return func_name, func_args, []

    # Collect all block IDs in the function region
    func_block_ids = set()
    if func_region_id and func_region_id in region_to_blocks:
        func_block_ids = set(region_to_blocks[func_region_id])
    else:
        func_block_ids = {state.func_block_id}

    # Register all result values with SSA names
    for raw in raw_ops:
        for rid in raw['result_ids']:
            state.register_value(rid)

    # Register block arguments for inner blocks (scf.for iter vars, etc.)
    for raw in raw_ops:
        if raw['block_id'] not in func_block_ids and raw['name'] not in ('tt.return',):
            for ba_id, ba_type in raw['block_arg_ids']:
                state.register_value(ba_id)

    # Build Op objects for the function body — include ALL function-level blocks
    # in walk order (^bb0 first, then ^bb1, ^bb2, etc.)
    func_body_raws = []
    seen_blocks = set()
    for raw in raw_ops:
        bid = raw['block_id']
        if bid in func_block_ids and bid not in seen_blocks:
            seen_blocks.add(bid)
            func_body_raws.extend(ops_by_block[bid])
    ops = _build_ops(func_body_raws, ops_by_block, state, parse_type, Op, TType,
                     float_consts, dense_consts)

    return func_name, func_args, ops


def _build_ops(raw_list, ops_by_block, state, parse_type, Op, TType,
               float_consts, dense_consts):
    """Convert raw op data into Op objects, handling nested regions."""
    ops = []

    for raw in raw_list:
        name = raw['name']

        # Skip structural terminators
        if name in ('tt.return', 'scf.yield', 'tt.reduce.return', 'tt.scan.return'):
            continue

        # Build result names
        results = [state.get_name(rid) for rid in raw['result_ids']]

        # Build operand names
        operands = [state.get_name(oid) for oid in raw['operand_ids']]

        # Build result types
        result_types = [parse_type(ts) for ts in raw['result_type_strs']]

        # Build attributes dict (string values, matching what regex parser produces)
        attrs = {}
        for k, v in raw['int_attrs'].items():
            attrs[k] = str(v)
        for k, v in raw['str_attrs'].items():
            attrs[k] = v

        # Build raw_text for ops that need it
        raw_text = _build_raw_text(raw, state, float_consts, dense_consts)

        # Handle scf.for
        if name == 'scf.for':
            op = _build_scf_for(raw, ops_by_block, state, parse_type, Op, TType,
                                results, operands, result_types, attrs, raw_text,
                                float_consts, dense_consts)
            if op:
                ops.append(op)
            continue

        # Handle scf.if
        if name == 'scf.if':
            op = _build_scf_if(raw, ops_by_block, state, parse_type, Op, TType,
                               results, operands, result_types, attrs, raw_text,
                               float_consts, dense_consts)
            if op:
                ops.append(op)
            continue

        # Handle tt.reduce (has body region for reduce op detection)
        if name == 'tt.reduce':
            op = _build_tt_reduce(raw, ops_by_block, state, parse_type, Op, TType,
                                  results, operands, result_types, attrs, raw_text)
            ops.append(op)
            continue

        # Handle tt.scan
        if name == 'tt.scan':
            op = _build_tt_scan(raw, ops_by_block, state, parse_type, Op, TType,
                                results, operands, result_types, attrs, raw_text)
            ops.append(op)
            continue

        # Handle tt.load: result type should be the loaded element type, not ptr
        if name == 'tt.load' and result_types:
            rt = result_types[0]
            if rt.is_ptr:
                result_types = [TType(dtype=rt.dtype, shape=rt.shape)]

        op = Op(
            results=results,
            opname=name,
            operands=operands,
            attrs=attrs,
            type_str='',
            result_types=result_types,
            raw_text=raw_text,
        )
        ops.append(op)

    return ops


def _build_raw_text(raw, state, float_consts, dense_consts):
    """Build raw_text string for ops that need text-based attribute extraction.

    The lowering reads raw_text for:
    - arith.cmpi/cmpf: predicate string (e.g. "slt, ...")
    - tt.get_program_id/tt.get_num_programs: axis letter (e.g. "x")
    - arith.constant: float/dense values
    - tt.atomic_rmw: atomic op kind (e.g. "fadd, ...")
    - tt.extern_elementwise: already handled via attrs['symbol']
    """
    name = raw['name']

    if name in ('arith.cmpi',):
        pred_int = raw['int_attrs'].get('predicate', 0)
        pred_str = _CMPI_PREDICATES.get(pred_int, 'eq')
        return f"{pred_str}, "

    if name in ('arith.cmpf',):
        pred_int = raw['int_attrs'].get('predicate', 0)
        pred_str = _CMPF_PREDICATES.get(pred_int, 'oeq')
        return f"{pred_str}, "

    if name in ('tt.get_program_id', 'tt.get_num_programs'):
        axis_int = raw['int_attrs'].get('axis', 0)
        return _AXIS_NAMES.get(axis_int, 'x')

    if name == 'arith.constant':
        # For float/dense constants, look up by value_id (matched by walk order)
        vid = raw['result_ids'][0] if raw['result_ids'] else None
        if vid is not None and vid in float_consts:
            # Include type suffix to match text parser format (e.g. "1.000000e+00 : f32")
            # The lowering regex expects the trailing ": type" to extract the value
            rtype = raw['result_type_strs'][0] if raw['result_type_strs'] else 'f32'
            return f"{float_consts[vid]} : {rtype}"
        if vid is not None and vid in dense_consts:
            return dense_consts[vid]
        # Integer constant — value is in int_attrs
        val = raw['int_attrs'].get('value')
        if val is not None:
            return str(val)
        return ''

    if name == 'tt.atomic_rmw':
        # Atomic kind is stored as 'atomic_rmw_op' int attr (NOT 'value')
        kind_int = raw['int_attrs'].get('atomic_rmw_op')
        if kind_int is None:
            return 'fadd, '
        kind_str = _ATOMIC_RMW_KINDS.get(kind_int, 'fadd')
        return f"{kind_str}, "

    return ''


def _build_scf_for(raw, ops_by_block, state, parse_type, Op, TType,
                   results, operands, result_types, attrs, raw_text,
                   float_consts, dense_consts):
    """Build an Op for scf.for with body_ops, iter_args, and yield_operands."""
    # scf.for operands: [lower_bound, upper_bound, step, init_val0, init_val1, ...]
    len(results)
    loop_start = operands[0] if len(operands) > 0 else ''
    loop_end = operands[1] if len(operands) > 1 else ''
    loop_step = operands[2] if len(operands) > 2 else ''
    iter_arg_inits = operands[3:] if len(operands) > 3 else []

    # Find the body block using region IDs.
    # scf.for has exactly 1 region. Look up blocks in that region.
    body_block_id = None
    body_block_args = []
    if raw['region_ids']:
        body_region_id = raw['region_ids'][0]
        body_block_ids = state.region_to_blocks.get(body_region_id, [])
        if body_block_ids:
            body_block_id = body_block_ids[0]
            if ops_by_block.get(body_block_id):
                body_block_args = ops_by_block[body_block_id][0]['block_arg_ids']

    # Register body block args
    loop_var = ''
    iter_arg_names = []
    if body_block_args:
        # First arg is the induction variable
        iv_id, _iv_type = body_block_args[0]
        loop_var = state.register_value(iv_id)
        # Remaining args are iter_args
        for ba_id, ba_type in body_block_args[1:]:
            iter_arg_names.append(state.register_value(ba_id))

    # Build body ops
    body_ops = []
    yield_operands = []
    if body_block_id is not None and body_block_id in ops_by_block:
        body_raws = ops_by_block[body_block_id]
        # Extract yield operands from scf.yield
        for braw in body_raws:
            if braw['name'] == 'scf.yield':
                yield_operands = [state.get_name(oid) for oid in braw['operand_ids']]
        body_ops = _build_ops(body_raws, ops_by_block, state, parse_type, Op, TType,
                              float_consts, dense_consts)
        # Remove this block from available blocks so nested scf.for don't reuse it
        del ops_by_block[body_block_id]

    return Op(
        results=results,
        opname='scf.for',
        operands=[loop_start, loop_end, loop_step] + iter_arg_inits,
        attrs=attrs,
        type_str='',
        result_types=result_types,
        raw_text=raw_text,
        body_ops=body_ops,
        loop_var=loop_var,
        loop_start=loop_start,
        loop_end=loop_end,
        loop_step=loop_step,
        iter_arg_names=iter_arg_names,
        iter_arg_inits=iter_arg_inits,
        yield_operands=yield_operands,
    )


def _build_scf_if(raw, ops_by_block, state, parse_type, Op, TType,
                  results, operands, result_types, attrs, raw_text,
                  float_consts, dense_consts):
    """Build an Op for scf.if with then/else body_ops."""
    # scf.if has 2 regions: then (region 0) and else (region 1).
    # Use region IDs to find the correct blocks.
    then_ops = []
    else_ops = None
    yield_operands = []

    region_ids = raw['region_ids']

    # Region 0 = then
    if len(region_ids) >= 1:
        then_region_id = region_ids[0]
        then_block_ids = state.region_to_blocks.get(then_region_id, [])
        if then_block_ids:
            then_bid = then_block_ids[0]
            if then_bid in ops_by_block:
                then_raws = ops_by_block[then_bid]
                for braw in then_raws:
                    if braw['name'] == 'scf.yield':
                        yield_operands = [state.get_name(oid) for oid in braw['operand_ids']]
                then_ops = _build_ops(then_raws, ops_by_block, state, parse_type, Op, TType,
                                      float_consts, dense_consts)
                del ops_by_block[then_bid]

    # Region 1 = else
    else_yield_operands = []
    if len(region_ids) >= 2:
        else_region_id = region_ids[1]
        else_block_ids = state.region_to_blocks.get(else_region_id, [])
        if else_block_ids:
            else_bid = else_block_ids[0]
            if else_bid in ops_by_block:
                else_raws = ops_by_block[else_bid]
                for braw in else_raws:
                    if braw['name'] == 'scf.yield':
                        else_yield_operands = [state.get_name(oid) for oid in braw['operand_ids']]
                else_ops = _build_ops(else_raws, ops_by_block, state, parse_type, Op, TType,
                                      float_consts, dense_consts)
                del ops_by_block[else_bid]

    return Op(
        results=results,
        opname='scf.if',
        operands=operands,
        attrs=attrs,
        type_str='',
        result_types=result_types,
        raw_text=raw_text,
        body_ops=then_ops,
        else_ops=else_ops,
        yield_operands=yield_operands,
        else_yield_operands=else_yield_operands,
    )


def _build_tt_reduce(raw, ops_by_block, state, parse_type, Op, TType,
                     results, operands, result_types, attrs, raw_text):
    """Build an Op for tt.reduce with reduce_op detection from body."""
    axis = attrs.get('axis', '0')

    # Detect reduce_op from body ops using region IDs
    reduce_op = 'add'
    if raw['region_ids']:
        body_region_id = raw['region_ids'][0]
        body_block_ids = state.region_to_blocks.get(body_region_id, [])
        if body_block_ids:
            bid = body_block_ids[0]
            if bid in ops_by_block:
                block_ops = ops_by_block[bid]
                body_names = [braw['name'] for braw in block_ops]
                body_str = ' '.join(body_names)
                if 'arith.maxnumf' in body_str or 'arith.maximumf' in body_str:
                    reduce_op = 'max'
                elif 'arith.minnumf' in body_str or 'arith.minimumf' in body_str:
                    reduce_op = 'min'
                elif 'arith.addf' in body_str or 'arith.addi' in body_str:
                    reduce_op = 'add'
                elif 'arith.mulf' in body_str or 'arith.muli' in body_str:
                    reduce_op = 'mul'
                # Detect argmax/argmin (multi-result with cmpf + select)
                if 'arith.select' in body_str and 'arith.cmpf' in body_str:
                    for braw in block_ops:
                        if braw['name'] == 'arith.cmpf':
                            pred = braw['int_attrs'].get('predicate', 0)
                            if pred == 4:  # olt
                                reduce_op = 'argmin'
                            elif pred == 2:  # ogt
                                reduce_op = 'argmax'
                del ops_by_block[bid]

    attrs['reduce_op'] = reduce_op
    attrs['axis'] = axis

    return Op(
        results=results,
        opname='tt.reduce',
        operands=operands,
        attrs=attrs,
        type_str='',
        result_types=result_types,
        raw_text=raw_text,
    )


def _build_tt_scan(raw, ops_by_block, state, parse_type, Op, TType,
                   results, operands, result_types, attrs, raw_text):
    """Build an Op for tt.scan with scan_op detection from body."""
    axis = attrs.get('axis', '0')

    scan_op = 'add'
    if raw['region_ids']:
        body_region_id = raw['region_ids'][0]
        body_block_ids = state.region_to_blocks.get(body_region_id, [])
        if body_block_ids:
            bid = body_block_ids[0]
            if bid in ops_by_block:
                block_ops = ops_by_block[bid]
                body_names = [braw['name'] for braw in block_ops]
                body_str = ' '.join(body_names)
                if 'arith.maxnumf' in body_str or 'arith.maximumf' in body_str:
                    scan_op = 'max'
                elif 'arith.minnumf' in body_str or 'arith.minimumf' in body_str:
                    scan_op = 'min'
                elif 'arith.addf' in body_str or 'arith.addi' in body_str:
                    scan_op = 'add'
                elif 'arith.mulf' in body_str or 'arith.muli' in body_str:
                    scan_op = 'mul'
                del ops_by_block[bid]

    attrs['scan_op'] = scan_op
    attrs['axis'] = axis

    return Op(
        results=results,
        opname='tt.scan',
        operands=operands,
        attrs=attrs,
        type_str='',
        result_types=result_types,
        raw_text=raw_text,
    )


def walk_module_from_text(mlir_text: str):
    """Extract IR from MLIR text (for testing without a live module).

    Parses MLIR text into a module using libtriton, then walks it.
    """
    import tempfile

    from triton._C.libtriton import ir

    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        f.write(mlir_text)
        fpath = f.name

    try:
        ctx = ir.context()
        ir.load_dialects(ctx)
        mod = ir.parse_mlir_module(fpath, ctx)
        return walk_module(mod)
    finally:
        os.unlink(fpath)
