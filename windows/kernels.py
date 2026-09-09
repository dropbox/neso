"""Small Triton kernels used by the Windows correctness and performance harness."""

import triton
import triton.language as tl


@triton.jit
def vector_add(x, y, output, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(
        output + offsets,
        tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask),
        mask=mask,
    )


@triton.jit
def scale(x, output, factor, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output + offsets, tl.load(x + offsets, mask=mask) * factor, mask=mask)
