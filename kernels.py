"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl


# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.


@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.
    """
    y_re = tl.dot(a_re, b_re, out_dtype=tl.float32) - tl.dot(a_im, b_im, out_dtype=tl.float32)
    y_im = tl.dot(a_re, b_im, out_dtype=tl.float32) + tl.dot(a_im, b_re, out_dtype=tl.float32)
    return y_re, y_im


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    assert N >= 2 and (N & (N - 1)) == 0, f"N must be a power of 2 >= 2; got {N}"
    k = N.bit_length() - 1
    n256, rb = divmod(k, 8)
    n16, rb2 = divmod(rb, 4)
    rsmall = 1 << rb2
    chunks = [256] * n256 + [16] * n16 + ([rsmall] if rsmall > 1 else [])
    assert math.prod(chunks) == N
    return chunks


f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_idx in range(0, N, BLOCK_K):
        x_re_block_ptr = x_re_ptr + offs_m[:, None] * N + (k_idx + offs_k)[None, :]
        x_im_block_ptr = x_im_ptr + offs_m[:, None] * N + (k_idx + offs_k)[None, :]
        x_mask = (offs_m[:, None] < B) & ((k_idx + offs_k)[None, :] < N)
        
        W_re_block_ptr = W_re_ptr + offs_n[None, :] * N + (k_idx + offs_k)[:, None]
        W_im_block_ptr = W_im_ptr + offs_n[None, :] * N + (k_idx + offs_k)[:, None]
        W_mask = ((k_idx + offs_k)[:, None] < N) & (offs_n[None, :] < N)
        
        x_re_val = tl.load(x_re_block_ptr, mask=x_mask, other=0.0)
        x_im_val = tl.load(x_im_block_ptr, mask=x_mask, other=0.0)
        W_re_val = tl.load(W_re_block_ptr, mask=W_mask, other=0.0)
        W_im_val = tl.load(W_im_block_ptr, mask=W_mask, other=0.0)
        
        dot_re, dot_im = _cdot(x_re_val, x_im_val, W_re_val, W_im_val)
        
        acc_re += dot_re
        acc_im += dot_im
        
    y_re_block_ptr = y_re_ptr + offs_m[:, None] * N + offs_n[None, :]
    y_im_block_ptr = y_im_ptr + offs_m[:, None] * N + offs_n[None, :]
    y_mask = (offs_m[:, None] < B) & (offs_n[None, :] < N)
    
    tl.store(y_re_block_ptr, acc_re, mask=y_mask)
    tl.store(y_im_block_ptr, acc_im, mask=y_mask)


def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.
    """
    B, N = x_re.shape
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    f1_kernel[grid](
        x_re, x_im, W_re, W_im, y_re, y_im, B, N,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N
    )


# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.
    """
    pid = tl.program_id(0)
    
    offs = tl.arange(0, N)
    perm = tl.load(perm_ptr + offs)
    
    v_re = tl.load(x_re_ptr + pid * N + perm)
    v_im = tl.load(x_im_ptr + pid * N + perm)
    
    for s in range(LOG2_N):
        partner_idx = offs ^ (1 << s)
        partner_re = tl.gather(v_re, partner_idx, axis=0)
        partner_im = tl.gather(v_im, partner_idx, axis=0)
        
        tw_idx = (offs & ((1 << s) - 1)) * (N >> (s + 1))
        w_re = tl.load(tw_re_ptr + tw_idx)
        w_im = tl.load(tw_im_ptr + tw_idx)
        
        is_high = (offs & (1 << s)) != 0
        val_to_mul_re = tl.where(is_high, v_re, partner_re)
        val_to_mul_im = tl.where(is_high, v_im, partner_im)
        
        mul_re = val_to_mul_re * w_re - val_to_mul_im * w_im
        mul_im = val_to_mul_re * w_im + val_to_mul_im * w_re
        
        v_re = tl.where(is_high, partner_re - mul_re, v_re + mul_re)
        v_im = tl.where(is_high, partner_im - mul_im, v_im + mul_im)
        
    if BAILEY_EPILOGUE:
        n1 = pid % OUTER_DIM
        bt_re = tl.load(bt_re_ptr + n1 * N + offs)
        bt_im = tl.load(bt_im_ptr + n1 * N + offs)
        
        v_re_new = v_re * bt_re - v_im * bt_im
        v_im_new = v_re * bt_im + v_im * bt_re
        v_re = v_re_new
        v_im = v_im_new
        
    if STRIDED_STORE:
        k2 = pid % OUTER_DIM
        b = pid // OUTER_DIM
        out_offs = b * N_TOTAL + offs * OUTER_DIM + k2
    else:
        out_offs = pid * N + offs
        
    tl.store(y_re_ptr + out_offs, v_re)
    tl.store(y_im_ptr + out_offs, v_im)


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Grid: (B,). One program per length-N signal. Vanilla mode."""
    B, N = x_re.shape
    LOG2_N = int(math.log2(N))
    f2_kernel[(B,)](
        x_re, x_im, y_re, y_im,
        tw_re, tw_im, perm,
        tw_re, tw_im, # sentinels
        1, 0,
        N=N, LOG2_N=LOG2_N,
        BAILEY_EPILOGUE=False, STRIDED_STORE=False
    )


# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)
    in_offset = pid_b * (R * C) + offs_r[:, None] * C + offs_c[None, :]
    
    val_re = tl.load(x_re_ptr + in_offset, mask=mask, other=0.0)
    val_im = tl.load(x_im_ptr + in_offset, mask=mask, other=0.0)
    
    out_offset = pid_b * (C * R) + offs_c[None, :] * R + offs_r[:, None]
    tl.store(y_re_ptr + out_offset, val_re, mask=mask)
    tl.store(y_im_ptr + out_offset, val_im, mask=mask)


# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = tl.arange(0, 256)
    
    # Load input, in the most efficient way
    x_re = tl.load(x_re_ptr + offs_b[:, None] * 256 + offs_n[None, :], mask=offs_b[:, None] < B, other=0.0)
    x_im = tl.load(x_im_ptr + offs_b[:, None] * 256 + offs_n[None, :], mask=offs_b[:, None] < B, other=0.0)
    
    tile_re = tl.reshape(x_re, (BLOCK_B, 16, 16))
    tile_im = tl.reshape(x_im, (BLOCK_B, 16, 16))
    
    #pog
    tile_re_perm = tl.permute(tile_re, (1, 0, 2))
    tile_im_perm = tl.permute(tile_im, (1, 0, 2))
    tile_re_2d = tl.reshape(tile_re_perm, (16, BLOCK_B * 16))
    tile_im_2d = tl.reshape(tile_im_perm, (16, BLOCK_B * 16))
    out_re_2d, out_im_2d = _cdot(F_re_ptr, F_im_ptr, tile_re_2d, tile_im_2d)
    out_re_perm = tl.reshape(out_re_2d, (16, BLOCK_B, 16))
    out_im_perm = tl.reshape(out_im_2d, (16, BLOCK_B, 16))
    tile_re = tl.permute(out_re_perm, (1, 0, 2)).to(tl.float16)
    tile_im = tl.permute(out_im_perm, (1, 0, 2)).to(tl.float16)
    
    # I am become death, the destroyer of 256 point DFTs
    if STAGE_STOP > 1:
        tile_re = tl.permute(tile_re, (0, 2, 1))
        tile_im = tl.permute(tile_im, (0, 2, 1))
        
        offs_row = tl.arange(0, 16)
        offs_col = tl.arange(0, 16)
        w_re = tl.load(tw_re_ptr + 256 + offs_row[:, None] * 16 + offs_col[None, :])
        w_im = tl.load(tw_im_ptr + 256 + offs_row[:, None] * 16 + offs_col[None, :])
        
        tile_re_new = tile_re * w_re[None, :, :] - tile_im * w_im[None, :, :]
        tile_im_new = tile_re * w_im[None, :, :] + tile_im * w_re[None, :, :]
        tile_re = tile_re_new
        tile_im = tile_im_new
        
        tile_re_perm = tl.permute(tile_re, (1, 0, 2))
        tile_im_perm = tl.permute(tile_im, (1, 0, 2))
        tile_re_2d = tl.reshape(tile_re_perm, (16, BLOCK_B * 16))
        tile_im_2d = tl.reshape(tile_im_perm, (16, BLOCK_B * 16))
        out_re_2d, out_im_2d = _cdot(F_re_ptr, F_im_ptr, tile_re_2d, tile_im_2d)
        out_re_perm = tl.reshape(out_re_2d, (16, BLOCK_B, 16))
        out_im_perm = tl.reshape(out_im_2d, (16, BLOCK_B, 16))
        tile_re = tl.permute(out_re_perm, (1, 0, 2)).to(tl.float16)
        tile_im = tl.permute(out_im_perm, (1, 0, 2)).to(tl.float16)
        
        tile_re = tl.permute(tile_re, (0, 2, 1))
        tile_im = tl.permute(tile_im, (0, 2, 1))
        
    y_re_flat = tl.reshape(tile_re, (BLOCK_B, 256))
    y_im_flat = tl.reshape(tile_im, (BLOCK_B, 256))
    
    if STORE_T:
        b_outer = offs_b // M
        b_inner = offs_b % M
        out_offs = b_outer[:, None] * (256 * M) + offs_n[None, :] * M + b_inner[:, None]
    else:
        out_offs = offs_b[:, None] * 256 + offs_n[None, :]
        
    tl.store(y_re_ptr + out_offs, y_re_flat.to(tl.float16), mask=offs_b[:, None] < B)
    tl.store(y_im_ptr + out_offs, y_im_flat.to(tl.float16), mask=offs_b[:, None] < B)


# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.
    """
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_r = tl.arange(0, 16)
    
    mask = (offs_b[:, None] < rows) & (offs_r[None, :] < R)
    x_re = tl.load(x_re_ptr + offs_b[:, None] * R + offs_r[None, :], mask=mask, other=0.0)
    x_im = tl.load(x_im_ptr + offs_b[:, None] * R + offs_r[None, :], mask=mask, other=0.0)
    
    offs_16 = tl.arange(0, 16)
    MT_re = tl.load(M_re_ptr + offs_16[None, :] * 16 + offs_16[:, None])
    MT_im = tl.load(M_im_ptr + offs_16[None, :] * 16 + offs_16[:, None])
    
    y_re_16, y_im_16 = _cdot(x_re, x_im, MT_re, MT_im)
    
    if STORE_T:
        b_outer = offs_b // M
        b_inner = offs_b % M
        out_offs = b_outer[:, None] * (R * M) + offs_r[None, :] * M + b_inner[:, None]
    else:
        out_offs = offs_b[:, None] * R + offs_r[None, :]
        
    store_mask = (offs_b[:, None] < rows) & (offs_r[None, :] < R)
    tl.store(y_re_ptr + out_offs, y_re_16.to(tl.float16), mask=store_mask)
    tl.store(y_im_ptr + out_offs, y_im_16.to(tl.float16), mask=store_mask)


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).
    """
    pid_m0 = tl.program_id(0)
    pid_M = tl.program_id(1)
    pid_r = tl.program_id(2)
    
    offs_m0 = pid_m0 * BLOCK_M0 + tl.arange(0, BLOCK_M0)
    offs_M = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)
    
    mask = (offs_m0[:, None] < m0) & (offs_M[None, :] < M)
    in_offs = pid_r * (m0 * M) + offs_m0[:, None] * M + offs_M[None, :]
    
    val_re = tl.load(x_re_ptr + in_offs, mask=mask, other=0.0)
    val_im = tl.load(x_im_ptr + in_offs, mask=mask, other=0.0)
    
    tw_offs = offs_m0[:, None] * M + offs_M[None, :]
    w_re = tl.load(tw_re_ptr + tw_offs, mask=mask, other=0.0)
    w_im = tl.load(tw_im_ptr + tw_offs, mask=mask, other=0.0)
    
    out_re = val_re * w_re - val_im * w_im
    out_im = val_re * w_im + val_im * w_re
    
    if STORE_T:
        out_offs = pid_r * (M * m0) + offs_M[None, :] * m0 + offs_m0[:, None]
    else:
        out_offs = pid_r * (m0 * M) + offs_m0[:, None] * M + offs_M[None, :]
        
    tl.store(y_re_ptr + out_offs, out_re.to(tl.float16), mask=mask)
    tl.store(y_im_ptr + out_offs, out_im.to(tl.float16), mask=mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )


def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )


def _lookup_tw(plan, m0, M, N_i):
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. The Bailey twiddle fuses into F2-A (BAILEY_EPILOGUE=True), and
    the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]
      2. F2-A:           length-N2 FFT over (B*N1) signals with Bailey epilogue
      3. T2 (transpose): Z[b, n1, k2] -> Z'[b, k2, n1]
      4. F2-B:           length-N1 FFT over (B*N2) signals with strided store
    """
    N1 = plan['N1']
    N2 = plan['N2']
    
    _transpose(in_re, in_im, mid_re, mid_im, B, N2, N1)
    
    f2_kernel[(B * N1,)](
        mid_re, mid_im, out_re, out_im,
        plan['tw_re_n2'], plan['tw_im_n2'],
        plan['perm_n2'],
        plan['bt_re'], plan['bt_im'],
        N1, 0,
        N=N2, LOG2_N=plan['LOG2_N2'],
        BAILEY_EPILOGUE=True, STRIDED_STORE=False
    )
    
    _transpose(out_re, out_im, mid_re, mid_im, B, N1, N2)
    
    f2_kernel[(B * N2,)](
        mid_re, mid_im, out_re, out_im,
        plan['tw_re_n1'], plan['tw_im_n1'],
        plan['perm_n1'],
        plan['tw_re_n1'], plan['tw_im_n1'], # sentinels
        N2, N1 * N2,
        N=N1, LOG2_N=plan['LOG2_N1'],
        BAILEY_EPILOGUE=False, STRIDED_STORE=True
    )


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]
      2. FFT-A: length-256 FFT along last axis -> Y[b, n1, k2]
      3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]
      5. FFT-B: length-256 FFT along last axis -> V[b, k2, k1]
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]   (final in b0)
    """
    
    _transpose(in_re, in_im, b0_re, b0_im, B, 256, 256)
    
    _fft_chunk(b0_re, b0_im, b1_re, b1_im, B * 256, 256, plan['f4_plan'])
    
    _scale(b1_re, b1_im, b0_re, b0_im, B, 256, 256, plan['bt_re'], plan['bt_im'])
    
    _transpose(b0_re, b0_im, b1_re, b1_im, B, 256, 256)
    
    _fft_chunk(b1_re, b1_im, b2_re, b2_im, B * 256, 256, plan['f4_plan'])
    
    _transpose(b2_re, b2_im, b0_re, b0_im, B, 256, 256)


# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.
    """
    if len(chunks) == 1:
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, chunks[0], plan)
        return out_re, out_im
        
    m0 = chunks[0]
    M = math.prod(chunks[1:])
    Ni = m0 * M
    
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)
    
    t2_re, t2_im = _f6_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)
    
    t3_re, t3_im = cyc.next()
    bt_re, bt_im = _lookup_tw(plan, m0, M, Ni)
    _scale(t2_re, t2_im, t3_re, t3_im, rows, m0, M, bt_re, bt_im, store_t=False)
    
    t4_re, t4_im = cyc.next()
    _transpose(t3_re, t3_im, t4_re, t4_im, rows, m0, M)
    
    t5_re, t5_im = cyc.next()
    _fft_chunk(t4_re, t4_im, t5_re, t5_im, rows * M, m0, plan, M=1, store_t=False)
    
    out_re, out_im = cyc.next()
    _transpose(t5_re, t5_im, out_re, out_im, rows, M, m0)
    
    return out_re, out_im


def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.
    """
    if len(chunks) == 1:
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, chunks[0], plan, M=1, store_t=False)
        return out_re, out_im
        
    m0 = chunks[0]
    M = math.prod(chunks[1:])
    Ni = m0 * M
    
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)
    
    t2_re, t2_im = _f7_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)
    
    t4_re, t4_im = cyc.next()
    bt_re, bt_im = _lookup_tw(plan, m0, M, Ni)
    _scale(t2_re, t2_im, t4_re, t4_im, rows, m0, M, bt_re, bt_im, store_t=True)
    
    out_re, out_im = cyc.next()
    _fft_chunk(t4_re, t4_im, out_re, out_im, rows * M, m0, plan, M=M, store_t=True)
    
    return out_re, out_im
