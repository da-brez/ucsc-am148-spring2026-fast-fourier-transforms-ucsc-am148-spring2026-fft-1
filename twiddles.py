"""STUDENT FILE: implement the canonical twiddle helpers.

Three patterns (so seven inconsistently-named lecture helpers collapse to ~3):
  1. radix-2 length-N/2 twiddles   make_radix2_twiddles
  2. per-stage radix-16 twiddles   make_radix16_twiddles
  3. Bailey cross-term twiddles    make_bailey_cross_twiddles

Plus two scaffolding tables (full DFT, padded-R DFT) and the bit-reversal
permutation. Use the forward-FFT sign convention exp(-2*pi*i * ...) and
return (re, im) tuples of separate real-valued tensors everywhere.

When you implement each function, the signature should match the docstring
exactly -- the harness expects (re, im) tuples with specific shapes/dtypes,
and sanity_check.py will FAIL if you return something else.
"""

import math

import torch


# =============================================================================
# Pattern 1: radix-2 length-N/2 twiddles  (F2, F3)
# =============================================================================

def make_radix2_twiddles(
    N: int,
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """w_N^k for k in [0, N/2). Returns (tw_re, tw_im), each shape (N//2,).

    Used by the radix-2 butterfly: stage s reads twiddle at index
    (k & (2**s - 1)) * (N >> (s+1)), so the table only needs the lower half
    of one full period."""
    k = torch.arange(N // 2, dtype=dtype, device=device)
    theta = -2.0 * math.pi * k / N
    tw_re = torch.cos(theta)
    tw_im = torch.sin(theta)
    return tw_re, tw_im


# =============================================================================
# Pattern 2: per-stage radix-16 twiddles  (F4; reused by F5/F6/F7 via F4)
# =============================================================================
# The index bookkeeping for this helper is given -- the per-stage permute
# schedule means the column-axis labels at stage s are a mix of already-
# transformed output digits and not-yet-transformed input digits, in an
# order set by the cumulative permutation history. 

def _column_axis_labeling(L: int) -> list[tuple]:
    """Track axis labels through the per-stage permute schedule.

    Convention: input n decomposes as n = sum_i d_i * 16^(L-1-i) with d_0 the
    high digit; output k similarly with e_i. Initial tile has axis i labeled
    ('d', i). At each stage s the kernel applies perm = (s,) + (others in
    original order), bringing axis s to position 0; the four-tl.dot then
    transforms position 0 from ('d', s) to ('e', L-1-s).

    Returns a list of length L; entry s is the tuple of L-1 labels at axis
    positions 1..L-1 of the (16,)*L tile *after* the stage-s permute.
    """
    A = [('d', i) for i in range(L)]
    out = []
    for s in range(L):
        P = [A[s]] + [A[i] for i in range(L) if i != s]
        out.append(tuple(P[1:]))
        A = [('e', L - 1 - s)] + P[1:]
    return out


def make_radix16_twiddles(
    N: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-stage radix-16 Cooley-Tukey twiddles, stacked. Returns
    (tw_re, tw_im), each shape (L, 16, N//16) fp16. L = log_16(N).

    Stage-0 slice is ones (kernel skips the multiply on s == 0). Stage s > 0
    is built from the labeling above via:
        tw[m, c] = exp(-2*pi*i * m * t / 16^(s+1))
        t = sum_{j=0}^{s-1} e_{L-1-j}_value(c) * 16^j
    where e_{L-1-j}_value(c) reads the base-16 digit of c at the position
    given by _column_axis_labeling(L)[s].
    """
    L = int(round(math.log(N, 16)))
    tw_re = torch.ones(L, 16, N // 16, dtype=torch.float16, device=device)
    tw_im = torch.zeros(L, 16, N // 16, dtype=torch.float16, device=device)
    
    if L <= 1:
        return tw_re, tw_im
        
    c_vals = torch.arange(N // 16, device=device)
    stage_labels = _column_axis_labeling(L)
    
    for s in range(1, L):
        labels = stage_labels[s]
        t = torch.zeros(N // 16, dtype=torch.float32, device=device)
        for j in range(s):
            target_label = ('e', L - 1 - j)
            idx = labels.index(target_label)
            # Extracted digit from linear index c
            digit_val = (c_vals // (16 ** (L - 2 - idx))) % 16
            t = t + digit_val * (16 ** j)
            
        m = torch.arange(16, dtype=torch.float32, device=device).unsqueeze(1)
        t_unsq = t.unsqueeze(0)
        theta = -2.0 * math.pi * m * t_unsq / (16 ** (s + 1))
        tw_re[s] = torch.cos(theta).to(torch.float16)
        tw_im[s] = torch.sin(theta).to(torch.float16)
        
    return tw_re, tw_im


# =============================================================================
# Pattern 3: Bailey cross-term twiddles  (F3, F5, F6, F7)
# =============================================================================

def make_bailey_cross_twiddles(
    m0: int,
    M: int,
    N: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """w_N^{n1 * kM} for n1 in [0, m0), kM in [0, M). Returns (re, im), each
    shape (m0, M).

    F3 calls this with dtype=torch.float32 (the radix-2 tier is fp32);
    F5/F6/F7 call it with dtype=torch.float16 (the tcFFT tier is fp16). The
    Bailey identity holds for any N >= m0 * M; in practice N == m0 * M.
    """
    n1 = torch.arange(m0, dtype=torch.float32, device=device).unsqueeze(1)
    kM = torch.arange(M, dtype=torch.float32, device=device).unsqueeze(0)
    theta = -2.0 * math.pi * n1 * kM / N
    tw_re = torch.cos(theta).to(dtype)
    tw_im = torch.sin(theta).to(dtype)
    return tw_re, tw_im


# =============================================================================
# Scaffolding tables
# =============================================================================

def make_dft_matrix(
    N: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full (N, N) DFT matrix. Returns (W_re, W_im).

    W[j, k] = exp(-2*pi*i * j * k / N). Used by F1 (DFT-as-complex-matmul).
    """
    j = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)
    k = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0)
    theta = -2.0 * math.pi * j * k / N
    tw_re = torch.cos(theta).to(dtype)
    tw_im = torch.sin(theta).to(dtype)
    return tw_re, tw_im


def make_dft_R_padded(
    R: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Length-R DFT padded to (16, 16) fp16. Returns (M_re, M_im).

    Pad the length-R row to 16 with zeros, hit it with a (16, 16) matrix whose
    first R columns are F_R (rows wrap mod R), take the first R output rows.
    This makes the >=16x16 tl.dot requirement hold for all R in {2, 4, 8, 16}.
    """
    W_re = torch.zeros(16, 16, dtype=torch.float16, device=device)
    W_im = torch.zeros(16, 16, dtype=torch.float16, device=device)
    
    j = torch.arange(16, device=device).unsqueeze(1)
    k = torch.arange(R, device=device).unsqueeze(0)
    theta = -2.0 * math.pi * (j % R) * k / R
    
    W_re[:, :R] = torch.cos(theta).to(torch.float16)
    W_im[:, :R] = torch.sin(theta).to(torch.float16)
    return W_re, W_im


def bit_reversal_perm(N: int, device: str = 'cuda') -> torch.Tensor:
    """Length-N bit-reversal permutation as a (N,) int32 tensor.

    rev[i] is the integer whose n_bits=log2(N) binary representation is i's
    bits in reversed order.
    """
    n_bits = int(round(math.log2(N)))
    vals = torch.arange(N, dtype=torch.int32, device=device)
    rev = torch.zeros(N, dtype=torch.int32, device=device)
    for i in range(n_bits):
        bit = (vals >> i) & 1
        rev = rev | (bit << (n_bits - 1 - i))
    return rev
