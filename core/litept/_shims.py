"""Pure-PyTorch stand-ins for the compiled helpers upstream LitePT imports.

* ``flash_attn.flash_attn_varlen_qkvpacked_func`` becomes
  ``torch.nn.functional.scaled_dot_product_attention`` over the serialized
  patches (batched when every patch has the same length, which is the
  case for a single crop after LitePT's padding; per patch otherwise).
* ``torch_scatter.segment_csr`` becomes ``index_add_`` / ``scatter_reduce``
  over the CSR pointer.

Both are correctness fallbacks, slower than the compiled originals but
free of extra dependencies. spconv stays the one compiled dependency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attn_varlen_qkvpacked_func(
    qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, **_ignored,
):
    """Attention over packed variable-length sequences.

    ``qkv`` is ``[N, 3, H, D]``; ``cu_seqlens`` the ``[P + 1]`` cumulative
    sequence boundaries. Returns ``[N, H, D]`` like FlashAttention.
    """
    boundaries = cu_seqlens.detach().cpu().tolist()
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if not lengths or min(lengths) <= 0 or max(lengths) > max_seqlen:
        raise ValueError("Invalid packed attention sequence lengths")
    n_tokens, three, heads, dim = qkv.shape
    if three != 3 or boundaries[-1] != n_tokens:
        raise ValueError("qkv must be [N, 3, H, D] covering all cu_seqlens tokens")

    if len(set(lengths)) == 1:
        # Equal patches: one batched call, [P, H, L, D].
        length = lengths[0]
        q, k, v = qkv.view(len(lengths), length, 3, heads, dim).unbind(dim=2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, scale=softmax_scale,
        )
        return out.transpose(1, 2).reshape(n_tokens, heads, dim)

    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        q, k, v = qkv[start:end].unbind(dim=1)              # [L, H, D]
        q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))  # [1, H, L, D]
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, scale=softmax_scale,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def segment_csr(src, indptr, reduce="sum"):
    """``torch_scatter.segment_csr`` for a 1-D CSR pointer over dim 0."""
    indptr = indptr.to(torch.long)
    counts = indptr[1:] - indptr[:-1]
    n_segments = int(counts.numel())
    if int(indptr[-1]) != src.shape[0]:
        raise ValueError("indptr must cover every row of src")
    segment_ids = torch.repeat_interleave(
        torch.arange(n_segments, device=src.device), counts,
    )
    out_shape = (n_segments,) + tuple(src.shape[1:])
    if reduce in ("sum", "mean"):
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        out.index_add_(0, segment_ids, src)
        if reduce == "mean":
            denom = counts.clamp(min=1).to(out.dtype)
            out = out / denom.view((-1,) + (1,) * (src.dim() - 1))
        return out
    if reduce in ("min", "max"):
        index = segment_ids.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        return out.scatter_reduce(
            0, index, src, reduce="amin" if reduce == "min" else "amax",
            include_self=False,
        )
    raise ValueError(f"Unsupported reduce: {reduce}")
