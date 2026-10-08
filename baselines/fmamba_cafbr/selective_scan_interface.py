from __future__ import annotations

import torch
from einops import rearrange

import selective_scan_cuda_core
import selective_scan_cuda_oflex


class _SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
        nrows=1,
        backnrows=-1,
    ):
        if z is not None:
            raise ValueError("The vendored selective_scan oflex path does not support z")
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        if D is not None and D.dtype != torch.float:
            ctx._d_dtype = D.dtype
            D = D.float()
        if delta_bias is not None and delta_bias.dtype != torch.float:
            ctx._delta_bias_dtype = delta_bias.dtype
            delta_bias = delta_bias.float()

        if backnrows > 0:
            if u.shape[1] % (B.shape[1] * backnrows) != 0:
                raise ValueError("u channels must be divisible by B groups * backnrows")
        else:
            backnrows = nrows
        ctx.backnrows = backnrows
        ctx.delta_softplus = delta_softplus

        out, x, *rest = selective_scan_cuda_oflex.fwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            delta_softplus,
            nrows,
            True,
        )
        last_state = x[:, :, -1, 1::2]
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out.to(u.dtype) if not return_last_state else (out.to(u.dtype), last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            delta_bias,
            dout,
            x,
            ctx.delta_softplus,
            ctx.backnrows,
        )
        if getattr(ctx, "squeeze_B", False):
            dB = dB.squeeze(1)
        if getattr(ctx, "squeeze_C", False):
            dC = dC.squeeze(1)
        if D is not None and dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
            dD = dD.to(ctx._d_dtype)
        if delta_bias is not None and ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
            ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
        return (
            du,
            ddelta,
            dA,
            dB,
            dC,
            dD if D is not None else None,
            None,
            ddelta_bias if delta_bias is not None else None,
            None,
            None,
            None,
            None,
        )


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    nrows=1,
    backnrows=-1,
):
    return _SelectiveScanFn.apply(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        return_last_state,
        nrows,
        backnrows,
    )


selective_scan_ref = selective_scan_fn
