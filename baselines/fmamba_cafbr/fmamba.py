import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence

import torch
import numpy as np

import time
import math
from functools import partial
from typing import Optional, Callable

# import pywt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
try:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
except ModuleNotFoundError:
    def to_2tuple(value):
        return value if isinstance(value, tuple) else (value, value)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    def drop_path(x, drop_prob=0.0, training=False):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            return drop_path(x, self.drop_prob, self.training)
selective_scan_fn = None
selective_scan_ref = None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
selective_scan_fn_v1 = None
selective_scan_ref_v1 = None
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass
if selective_scan_fn is None and selective_scan_fn_v1 is None:
    try:
        from .selective_scan_interface import (
            selective_scan_fn as selective_scan_fn_v1,
            selective_scan_ref as selective_scan_ref_v1,
        )
    except:
        pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=2):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

def show_pic(arr, cmap='jet', vmax=1, vmin=0):
    from matplotlib import pyplot as plt
    plt.imshow(arr, cmap)
    plt.axis('off')
    plt.clim(vmin, vmax)
    plt.tight_layout()
    plt.show()
    plt.savefig("./test_predictions_vmunet/text.png", bbox_inches='tight', pad_inches=0)

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops

def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # self.in_proj_kan = KANLinear(self.d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.ca = ChannelAttention(in_planes=self.d_model*2)
        # self.down = nn.MaxPool2d(kernel_size=2)
        

        # self.x_proj = (
        #     nn.Linear(1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        #     nn.Linear(1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        #     nn.Linear(1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        #     nn.Linear(1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        # )
        # self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        # del self.x_proj

        # self.dt_projs = (
        #     self.dt_init(self.dt_rank, 1, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        #     self.dt_init(self.dt_rank, 1, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        #     self.dt_init(self.dt_rank, 1, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        #     self.dt_init(self.dt_rank, 1, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        # )
        # self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        # self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        # del self.dt_projs

        # self.A_logs = self.A_log_init(self.d_state, 1, copies=4, merge=True) # (K=4, D, N)
        # self.Ds = self.D_init(1, copies=4, merge=True) # (K=4, D, N)


        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0 if selective_scan_fn is not None else self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) #[16,2,192,4096]
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l) #[16,4,192,4096]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) #[16,4,38,4096]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #[16,4,6,4096],[16,4,16,4096],[16,4,16,4096]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) #[16,4,192,4096]
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l) [16,768,4096]
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l) [16,768,4096]
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l) [16,4,16,4096]
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l) [16,4,16,4096]
        Ds = self.Ds.float().view(-1) # (k * d) 768
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768,16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d) 768

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float #[16,4,192,4096]

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #[16,2,192,4096]
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #[16,192,4096]
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #[16,192,4096]

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    

    def forward_corev2(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x1 = x
        x1 = self.down(x)
        B1, C1, H1, W1 = x1.shape
        L1 = H1 * W1

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) #[16,2,192,4096]
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l) #[16,4,192,4096]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) #[16,4,38,4096]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #[16,4,6,4096],[16,4,16,4096],[16,4,16,4096]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) #[16,4,192,4096]
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l) [16,768,4096]
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l) [16,768,4096]
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l) [16,4,16,4096]
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l) [16,4,16,4096]
        Ds = self.Ds.float().view(-1) # (k * d) 768
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768,16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d) 768

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float #[16,4,192,4096]

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #[16,2,192,4096]
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #[16,192,4096]
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #[16,192,4096]

        # down
        x_hwwh1 = torch.stack([x1.view(B, -1, L), torch.transpose(x1, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) #[16,2,192,4096]
        xs1 = torch.cat([x_hwwh1, torch.flip(x_hwwh1, dims=[-1])], dim=1) # (b, k, d, l) #[16,4,192,4096]

        x_dbl1 = torch.einsum("b k d l, k c d -> b k c l", xs1.view(B, K, -1, L), self.x_proj_weight) #[16,4,38,4096]
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts1, Bs1, Cs1 = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #[16,4,6,4096],[16,4,16,4096],[16,4,16,4096]
        dts1 = torch.einsum("b k r l, k d r -> b k d l", dts1.view(B, K, -1, L), self.dt_projs_weight) #[16,4,192,4096]
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs1 = xs1.float().view(B1, -1, L1) # (b, k * d, l) [16,768,4096]
        dts1 = dts1.contiguous().float().view(B1, -1, L1) # (b, k * d, l) [16,768,4096]
        Bs1 = Bs1.float().view(B1, K, -1, L1) # (b, k, d_state, l) [16,4,16,4096]
        Cs1 = Cs1.float().view(B1, K, -1, L1) # (b, k, d_state, l) [16,4,16,4096]
        Ds1 = self.Ds1.float().view(-1) # (k * d) 768
        As1 = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) [768,16]
        dt_projs_bias1 = self.dt_projs_bias.float().view(-1) # (k * d) 768

        out_y1 = self.selective_scan(
            xs1, dts1, 
            As1, Bs1, Cs1, Ds1, z=None,
            delta_bias=dt_projs_bias1,
            delta_softplus=True,
            return_last_state=False,
        ).view(B1, K, -1, L1)
        assert out_y1.dtype == torch.float #[16,4,192,4096]

        inv_y1 = torch.flip(out_y1[:, 2:4], dims=[-1]).view(B1, 2, -1, L1) #[16,2,192,4096]
        wh_y1 = torch.transpose(out_y1[:, 1].view(B1, -1, W1, H1), dim0=2, dim1=3).contiguous().view(B1, -1, L1) #[16,192,4096]
        invwh_y1 = torch.transpose(inv_y1[:, 1].view(B1, -1, W1, H1), dim0=2, dim1=3).contiguous().view(B1, -1, L1) #[16,192,4096]

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    
    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        
        x = x.view(-1, C) 
        xz = self.in_proj(x)
        # xz = self.in_proj_kan(x)
        xz = xz.view(B, H, W, -1).contiguous()
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        # c_z = z.permute(0, 3, 1, 2).contiguous()
        # c_z = self.ca(c_z)
        # c_z = c_z.permute(0, 2, 3, 1).contiguous()

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        # y = c_z * y #1.14
        # z = c_z * z

        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out 
    
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, in_channels * 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        original_dtype = x.dtype
        b, c, h, w = x.size()
        r_size = x.size()
        # show_pic(x[0,0].detach().cpu())

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            ffted = torch.fft.fft2(x, norm="ortho")
            ffted_real = ffted.real
            ffted_imag = ffted.imag

            ffted = torch.cat([ffted_real, ffted_imag], dim=1)

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted))

            w_ffted = self.avg_pool(ffted).view(b,c*2)
            w_ffted = self.fc(w_ffted).view(b,c*2,1,1)
            ffted = ffted * w_ffted.expand_as(ffted)

            ffted_real, ffted_imag = torch.chunk(ffted, 2, dim=1)
            ffted = torch.complex(ffted_real, ffted_imag)

            output = torch.fft.ifft2(ffted, s=(h, w), norm="ortho").real
        # show_pic(output[0,0].detach().cpu())
        return output.to(dtype=original_dtype)

class FourierResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(FourierResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fourier_unit = FourierUnit(in_channels, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x  # 残差连接
        x = self.conv1(x)
        x = self.relu(x)
        x = self.fourier_unit(x)
        x = self.conv2(x)
        return x + residual  # 残差连接
    

class FourierVSSBlock(nn.Module):
    def __init__(self, in_channels):
        super(FourierVSSBlock, self).__init__()
        self.fourier_block = FourierResidualBlock(in_channels)
        self.vss_block = VSSBlock(hidden_dim=in_channels, norm_layer=nn.LayerNorm, attn_drop_rate=0.1, d_state=16)
        # 初始化一个可学习权重，经过 softmax 后得到两个分支的比例
        self.alpha = nn.Parameter(torch.tensor([0.5, 0.5]))  # 初始均分权重

    def forward(self, x):
        Fourier = self.fourier_block(x)

        vss_input = x.permute(0, 2, 3, 1)  # 变换维度 [b, c, h, w] -> [b, h, w, c]
        vss_out = self.vss_block(vss_input)  
        vss_out = vss_out.permute(0, 3, 1, 2)  # 还原形状 [b, h, w, c] -> [b, c, h, w]
        weight = F.softmax(self.alpha, dim=0)  # 保证和为 1
        out = weight[0] * Fourier + weight[1] * vss_out

        return out  # 融合两个分支的结果


class DecoderGuidedBoundarySkipRefinement(nn.Module):
    def __init__(
        self,
        channels,
        reduction=8,
        spatial_kernel=7,
        detail_kernel=3,
        detail_scale_init=0.0,
        gate_bias=0.0,
    ):
        super().__init__()
        hidden = max(channels // int(reduction), 16)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.spatial_gate = nn.Conv2d(4, 1, kernel_size=int(spatial_kernel), padding=int(spatial_kernel) // 2, bias=True)
        self.detail_refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=int(detail_kernel),
                padding=int(detail_kernel) // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_scale_init), dtype=torch.float32))
        self.gate_center = float(torch.sigmoid(torch.tensor(float(gate_bias))).item() ** 2)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.constant_(self.channel_gate[-1].bias, float(gate_bias))
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.constant_(self.spatial_gate.bias, float(gate_bias))

    def forward(self, skip, decoder):
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(skip, 1),
                F.adaptive_avg_pool2d(decoder, 1),
            ],
            dim=1,
        )
        channel_gate = torch.sigmoid(self.channel_gate(pooled))
        spatial_source = torch.cat(
            [
                skip.mean(dim=1, keepdim=True),
                skip.amax(dim=1, keepdim=True),
                decoder.mean(dim=1, keepdim=True),
                decoder.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        spatial_gate = torch.sigmoid(self.spatial_gate(spatial_source))
        modulation = channel_gate * spatial_gate
        low_pass = F.avg_pool2d(skip, kernel_size=3, stride=1, padding=1)
        detail = skip - low_pass
        detail = self.detail_refine(detail)
        detail_scale = self.detail_scale.to(dtype=skip.dtype, device=skip.device)
        return skip * (1.0 + modulation - self.gate_center) + detail_scale * detail


class SkipIdentity(nn.Module):
    def forward(self, skip, decoder=None):
        return skip


class CloudAdaptiveFrequencyBoundaryRefinement(nn.Module):
    def __init__(
        self,
        channels,
        reduction=8,
        spatial_kernel=7,
        detail_kernel=3,
        gate_bias=0.0,
        gate_scale_init=1.0,
        detail_scale_init=0.1,
        context_scale_init=0.05,
        spectral_scale_init=0.05,
        use_fourier=True,
        spectral_reduction=4,
        use_channel_gate=True,
        use_spatial_gate=True,
        use_detail_branch=True,
        use_context_branch=True,
        use_spectral_branch=True,
    ):
        super().__init__()
        hidden = max(channels // int(reduction), 16)
        spectral_channels = max(channels // int(spectral_reduction), 16)
        spatial_kernel = int(spatial_kernel)
        detail_kernel = int(detail_kernel)
        self.use_channel_gate = bool(use_channel_gate)
        self.use_spatial_gate = bool(use_spatial_gate)
        self.use_detail_branch = bool(use_detail_branch)
        self.use_context_branch = bool(use_context_branch)

        if self.use_channel_gate:
            self.channel_gate = nn.Sequential(
                nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            )
        else:
            self.channel_gate = None
        if self.use_spatial_gate:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, bias=True),
            )
        else:
            self.spatial_gate = None
        if self.use_detail_branch:
            self.detail_reduce = nn.Sequential(
                nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.detail_mixer = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=detail_kernel,
                    padding=detail_kernel // 2,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.detail_reduce = None
            self.detail_mixer = None
        if self.use_context_branch:
            self.context_refine = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.context_refine = None
        self.use_fourier = bool(use_fourier) and bool(use_spectral_branch)
        if self.use_fourier:
            self.spectral_reduce = nn.Sequential(
                nn.Conv2d(channels, spectral_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(spectral_channels),
                nn.ReLU(inplace=True),
            )
            self.spectral_fourier = FourierUnit(spectral_channels, spectral_channels)
            self.spectral_expand = nn.Sequential(
                nn.Conv2d(spectral_channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.spectral_reduce = None
            self.spectral_fourier = None
            self.spectral_expand = None

        self.gate_scale = nn.Parameter(torch.tensor(float(gate_scale_init), dtype=torch.float32))
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_scale_init), dtype=torch.float32))
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init), dtype=torch.float32))
        self.spectral_scale = nn.Parameter(torch.tensor(float(spectral_scale_init), dtype=torch.float32))
        initial_gate = 1.0
        if self.use_channel_gate:
            initial_gate *= float(torch.sigmoid(torch.tensor(float(gate_bias))).item())
        if self.use_spatial_gate:
            initial_gate *= float(torch.sigmoid(torch.tensor(float(gate_bias))).item())
        self.gate_center = initial_gate

        if self.use_channel_gate:
            nn.init.zeros_(self.channel_gate[-1].weight)
            nn.init.constant_(self.channel_gate[-1].bias, float(gate_bias))
        if self.use_spatial_gate:
            nn.init.zeros_(self.spatial_gate[-1].weight)
            nn.init.constant_(self.spatial_gate[-1].bias, float(gate_bias))
        if self.use_detail_branch:
            nn.init.zeros_(self.detail_mixer[-1].weight)
        if self.use_context_branch:
            nn.init.zeros_(self.context_refine[-1].weight)
        if self.use_fourier:
            nn.init.zeros_(self.spectral_expand[-1].weight)

    def forward(self, skip, decoder):
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        if self.use_channel_gate:
            pooled = torch.cat(
                [
                    F.adaptive_avg_pool2d(skip, 1),
                    F.adaptive_max_pool2d(skip, 1),
                    F.adaptive_avg_pool2d(decoder, 1),
                    F.adaptive_max_pool2d(decoder, 1),
                ],
                dim=1,
            )
            channel_gate = torch.sigmoid(self.channel_gate(pooled))
        else:
            channel_gate = skip.new_ones((skip.shape[0], skip.shape[1], 1, 1))

        if self.use_spatial_gate:
            diff = (skip - decoder).abs()
            spatial_source = torch.cat(
                [
                    skip.mean(dim=1, keepdim=True),
                    skip.amax(dim=1, keepdim=True),
                    decoder.mean(dim=1, keepdim=True),
                    decoder.amax(dim=1, keepdim=True),
                    diff.mean(dim=1, keepdim=True),
                    diff.amax(dim=1, keepdim=True),
                ],
                dim=1,
            )
            spatial_gate = torch.sigmoid(self.spatial_gate(spatial_source))
        else:
            spatial_gate = skip.new_ones((skip.shape[0], 1, skip.shape[2], skip.shape[3]))
        gate = channel_gate * spatial_gate

        out = skip * (1.0 + self.gate_scale.to(dtype=skip.dtype, device=skip.device) * (gate - self.gate_center))
        skip_low3 = None
        skip_low5 = None
        decoder_low3 = None
        if self.use_detail_branch:
            skip_low3 = F.avg_pool2d(skip, kernel_size=3, stride=1, padding=1)
            skip_low5 = F.avg_pool2d(skip, kernel_size=5, stride=1, padding=2)
            decoder_low3 = F.avg_pool2d(decoder, kernel_size=3, stride=1, padding=1)
            detail = self.detail_reduce(torch.cat([skip - skip_low3, skip - skip_low5, decoder - decoder_low3], dim=1))
            detail = self.detail_mixer(detail)
            out = out + self.detail_scale.to(dtype=skip.dtype, device=skip.device) * detail
        if self.use_context_branch:
            if skip_low5 is None:
                skip_low5 = F.avg_pool2d(skip, kernel_size=5, stride=1, padding=2)
            if decoder_low3 is None:
                decoder_low3 = F.avg_pool2d(decoder, kernel_size=3, stride=1, padding=1)
            context = self.context_refine(torch.cat([skip_low5, decoder_low3], dim=1))
            out = out + self.context_scale.to(dtype=skip.dtype, device=skip.device) * context
        if self.use_fourier:
            if skip_low5 is None:
                skip_low5 = F.avg_pool2d(skip, kernel_size=5, stride=1, padding=2)
            spectral = self.spectral_reduce(skip - skip_low5)
            spectral = self.spectral_fourier(spectral)
            spectral = self.spectral_expand(spectral)
            out = out + self.spectral_scale.to(dtype=skip.dtype, device=skip.device) * spectral
        return out


class DynamicChannelGrouping(nn.Module):
    def __init__(self, in_channels, num_groups, embed_dim=16, group_out_channels=None, return_groups=False):
        """
        :param in_channels: 输入通道数
        :param num_groups: 分组数量
        :param embed_dim: 用于通道嵌入的维度
        :param group_out_channels: 每组输出通道数（默认自动设置为 C // G）
        :param return_groups: 是否返回每组的中间输出（用于可视化/分析）
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.embed_dim = embed_dim
        self.return_groups = return_groups

        # 自动设置输出通道数，保证输出通道数保持为 C
        self.group_out_channels = group_out_channels or (in_channels // num_groups)
        assert self.group_out_channels * num_groups == in_channels, "group_out_channels * num_groups must == in_channels"

        # 通道描述子嵌入器（非空间信息）
        self.embedder = nn.Linear(in_channels, embed_dim)

        # 可训练组中心向量（用于动态分组相似度）
        self.group_centers = nn.Parameter(torch.randn(num_groups, embed_dim))  # [G, D]

        # 每组的通道选择权重 [G, C]
        self.mask_weights = nn.Parameter(torch.randn(num_groups, in_channels))

        # 每组的特征提取模块（卷积 + ReLU）
        self.group_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=self.num_groups),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, self.group_out_channels, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_groups)
        ])

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape

        # === Step 1: 获取每个样本的通道描述 ===
        x_desc = x.mean(dim=[2, 3])            # [B, C]：按空间求均值，表示通道级描述
        emb = self.embedder(x_desc)            # [B, D]：嵌入到D维
        sim = torch.matmul(emb, self.group_centers.t())  # [B, G]：与组中心求相似度
        group_weights = F.softmax(sim, dim=1)  # [B, G]：每个样本对每组的权重（当前未用）

        # === Step 2: 逐组提取特征 ===
        out_groups = []
        for g in range(self.num_groups):
            # 通道加权掩码：结构性通道选择
            mask_g = self.mask_weights[g].view(1, C, 1, 1)  # [1, C, 1, 1]
            x_g = x * mask_g                                # [B, C, H, W]：显式通道选择/抑制

            # 特征提取（每组一个卷积模块）
            out = self.group_modules[g](x_g)                # [B, Cout, H, W]
            out_groups.append(out)

        # === Step 3: 拼接所有组的输出 ===
        out = torch.cat(out_groups, dim=1)  # [B, C, H, W]

        if self.return_groups:
            return out, out_groups  # 返回所有组中间结果
        else:
            return out


class QwenClassSpatialConditioner(nn.Module):
    def __init__(
        self,
        qwen_dim,
        out_channels,
        num_classes,
        query_dim=256,
        temperature=0.7,
        weak_class_indices=None,
        weak_focus_init=1.5,
        other_focus_init=0.75,
        modulation_scale=0.25,
        modulation_init_std=1e-3,
        dropout=0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.modulation_scale = float(modulation_scale)
        self.weak_class_indices = [int(idx) for idx in (weak_class_indices or [])]
        self.key_proj = nn.Conv2d(qwen_dim, int(query_dim), kernel_size=1, bias=False)
        self.class_queries = nn.Parameter(torch.empty(self.num_classes, int(query_dim)))
        nn.init.normal_(self.class_queries, std=0.02)

        focus_values = torch.full((self.num_classes,), float(other_focus_init), dtype=torch.float32)
        for idx in self.weak_class_indices:
            if 0 <= idx < self.num_classes:
                focus_values[idx] = float(weak_focus_init)
        self.class_focus_logits = nn.Parameter(torch.log(torch.expm1(focus_values.clamp_min(1e-4))))

        self.map_to_modulation = nn.Conv2d(self.num_classes, out_channels, kernel_size=1, bias=True)
        nn.init.normal_(self.map_to_modulation.weight, std=float(modulation_init_std))
        nn.init.zeros_(self.map_to_modulation.bias)
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, qwen_bottleneck, stage_feat):
        qwen_map = qwen_bottleneck.to(dtype=stage_feat.dtype)
        if qwen_map.shape[-2:] != stage_feat.shape[-2:]:
            qwen_map = F.interpolate(
                qwen_map,
                size=stage_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        keys = self.key_proj(qwen_map)
        keys_norm = F.normalize(keys.float(), dim=1)
        query_norm = F.normalize(self.class_queries.float(), dim=1)
        scores = torch.einsum("bdhw,kd->bkhw", keys_norm, query_norm)
        scores = scores / max(self.temperature, 1e-4)
        class_maps = F.softmax(scores.flatten(2), dim=-1).view_as(scores)
        class_maps = class_maps * float(class_maps.shape[-2] * class_maps.shape[-1])
        focus = F.softplus(self.class_focus_logits).view(1, -1, 1, 1)
        class_maps = class_maps.to(dtype=qwen_map.dtype) * focus.to(device=qwen_map.device, dtype=qwen_map.dtype)
        class_maps = self.dropout(class_maps)

        modulation_logits = self.map_to_modulation(class_maps)
        modulation = 1.0 + self.modulation_scale * torch.tanh(modulation_logits)
        return modulation.to(dtype=stage_feat.dtype), class_maps


class Fmamba(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=16,
        in_chan=None,
        up_chan=None,
        qwen_bottleneck_dim=None,
        fusion_mode="concat",
        gate_init_bias=-3.0,
        fusion_alpha=1.0,
        fusion_positions=None,
        stage_gate_init_bias=None,
        stage_fusion_alpha=None,
        log_fusion_stats=False,
        class_conditional_fusion=None,
        scgm_num_groups=None,
        skip_refinement=None,
        base_channels=64,
        downsample_stages=4,
    ):
        super(Fmamba, self).__init__()
        if base_channels not in (16, 32, 64):
            raise ValueError("FMamba base_channels must be 16, 32, or 64")
        if downsample_stages not in (4, 5):
            raise ValueError("FMamba downsample_stages must be 4 or 5")
        if qwen_bottleneck_dim is not None and (base_channels != 64 or downsample_stages != 4):
            raise ValueError("Qwen fusion supports only the original base-64 four-downsample model")
        self.base_channels = base_channels
        self.downsample_stages = downsample_stages
        encoder_channels = [base_channels * 2 ** i for i in range(5)]
        expected_in = encoder_channels[:-1]
        expected_up = expected_in[::-1]
        if in_chan is not None and list(in_chan) != expected_in:
            raise ValueError("in_chan must match the configured encoder channels")
        if up_chan is not None and list(up_chan) != expected_up:
            raise ValueError("up_chan must match the configured decoder channels")
        in_chan, up_chan = expected_in, expected_up
        self.fusion_mode = fusion_mode
        if self.fusion_mode not in {"concat", "add", "gated_residual"}:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        self.fusion_positions = set(fusion_positions or ["stage_5"])
        unsupported_positions = self.fusion_positions.difference({"stage_4", "stage_5"})
        if unsupported_positions:
            raise ValueError(f"Unsupported qwen fusion positions: {sorted(unsupported_positions)}")
        self.stage_gate_init_bias = stage_gate_init_bias or {}
        self.stage_fusion_alpha = stage_fusion_alpha or {}
        self.log_fusion_stats = bool(log_fusion_stats)
        self.last_fusion_stats = {}
        self.class_conditional_fusion = class_conditional_fusion or {}
        self.class_conditional_enabled = bool(self.class_conditional_fusion.get("enabled", False))
        self.class_conditional_positions = set(
            self.class_conditional_fusion.get("positions", list(self.fusion_positions))
        )
        self.skip_refinement_cfg = skip_refinement or {}
        self.skip_refinement_enabled = bool(self.skip_refinement_cfg.get("enabled", False))
        # Keep the original module names and Sequential indices so existing
        # base-64 / four-downsample checkpoints remain strictly loadable.
        def conv_stage(channels, pool=False, first_stride=1):
            layers = [nn.MaxPool2d(kernel_size=2)] if pool else []
            for index, (source, target) in enumerate(zip(channels[:-1], channels[1:])):
                layers.extend([nn.Conv2d(source, target, kernel_size=3, padding=1,
                                         stride=first_stride if index == 0 else 1),
                               nn.BatchNorm2d(target), nn.ReLU()])
            return nn.Sequential(*layers)

        self.stage_1 = conv_stage([in_channels, base_channels // 2, base_channels, base_channels],
                                  first_stride=2 if downsample_stages == 5 else 1)
        for index in range(1, 5):
            setattr(self, f"stage_{index + 1}", conv_stage(
                [encoder_channels[index - 1], encoder_channels[index], encoder_channels[index]], pool=True))
        for stage in range(4, 0, -1):
            channels = encoder_channels[stage - 1]
            setattr(self, f"upsample_{stage}", nn.Sequential(nn.ConvTranspose2d(
                encoder_channels[stage], channels, kernel_size=4, stride=2, padding=1)))
        # Preserve parameter order as well as names for legacy AdamW resumes.
        for stage in range(4, 0, -1):
            channels = encoder_channels[stage - 1]
            setattr(self, f"stage_up_{stage}", conv_stage([channels * 2, channels, channels]))
        if downsample_stages == 5:
            # The final full-resolution block has no encoder skip, as in SMP U-Net.
            self.upsample_0 = nn.Sequential(nn.ConvTranspose2d(
                base_channels, base_channels, kernel_size=4, stride=2, padding=1))
            self.stage_up_0 = conv_stage([base_channels, base_channels, base_channels])
        self.final = nn.Sequential(nn.Conv2d(base_channels, num_classes, kernel_size=3, padding=1))

        self.qwen_proj = None
        self.bottleneck_fuse = None
        self.qwen_gate = None
        self.qwen_proj_stage4 = None
        self.qwen_gate_stage4 = None
        self.qwen_class_conditioner = None
        self.qwen_class_conditioner_stage4 = None
        if qwen_bottleneck_dim is not None:
            if "stage_5" in self.fusion_positions:
                self.qwen_proj = nn.Sequential(
                    nn.Conv2d(qwen_bottleneck_dim, 1024, kernel_size=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True),
                )
                if self.fusion_mode == "concat":
                    self.bottleneck_fuse = nn.Sequential(
                        nn.Conv2d(2048, 1024, kernel_size=1, bias=False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(inplace=True),
                    )
                elif self.fusion_mode == "gated_residual":
                    self.qwen_gate = nn.Conv2d(2048, 1024, kernel_size=1, bias=True)
                    nn.init.zeros_(self.qwen_gate.weight)
                    nn.init.constant_(self.qwen_gate.bias, float(self._stage_config_value("stage_5", self.stage_gate_init_bias, gate_init_bias)))
                    self.fusion_alpha = nn.Parameter(torch.tensor(float(self._stage_config_value("stage_5", self.stage_fusion_alpha, fusion_alpha)), dtype=torch.float32))
                if self.class_conditional_enabled and "stage_5" in self.class_conditional_positions:
                    self.qwen_class_conditioner = self._build_class_conditioner(
                        qwen_bottleneck_dim=qwen_bottleneck_dim,
                        out_channels=1024,
                        num_classes=num_classes,
                        stage_name="stage_5",
                    )
            if "stage_4" in self.fusion_positions:
                if self.fusion_mode == "concat":
                    raise ValueError("stage_4 qwen fusion only supports add or gated_residual modes")
                self.qwen_proj_stage4 = nn.Sequential(
                    nn.Conv2d(qwen_bottleneck_dim, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
                if self.fusion_mode == "gated_residual":
                    self.qwen_gate_stage4 = nn.Conv2d(1024, 512, kernel_size=1, bias=True)
                    nn.init.zeros_(self.qwen_gate_stage4.weight)
                    nn.init.constant_(self.qwen_gate_stage4.bias, float(self._stage_config_value("stage_4", self.stage_gate_init_bias, gate_init_bias)))
                    self.fusion_alpha_stage4 = nn.Parameter(torch.tensor(float(self._stage_config_value("stage_4", self.stage_fusion_alpha, fusion_alpha)), dtype=torch.float32))
                if self.class_conditional_enabled and "stage_4" in self.class_conditional_positions:
                    self.qwen_class_conditioner_stage4 = self._build_class_conditioner(
                        qwen_bottleneck_dim=qwen_bottleneck_dim,
                        out_channels=512,
                        num_classes=num_classes,
                        stage_name="stage_4",
                    )

        # self.blocks = nn.ModuleList()
        # for i in range(4):
        #     block = VSSBlock(
        #             hidden_dim=in_chan[i],
        #             norm_layer=nn.LayerNorm,
        #             attn_drop_rate=0.1,
        #             d_state=16,
        #         )
        #     self.blocks.append(block)

        # self.up_blocks = nn.ModuleList()
        # for i in range(4):
        #     block1 = VSSBlock(
        #             hidden_dim=up_chan[i],
        #             norm_layer=nn.LayerNorm,
        #             attn_drop_rate=0.1,
        #             d_state=16,
        #         )
        #     self.up_blocks.append(block1)

        # self.ffa1 = FourierResidualBlock(in_channels=in_chan[0])
        # self.ffa2 = FourierResidualBlock(in_channels=in_chan[1])
        # self.ffa3 = FourierResidualBlock(in_channels=in_chan[2])
        # self.ffa4 = FourierResidualBlock(in_channels=in_chan[3])
        self.blocks = nn.ModuleList()
        for i in range(4):
            block = FourierVSSBlock(in_channels=in_chan[i])
            self.blocks.append(block)

        self.up_blocks = nn.ModuleList()
        for channels in up_chan + ([base_channels] if downsample_stages == 5 else []):
            block1 = FourierVSSBlock(in_channels=channels)
            self.up_blocks.append(block1)

        self.skip_refiners = nn.ModuleList()
        if self.skip_refinement_enabled:
            refinement_type = str(self.skip_refinement_cfg.get("type", "decoder_guided_boundary")).lower()
            reduction = int(self.skip_refinement_cfg.get("reduction", 8))
            spatial_kernel = int(self.skip_refinement_cfg.get("spatial_kernel", 7))
            detail_kernel = int(self.skip_refinement_cfg.get("detail_kernel", 3))
            detail_scale_init = float(self.skip_refinement_cfg.get("detail_scale_init", 0.0))
            gate_bias = float(self.skip_refinement_cfg.get("gate_bias", 0.0))
            stage_names = [f"stage_{stage}" for stage in range(4, 0, -1)]
            fourier_stages = set(self.skip_refinement_cfg.get("fourier_stages", ["stage_4", "stage_3", "stage_2"]))
            for stage_name, channels in zip(stage_names, up_chan):
                if refinement_type in {"cloud_adaptive_frequency_boundary", "cafbr", "frequency_boundary"}:
                    self.skip_refiners.append(
                        CloudAdaptiveFrequencyBoundaryRefinement(
                            channels=channels,
                            reduction=reduction,
                            spatial_kernel=spatial_kernel,
                            detail_kernel=detail_kernel,
                            gate_bias=gate_bias,
                            gate_scale_init=float(self.skip_refinement_cfg.get("gate_scale_init", 1.0)),
                            detail_scale_init=detail_scale_init,
                            context_scale_init=float(self.skip_refinement_cfg.get("context_scale_init", 0.05)),
                            spectral_scale_init=float(self.skip_refinement_cfg.get("spectral_scale_init", 0.05)),
                            use_fourier=bool(self.skip_refinement_cfg.get("use_fourier", True)) and stage_name in fourier_stages,
                            spectral_reduction=int(self.skip_refinement_cfg.get("spectral_reduction", 4)),
                            use_channel_gate=bool(self.skip_refinement_cfg.get("use_channel_gate", True)),
                            use_spatial_gate=bool(self.skip_refinement_cfg.get("use_spatial_gate", True)),
                            use_detail_branch=bool(self.skip_refinement_cfg.get("use_detail_branch", True)),
                            use_context_branch=bool(self.skip_refinement_cfg.get("use_context_branch", True)),
                            use_spectral_branch=bool(self.skip_refinement_cfg.get("use_spectral_branch", True)),
                        )
                    )
                elif refinement_type in {"decoder_guided_boundary", "dgbsr"}:
                    self.skip_refiners.append(
                        DecoderGuidedBoundarySkipRefinement(
                            channels=channels,
                            reduction=reduction,
                            spatial_kernel=spatial_kernel,
                            detail_kernel=detail_kernel,
                            detail_scale_init=detail_scale_init,
                            gate_bias=gate_bias,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported skip_refinement.type: {refinement_type}")
        else:
            for _ in range(4):
                self.skip_refiners.append(SkipIdentity())
        self.skip_refinement_active = self.skip_refinement_enabled

        channel_groups = int(scgm_num_groups) if scgm_num_groups is not None else (4 if in_channels % 4 == 0 else 1)
        if channel_groups <= 0 or in_channels % channel_groups != 0:
            raise ValueError(f"SCGM num_groups must be a positive divisor of in_channels={in_channels}, got {channel_groups}")
        self.scgm_num_groups = channel_groups
        self.channelGroup1 = DynamicChannelGrouping(in_channels=in_channels, num_groups=channel_groups)

    def set_skip_refinement_active(self, active):
        self.skip_refinement_active = bool(active)

    def _refine_skip(self, index, skip, decoder):
        if self.skip_refinement_enabled and self.skip_refinement_active:
            return self.skip_refiners[index](skip, decoder)
        return skip

    @staticmethod
    def _stage_config_value(stage_name, value, default):
        if isinstance(value, dict):
            return value.get(stage_name, default)
        if value is None:
            return default
        return value

    def _build_class_conditioner(self, qwen_bottleneck_dim, out_channels, num_classes, stage_name):
        cfg = self.class_conditional_fusion
        return QwenClassSpatialConditioner(
            qwen_dim=qwen_bottleneck_dim,
            out_channels=out_channels,
            num_classes=num_classes,
            query_dim=int(self._stage_config_value(stage_name, cfg.get("query_dim"), 256)),
            temperature=float(self._stage_config_value(stage_name, cfg.get("temperature"), 0.7)),
            weak_class_indices=cfg.get("weak_class_indices", []),
            weak_focus_init=float(cfg.get("weak_focus_init", 1.5)),
            other_focus_init=float(cfg.get("other_focus_init", 0.75)),
            modulation_scale=float(self._stage_config_value(stage_name, cfg.get("modulation_scale"), 0.25)),
            modulation_init_std=float(cfg.get("modulation_init_std", 1e-3)),
            dropout=float(cfg.get("dropout", 0.0)),
        )

    def _project_qwen_to_stage(self, stage_feat, qwen_bottleneck, projection):
        if qwen_bottleneck is None:
            return None
        if projection is None:
            return None
        qwen_delta = qwen_bottleneck.to(dtype=stage_feat.dtype)
        if qwen_delta.shape[-2:] != stage_feat.shape[-2:]:
            qwen_delta = F.interpolate(
                qwen_delta,
                size=stage_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return projection(qwen_delta)

    def _apply_class_conditioning(self, stage_name, stage_feat, qwen_bottleneck, qwen_delta, conditioner):
        if conditioner is None:
            return qwen_delta
        modulation, class_maps = conditioner(qwen_bottleneck, stage_feat)
        conditioned_delta = qwen_delta * modulation
        if hasattr(self, "last_fusion_regularizers"):
            stage_regularizers = self.last_fusion_regularizers.setdefault(stage_name, {})
            stage_regularizers.update(
                {
                    "class_modulation_l2": (modulation.float() - 1.0).pow(2).mean(),
                }
            )
        if self.log_fusion_stats:
            with torch.no_grad():
                stats = {
                    f"{stage_name}_class_modulation_mean": float(modulation.detach().float().mean().cpu()),
                    f"{stage_name}_class_modulation_std": float(modulation.detach().float().std(unbiased=False).cpu()),
                    f"{stage_name}_class_modulation_min": float(modulation.detach().float().min().cpu()),
                    f"{stage_name}_class_modulation_max": float(modulation.detach().float().max().cpu()),
                    f"{stage_name}_class_map_mean": float(class_maps.detach().float().mean().cpu()),
                    f"{stage_name}_class_map_max": float(class_maps.detach().float().max().cpu()),
                }
                weak_indices = [
                    idx for idx in getattr(conditioner, "weak_class_indices", []) if 0 <= idx < class_maps.shape[1]
                ]
                if weak_indices:
                    stats[f"{stage_name}_weak_class_map_mean"] = float(
                        class_maps[:, weak_indices].detach().float().mean().cpu()
                    )
                    stats[f"{stage_name}_weak_class_focus_mean"] = float(
                        F.softplus(conditioner.class_focus_logits[weak_indices]).detach().float().mean().cpu()
                    )
                self.last_fusion_stats.update(stats)
        return conditioned_delta

    def _record_gated_fusion_stats(self, stage_name, stage_feat, qwen_delta, gate, alpha, fused):
        if not self.log_fusion_stats:
            return
        with torch.no_grad():
            prefix = stage_name
            stats = {
                f"{prefix}_gate_mean": float(gate.detach().float().mean().cpu()),
                f"{prefix}_gate_std": float(gate.detach().float().std(unbiased=False).cpu()),
                f"{prefix}_gate_min": float(gate.detach().float().min().cpu()),
                f"{prefix}_gate_max": float(gate.detach().float().max().cpu()),
                f"{prefix}_fusion_alpha": float(alpha.detach().float().cpu()) if hasattr(alpha, "detach") else float(alpha),
                f"{prefix}_qwen_delta_norm": float(qwen_delta.detach().float().norm().cpu()),
                f"{prefix}_fmamba_feat_norm": float(stage_feat.detach().float().norm().cpu()),
                f"{prefix}_fused_feat_norm": float(fused.detach().float().norm().cpu()),
            }
            self.last_fusion_stats.update(stats)
            if stage_name == "stage_5":
                self.last_fusion_stats.update(
                    {
                        "fusion_gate_mean": stats["stage_5_gate_mean"],
                        "fusion_gate_std": stats["stage_5_gate_std"],
                        "fusion_gate_min": stats["stage_5_gate_min"],
                        "fusion_gate_max": stats["stage_5_gate_max"],
                        "fusion_alpha": stats["stage_5_fusion_alpha"],
                        "qwen_delta_norm": stats["stage_5_qwen_delta_norm"],
                        "fmamba_feat_norm": stats["stage_5_fmamba_feat_norm"],
                        "fused_feat_norm": stats["stage_5_fused_feat_norm"],
                    }
                )

    def _fuse_stage_gated_residual(self, stage_feat, qwen_delta, gate_layer, alpha_param, stage_name):
        gate = torch.sigmoid(gate_layer(torch.cat([stage_feat, qwen_delta], dim=1)))
        alpha = alpha_param.to(device=stage_feat.device, dtype=stage_feat.dtype)
        fused = stage_feat + alpha * gate * qwen_delta
        if hasattr(self, "last_fusion_regularizers"):
            qwen_delta_float = qwen_delta.float()
            stage_feat_float = stage_feat.float()
            stage_regularizers = self.last_fusion_regularizers.setdefault(stage_name, {})
            stage_regularizers.update(
                {
                    "gate_mean": gate.float().mean(),
                    "gate_l2": gate.float().pow(2).mean(),
                    "delta_to_feat_ratio": qwen_delta_float.norm()
                    / stage_feat_float.detach().norm().clamp_min(1e-6),
                    "alpha_abs": alpha.float().abs(),
                }
            )
        self._record_gated_fusion_stats(stage_name, stage_feat, qwen_delta, gate, alpha, fused)
        return fused

    def _fuse_stage4(self, stage_4, qwen_bottleneck):
        qwen_delta = self._project_qwen_to_stage(stage_4, qwen_bottleneck, self.qwen_proj_stage4)
        if qwen_delta is None:
            return stage_4
        qwen_delta = self._apply_class_conditioning(
            "stage_4",
            stage_4,
            qwen_bottleneck,
            qwen_delta,
            self.qwen_class_conditioner_stage4,
        )
        if self.fusion_mode == "add":
            return stage_4 + qwen_delta
        if self.fusion_mode == "gated_residual":
            return self._fuse_stage_gated_residual(
                stage_4,
                qwen_delta,
                self.qwen_gate_stage4,
                self.fusion_alpha_stage4,
                "stage_4",
            )
        return stage_4

    def _fuse_bottleneck(self, stage_5, qwen_bottleneck):
        if qwen_bottleneck is None:
            return stage_5
        qwen_delta = self._project_qwen_to_stage(stage_5, qwen_bottleneck, self.qwen_proj)
        if qwen_delta is None:
            return stage_5
        qwen_delta = self._apply_class_conditioning(
            "stage_5",
            stage_5,
            qwen_bottleneck,
            qwen_delta,
            self.qwen_class_conditioner,
        )
        if self.fusion_mode == "add":
            fused = stage_5 + qwen_delta
        elif self.fusion_mode == "gated_residual":
            fused = self._fuse_stage_gated_residual(
                stage_5,
                qwen_delta,
                self.qwen_gate,
                self.fusion_alpha,
                "stage_5",
            )
        else:
            fused = self.bottleneck_fuse(torch.cat([stage_5, qwen_delta], dim=1))
        return fused

    def forward(self, x, qwen_bottleneck=None, return_features=False, return_pyramid=False):
        x = x.float()
        self.last_fusion_stats = {}
        self.last_fusion_regularizers = {}

        x = self.channelGroup1(x)

        if x.shape[-2] % (2 ** self.downsample_stages) or x.shape[-1] % (2 ** self.downsample_stages):
            raise ValueError(f"Input height and width must be divisible by {2 ** self.downsample_stages}")
        if return_pyramid and self.downsample_stages != 4:
            raise ValueError("Mask2Former pyramid currently supports only four-downsample FMamba")
        skips = []
        for index in range(4):
            x = getattr(self, f"stage_{index + 1}")(x)
            x = self.blocks[index](x)
            skips.append(x)
        x = self.stage_5(x)
        x = self._fuse_bottleneck(x, qwen_bottleneck)
        # The original optional Qwen stage-4 fusion remains unchanged.
        if self.downsample_stages == 4:
            skips[-1] = self._fuse_stage4(skips[-1], qwen_bottleneck)
        decoded = []
        for index, stage in enumerate(range(4, 0, -1)):
            x = getattr(self, f"upsample_{stage}")(x)
            skip = self._refine_skip(index, skips[stage - 1], x)
            x = getattr(self, f"stage_up_{stage}")(torch.cat([x, skip], dim=1))
            x = self.up_blocks[index](x)
            decoded.append(x)
        if self.downsample_stages == 5:
            x = self.upsample_0(x)
            x = self.stage_up_0(x)
            x = self.up_blocks[4](x)
        if return_pyramid:
            return [F.avg_pool2d(feature, 4) for feature in reversed(decoded)]
        output = self.final(x)
        if return_features:
            return output, x
        return output


if __name__ == '__main__':
    model= Fmamba(11)

    image = torch.randn(32, 16, 256, 256)

    output = model(image)

    print("input:", image.shape)
    print("output:", output.shape)
