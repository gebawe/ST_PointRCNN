import torch.nn as nn
from typing import List, Tuple


class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False,
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                    instance_norm=instance_norm
                )
            )


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)

###
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
###
# Based on: https://github.com/kuangliu/pytorch-groupnorm

class _GroupNorm(nn.Module):
    dim_to_params_shape = {
        3: (1, 1, 1, 1, 1),
        2: (1, 1, 1, 1),
        1: (1, 1, 1)
    }

    def __init__(self, num_features, dim, num_groups=32, eps=1e-5):
        super(_GroupNorm, self).__init__()
        assert dim in [1, 2, 3], f'Unsupported dimensionality: {dim}'
        params_shape = list(self.dim_to_params_shape[dim])
        params_shape[1] = num_features
        self.weight = nn.Parameter(torch.ones(params_shape))
        self.bias = nn.Parameter(torch.zeros(params_shape))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        self._check_input_dim(x)
        # save original shape
        shape = x.size()

        N = shape[0]
        C = shape[1]
        G = self.num_groups
        assert C % G == 0, 'Channel dim must be multiply of number of groups'

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()

        # restore original shape
        x = x.view(shape)
        return x * self.weight + self.bias

    def _check_input_dim(self, x):
        raise NotImplementedError


class GroupNorm3d(_GroupNorm):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm3d, self).__init__(num_features, 3, num_groups, eps)

    def _check_input_dim(self, x):
        if x.dim() != 5:
            raise ValueError(f'Expected 5D input (got {x.dim()}D input)')


class GroupNorm2d(_GroupNorm):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm2d, self).__init__(num_features, 2, num_groups, eps)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f'Expected 4D input (got {x.dim()}D input)')


class GroupNorm1d(_GroupNorm):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm1d, self).__init__(num_features, 1, num_groups, eps)

    def _check_input_dim(self, x):
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input (got {x.dim()}D input)')
            
###--------------------------------------------------------------------------------------------------




class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))
                ###self.add_module(name + 'bn', GroupNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))
                ###self.add_module(name + 'bn', GroupNorm1d(in_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)

