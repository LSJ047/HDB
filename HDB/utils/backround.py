import torch

class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = torch.logical_or(x >= 1e-6, g < 0.0)
        t = pass_through_if+0.0

        return grad1 * t