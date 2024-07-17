import torch

class PrintGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print("Gradient at this layer:", grad_output)
        print("Gradient norm at this layer:", grad_output.norm())
        return grad_output