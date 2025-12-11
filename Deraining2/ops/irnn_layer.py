import torch
import torch.nn as nn
from torch.autograd import Function
import irnn

class IRNNFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        input = input.contiguous()
        weight = weight.contiguous() 
        bias = bias.contiguous()
        
        output = irnn.forward(input, weight, bias)[0]
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = irnn.backward(grad_output, input, weight, bias, output)[0]
        
        return grad_input, None, None

class IRNN(nn.Module):
    def __init__(self, in_channels):
        super(IRNN, self).__init__()
        self.conv_left  = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=True)
        self.conv_right = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=True)
        self.conv_up    = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=True)
        self.conv_down  = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=True)
        
        # 初始化
        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.conv_left, self.conv_right, self.conv_up, self.conv_down]:
            nn.init.constant_(m.weight, 0.2) # Alpha 初始值
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Left to Right
        # squeeze() 會把 (C, 1, 1, 1) 變成 (C)，剛好給 C++ 吃
        left = IRNNFunction.apply(x, self.conv_left.weight.squeeze(), self.conv_left.bias)
        
        # 2. Right to Left
        x_flip = torch.flip(x, [3])
        right = IRNNFunction.apply(x_flip, self.conv_right.weight.squeeze(), self.conv_right.bias)
        right = torch.flip(right, [3])
        
        # 3. Up to Down
        x_transpose = torch.transpose(x, 2, 3)
        up = IRNNFunction.apply(x_transpose, self.conv_up.weight.squeeze(), self.conv_up.bias)
        up = torch.transpose(up, 2, 3)
        
        # 4. Down to Up
        x_trans_flip = torch.flip(x_transpose, [3])
        down = IRNNFunction.apply(x_trans_flip, self.conv_down.weight.squeeze(), self.conv_down.bias)
        down = torch.flip(down, [3])
        down = torch.transpose(down, 2, 3)

        return left + right + up + down