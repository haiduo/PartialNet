import torch
import torch.nn as nn
import random
from torch.nn import functional as F
import math


def aggregate(gate_k, I, one, K, sort=False):
    if sort:
        _, ind = gate_k.sort(descending=True)
        gate_k = gate_k[:, ind[0, :]]

    U = [(gate_k[0, i] * one + gate_k[1, i] * I) for i in range(K)]
    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(torch.kron(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate_k

def check01(arr):
    index = -1
    flag = False
    for i in range(len(arr)):
        if arr[i] == 0:
            flag = True
        elif flag and arr[i] == 1:
            index = i
            break

    if index != -1:
        return index
    else:
        return True

class DcSign1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        torch.where(input < 0, torch.tensor(0), torch.tensor(1))
        return input
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(0), 0)
        return grad_input
    
class DcSign2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input

class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, 
                 sort=False, u_regular=False, l_gate=False, index_div=False, n_div=4, print_n_div=False, pre_epoch=100):
        super(DGConv2d, self).__init__()
        self.register_buffer('I', torch.eye(2)) 
        self.register_buffer('one', torch.ones(2, 2))
        self.register_buffer('c_div', torch.tensor(n_div, dtype=torch.int32))
        self.register_buffer('one_channel', torch.tensor(in_channels//n_div, dtype=torch.int32))
        self.K = int(math.log2(in_channels))
        eps = 1e-8 
        gate_init = [eps * random.choice([-1, 1]) for _ in range(self.K)]
        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.consistent_train = False
        if self.consistent_train:
            self.start_traing = False
            self.U_M = torch.zeros((in_channels, out_channels))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sort = sort
        self.u_regular=u_regular
        self.l_gate=l_gate
        self.index_div = index_div
        self.n_div=torch.tensor(n_div)  
        self.pre_epoch = pre_epoch 
        self.print_n_div=print_n_div
        self.print_n_div_eval = True
        if self.u_regular:
            global global_gconv
            from .model_api import global_gconv
            global_gconv.append(in_channels*out_channels)

    def forward(self, x):
        if self.training:
            global global_epoch
            from .model_api import global_epoch
            if self.pre_epoch == 0 or global_epoch >= self.pre_epoch:
                # print(f"global_epoch:{global_epoch}")
                direct_STE = False
                if direct_STE:
                    sign_g = DcSign1.apply(self.gate)
                else:
                    sign_g = DcSign2.apply(self.gate)
                    sign_g = (sign_g + 1)/2

                gate_k = torch.stack((sign_g, 1-sign_g)) 

                index_bool = check01(1-sign_g)  # Check if it is a legal sequence
                if self.l_gate:
                    if isinstance(index_bool, bool):
                        loss_gate = torch.tensor(0).to(self.gate.device)
                    else:
                        # loss_gate = sum(self.gate[index_bool:].abs()).detach()  
                        loss_gate = sum(self.gate[index_bool:].abs())

                    global global_loss_gate
                    from .model_api import global_loss_gate
                    global_loss_gate.append(loss_gate)

                self.c_div = 2**(sum(1-sign_g)).int()
                # if isinstance(index_bool, bool):
                #     self.c_div = 2**(sum(1-sign_g)).int()
                # elif self.index_div:
                #     self.c_div = 2**(sum((1-sign_g)[:index_bool])).int()
                # else:
                #     self.c_div = self.n_div.to(self.c_div.device) 

                if self.print_n_div:
                    global global_n_div
                    from .model_api import global_n_div
                    # global_n_div.append(self.c_div.cpu().numpy().item())
                    global_n_div.append(self.c_div.data)

                if self.u_regular:
                    U_regularizer =  2 ** (self.K  + torch.sum(sign_g))
                    # U_regularizer =  2 ** (self.K  + torch.sum(sign_g))/self.c_div
                    global global_regularizer
                    from .model_api import global_regularizer
                    global_regularizer.append(U_regularizer)

                self.one_channel = (self.in_channels / self.c_div).int()
                # mask_1d = torch.zeros(self.in_channels).to(self.one_channel.device)
                # mask_1d[:self.one_channel] = 1

                U, gate_k = aggregate(gate_k, self.I, self.one, self.K, sort=self.sort)
                # U = U * mask_1d
                if self.consistent_train:
                    self.start_traing = True
                    self.U_M.data = U.data
                masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)
                consistent_PConv = False
                if consistent_PConv:
                    x1, x2 = torch.split(x, [self.one_channel, self.in_channels-self.one_channel], dim=1)
                    pconv_weight = masked_weight[:self.one_channel, :self.one_channel, ...]
                    x1 = F.conv2d(x1, pconv_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                    x_out = torch.cat((x1, x2), 1)
                else:
                    x_out = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                    # x_out[:, self.one_channel:, :, :] = x[:, self.one_channel:, :, :]
            else:
                self.c_div = self.n_div.to(self.gate.device)
                self.one_channel = (self.in_channels / self.c_div).int()
                U = torch.zeros(self.in_channels, self.in_channels, dtype=torch.int32).to(self.gate.device)
                U[:self.one_channel, :self.one_channel] = 1
                masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)
                x_out = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                x_out[:, self.one_channel:, :, :] = x[:, self.one_channel:, :, :]
        else:
            if self.consistent_train and self.start_traing:
                masked_weight = self.conv.weight * self.U_M.view(self.out_channels, self.in_channels, 1, 1)
                x_out = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                # self.one_channel = (self.in_channels / self.c_div).int()
                # x_out[:, self.one_channel:, :, :] = x[:, self.one_channel:, :, :]
            else:
                if self.print_n_div_eval:
                    print(self.c_div)
                    self.print_n_div_eval = False
                pconv_enval = True
                if pconv_enval:
                    x1, x2 = torch.split(x, [self.one_channel, self.in_channels-self.one_channel], dim=1)
                    pconv_weight = self.conv.weight[:self.one_channel, :self.one_channel, ...]
                    x1 = F.conv2d(x1, pconv_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                    x_out = torch.cat((x1, x2), 1)
                    # x_out = F.conv2d(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                else: 
                    group_weight = list(torch.split(self.conv.weight, self.in_channels//self.c_div, dim=1))
                    for i in range(self.c_div):
                        group_weight[i] = torch.split(group_weight[i], self.out_channels//self.c_div, dim=0)[i]
                    group_weight = tuple(group_weight)
                    x_groups = torch.split(x, self.in_channels//self.c_div, dim=1)

                    x_out = [F.conv2d(x_group, weight_group, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
                             for x_group, weight_group in zip(x_groups, group_weight)]
                    
                    x_out = torch.cat(x_out, dim=1)
            
        return x_out


if __name__ == '__main__':
    temp=torch.randn(1,8,32,32)
    conv2 = DGConv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False, sort=True)
    hout, U1, loss = conv2(temp)
    conv2.eval()
    hout, _= conv2(temp)

    conv2_1 = DGConv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
    hout2, U2, loss = conv2_1(temp)
    conv2_1.eval()
    hout, _= conv2_1(temp)