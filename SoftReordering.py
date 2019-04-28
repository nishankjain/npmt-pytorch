import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math
import torch.autograd as autograd

class WinUnit(nn.Module):
    def __init__(self, emb_dim, window_size):
        super().__init__()
        self.weighted_embeddings = nn.Linear(window_size * emb_dim, window_size)
        
    def forward(self, windowed_padded_seq):
        (bsz, window_size, emb_dim) = windowed_padded_seq.size()    # B x W x D
        windowed_padded_seq_reshaped = windowed_padded_seq.view(bsz, -1)    # B x (WxD)
        windowed_weights = self.weighted_embeddings(windowed_padded_seq_reshaped)   # B x W
        sigmoided_windowed_weights = torch.sigmoid(windowed_weights)    # B x W
        weights = sigmoided_windowed_weights.unsqueeze(-1).expand(bsz, window_size, emb_dim)    # B x W x D
        weighted_input = weights * windowed_padded_seq  # B x W x D
        score = torch.sum(weighted_input, dim=1)    # B x D
        self.ht = torch.tanh(score)     # B x D
        return self.ht


# class ManageGradients(autograd.Function):
#     @staticmethod
#     def forward(ctx, input, win_unit_clones, sequence_length, window_size):
#         ctx.win_unit_clones = win_unit_clones
#         ctx.sequence_length = sequence_length
#         ctx.window_size = window_size
#         ctx.input = input
#         window_width = math.floor(ctx.window_size / 2)
#         padding = (0, 0, window_width, window_width) # adding left and right padding to dim 2
#         ctx.padded_input = F.pad(ctx.input, padding, "constant", 0)
#         print("Input Size: ", ctx.input.size())
#         print("Padded Input Size: ", ctx.padded_input.size())
        
#         ctx.output = torch.zeros(ctx.input.size(), requires_grad=True).cuda()
#         for t in range(sequence_length):
#             x = ctx.padded_input[:, t:t+ctx.window_size, :]
#             ht = ctx.win_unit_clones[t](x)
#             ctx.output[:, t, :] = ht
#         print("Output Size: ", ctx.output.size())
#         return ctx.output
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         print("Grad Output Size: ", grad_output.size())
#         grad_padded_input = ctx.padded_input.new_zeros(ctx.padded_input.size())
#         for t in range(ctx.sequence_length):
#             x = ctx.padded_input[:, t:t+ctx.window_size, :]
#             grads = autograd.grad(ctx.output[:, t, :], ctx.input[:, t:t+ctx.window_size, :], grad_output[:, t, :])
#             print(grads)
#             # for name, param in ctx.win_unit_clones[t].named_parameters():
#             #     print(name)
#             #     print(param.data.size())
#             #     print(param.size())
#             #     print(param.grad)
#             # print(ctx.win_unit_clones[t].size())
#             # print(grad_output.size())
#             # print(grad_output[:, t, :])
#             # grad_x = ctx.win_unit_clones[t].backward(x, grad_output[:, t, :])
#             # grad_padded_input[:, t:t+ctx.window_size, :] = grad_x
#         # print(grad_padded_input)
#         return grad_padded_input


# class SoftReordering(nn.Module):
#     def __init__(self, emb_dim, window_size, padding_idx):
#         super().__init__()
#         self.padding_idx = padding_idx
#         self.window_size = window_size
#         self.emb_dim = emb_dim
#         self.winUnit = WinUnit(emb_dim, window_size)
#         self.max_number_of_windows = 0
#         self.win_unit_clones = []

#     def forward(self, input):
#         sequence_length = input.size(1)
#         if self.max_number_of_windows < sequence_length:
#             for i in range(sequence_length - self.max_number_of_windows):
#                 self.win_unit_clones.append(copy.deepcopy(self.winUnit))
#                 self.win_unit_clones[-1].load_state_dict(self.winUnit.state_dict())
#             self.max_number_of_windows = sequence_length
#         # print("Max Windows: ", self.max_number_of_windows)
        
#         # print("Unit 1")
#         # for name, param in self.win_unit_clones[0].named_parameters():
#         #     print(name)
#         #     print(param.data)
#         #     print(param.size())
#         # print("Unit 2")
#         # for name, param in self.win_unit_clones[1].named_parameters():
#         #     print(name)
#         #     print(param.data)
#         #     print(param.size())
#         return ManageGradients.apply(input, self.win_unit_clones, sequence_length, self.window_size)


class SoftReordering(nn.Module):
    def __init__(self, emb_dim, window_size, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.window_size = window_size
        self.emb_dim = emb_dim
        # self.winUnit = WinUnit(emb_dim, window_size)
        # self.max_number_of_windows = 0
        self.win_unit_clones = nn.ModuleList([])
        for i in range(175):
            self.win_unit_clones.append(WinUnit(emb_dim, window_size))
            # self.win_unit_clones.append(copy.deepcopy(self.winUnit))
            # self.win_unit_clones[i].load_state_dict(self.winUnit.state_dict())

    def forward(self, input):
        sequence_length = input.size(1)
        # if self.max_number_of_windows < sequence_length:
            # for i in range(sequence_length - self.max_number_of_windows):
            #     self.win_unit_clones.append(copy.deepcopy(self.winUnit))
            #     self.win_unit_clones[-1].load_state_dict(self.winUnit.state_dict())
            # self.max_number_of_windows = sequence_length
        # print("Max Windows: ", self.max_number_of_windows)

        window_width = math.floor(self.window_size / 2)
        padding = (0, 0, window_width, window_width) # adding left and right padding to dim 2
        self.padded_input = F.pad(input, padding, "constant", 0)
        
        self.output = torch.zeros(input.size()).cuda()
        for t in range(sequence_length):
            x = self.padded_input[:, t:t+self.window_size, :]
            ht = self.win_unit_clones[t](x)
            self.output[:, t, :] = ht
        
        # print("Unit 1")
        # for name, param in self.win_unit_clones[0].named_parameters():
        #     print(name)
        #     print(param.data)
        #     print(param.size())
        # print("Unit 2")
        # for name, param in self.win_unit_clones[1].named_parameters():
        #     print(name)
        #     print(param.data)
        #     print(param.size())
        return self.output