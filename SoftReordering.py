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


class SoftReordering(nn.Module):
    def __init__(self, emb_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.win_unit_clones = nn.ModuleList([])
        self.max_length = 51
        for i in range(self.max_length):
            self.win_unit_clones.append(WinUnit(emb_dim, window_size))

    def forward(self, input):
        sequence_length = input.size(1)
        if sequence_length > self.max_length:
            sequence_length = self.max_length

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