import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math

class WinUnit(nn.Module):
    def __init__(self, emb_dim, window_size):
        super().__init__()
        self.weighted_embeddings = nn.Linear(emb_dim * window_size, window_size)
        
    def forward(self, windowed_padded_seq):
        (bsz, window_width, emb_dim) = windowed_padded_seq.size()
        windowed_padded_seq_reshaped = windowed_padded_seq.view(bsz, -1)
        windowed_weights = self.weighted_embeddings(windowed_padded_seq_reshaped)
        sigmoided_windowed_weights = torch.sigmoid(windowed_weights)
        sigmoided_windowed_weights = sigmoided_windowed_weights.unsqueeze(-1).expand(bsz, window_width, emb_dim)
        weighted_input = sigmoided_windowed_weights * windowed_padded_seq
        score = torch.sum(weighted_input, dim=1)
        self.ht = torch.tanh(score)
        return self.ht

    def clearState(self):
        self.zero_grad()
        self.ht = torch.zeros(self.ht.size())

class SoftReordering(nn.Module):
    def __init__(self, emb_dim, window_size, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.window_size = window_size
        self.winUnit = WinUnit(emb_dim, window_size)
        self.max_number_of_windows = 0
        self.win_unit_clones = []

    def forward(self, input):
        window_width = math.floor(self.window_size / 2)
        self.window_size = window_width * 2 + 1
        padding = (0, 0, window_width, window_width) # adding left and right padding to dim 2
        self.padded_input = F.pad(input, padding, "constant", self.padding_idx)
        
        sequence_length = input.size(1)
        if self.max_number_of_windows < sequence_length:
            # self.winUnit.clearState()
            for i in range(sequence_length - self.max_number_of_windows):
                # self.win_unit_clones.append(self.winUnit)
                self.win_unit_clones.append(copy.deepcopy(self.winUnit))
            self.max_number_of_windows = sequence_length
        # for t in range(sequence_length):
        #     self.win_unit_clones[t].clearState()
        
        self.output = torch.zeros(input.size()).cuda()
        for t in range(sequence_length):
            x = self.padded_input[:, t:t+self.window_size, :]
            ht = self.win_unit_clones[t](x)
            self.output[:, t, :] = ht
        return self.output
    
    # def backward():
    #     grad_padded_input = self.padded_input.new(self.padded_input:size()):zero()
    #     sequence_length = input.size(2)
    #     for t = 1, sequence_length do
    #         x = self.padded_input[:, t: t+self.window_size-1, :]
    #         grad_x = self.win_unit_clones[t].backward()
    #         grad_padded_input[:, t: t+self.window_size-1, :] = grad_x
    #     self.gradInput = grad_padded_inpur#.?? reverse pad?
    #     return self.gradInput