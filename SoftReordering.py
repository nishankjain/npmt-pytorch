import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class winUnit(nn.Module):
    def __init__(self, emb_dim, window_size):
        super().__init__()
        self.weighted_embeddings = nn.Linear(emb_dim * window_size, window_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(padded_sequence_window):
        weighted_embeddings_window = self.weighted_embeddings(padded_sequence_window)
        weighted_embeddings_window_sigmoid = self.sigmoid(weighted_embeddings_window)
        weighted_input = weighted_embeddings_window_sigmoid * padded_sequence_window
        score = torch.sum(weighted_input, dim=1)
        self.ht = self.tanh(score)
        print("ht size: ", ht.size())
        return self.ht

    def clearState(self):
        self.zero_grad()
        self.ht = torch.zeros(self.ht.size())

class winAttn(nn.Module):
    def __init__(self, emb_dim, window_size, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.window_size = window_size
        self.winUnit = winUnit(emb_dim, window_size)
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
                self.win_unit_clones.append(copy.deepcopy(self.winUnit))
            self.max_number_of_windows = sequence_length
        # for t in range(sequence_length):
        #     self.win_unit_clones[t].clearState()
        
        self.output = torch.zeros(input.size())
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