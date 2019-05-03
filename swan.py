import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math

class SWAN(nn.Module):
    def __init__(
        self, tgt_dictionary, decoder_embed_dim, decoder_lstm_hidden_size, decoder_out_embed_dim,
        decoder_lstm_num_layers, decoder_embed_dropout, decoder_lstm_out_dropout, encoder_output_units
    ):
        super().__init__(tgt_dictionary)
        self.max_segment_length = max_segment_length
        self.decoder_embed_dropout = decoder_embed_dropout
        self.decoder_lstm_out_dropout = decoder_lstm_out_dropout
        self.decoder_lstm_num_layers = decoder_lstm_num_layers
        self.decoder_lstm_hidden_size = decoder_lstm_hidden_size

        self.target_token_count = len(tgt_dictionary)
        self.start_symbol = self.target_token_count + 1
        tgt_padding_idx = tgt_dictionary.pad()
        self.embed_tokens = Embedding(self.target_token_count, decoder_embed_dim, tgt_padding_idx)

        self.encoder_hidden_proj = None
        self.encoder_cell_proj = None

        if encoder_output_units != decoder_lstm_hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, decoder_lstm_hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, decoder_lstm_hidden_size)
        
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=decoder_lstm_hidden_size + decoder_embed_dim if layer == 0 else decoder_lstm_hidden_size,
                hidden_size=decoder_lstm_hidden_size,
            )
            for layer in range(self.decoder_lstm_num_layers)
        ])
        
        if decoder_lstm_hidden_size != decoder_out_embed_dim:
            self.additional_fc = Linear(decoder_lstm_hidden_size, decoder_out_embed_dim)
        self.fc_out = Linear(decoder_out_embed_dim, target_token_count, dropout=decoder_lstm_out_dropout)
    
    def forward(self, prev_output_tokens, encoder_out_dict):
        encoder_out = encoder_out_dict['encoder_out']

        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.decoder_embed_dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        T1 = encoder_outs.size(0)
        T2 = seqlen

        self.batch_max_segment_len = math.min(self.max_segment_len, T2)
        encoder_batch_size = encoder_outs.size(1)
        self.logpy = torch.Tensor(encoder_batch_size, T1, T2+1, self.batch_max_segment_len+1).fill_(-float('Inf')).cuda()
        start_vector = prev_output_tokens.new(encoder_batch_size, 1).fill_(self.start_symbol)