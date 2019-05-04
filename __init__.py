# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os, itertools, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import AdaptiveSoftmax
from fairseq.models import (
    FairseqEncoder, FairseqDecoder, FairseqModel, register_model,
    register_model_architecture,
)

from fairseq.tasks import FairseqTask, register_task
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset
)

from .SoftReordering import SoftReordering
from .upload import *


@register_model('npmt')
class NPMT(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--reordering-window-size', type=int, metavar='N',
                            help='window size for reordering layer')
        parser.add_argument('--encoder-lstm-hidden-size', type=int, metavar='N',
                            help='encoder lstm hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')

    @classmethod
    def build_model(cls, args, task):
        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        npmt_encoder = NPMT_Encoder(
            src_dictionary = task.source_dictionary,
            encoder_embed_dim = args.encoder_embed_dim,
            reordering_window_size = args.reordering_window_size,
            encoder_lstm_hidden_size = args.encoder_lstm_hidden_size,
            encoder_lstm_num_layers = args.encoder_layers,
            encoder_embed_dropout = args.encoder_dropout_in,
            encoder_lstm_out_dropout = args.encoder_dropout_out,
            encoder_lstm_bidirectional = args.encoder_bidirectional
        )

        npmt_decoder = NPMT_Decoder(
            tgt_dictionary = task.target_dictionary,
            decoder_embed_dim = args.decoder_embed_dim,
            decoder_lstm_hidden_size = args.decoder_hidden_size,
            decoder_out_embed_dim = args.decoder_out_embed_dim,
            decoder_lstm_num_layers = args.decoder_layers,
            decoder_embed_dropout = args.decoder_dropout_in,
            decoder_lstm_out_dropout = args.decoder_dropout_out,
            encoder_output_units = npmt_encoder.output_units
        )
        return cls(npmt_encoder, npmt_decoder)


class NPMT_Encoder(FairseqEncoder):
    def __init__(
        self, src_dictionary, encoder_embed_dim, encoder_lstm_hidden_size, encoder_lstm_num_layers, encoder_embed_dropout,
        encoder_lstm_out_dropout, encoder_lstm_bidirectional, reordering_window_size
    ):
        super().__init__(src_dictionary)
        self.encoder_lstm_num_layers = encoder_lstm_num_layers
        self.encoder_embed_dropout = encoder_embed_dropout
        self.encoder_lstm_out_dropout = encoder_lstm_out_dropout
        self.encoder_lstm_bidirectional = encoder_lstm_bidirectional
        self.encoder_lstm_hidden_size = encoder_lstm_hidden_size
        self.reordering_window_size = reordering_window_size
        self.best_stamp = 0
        self.last_stamp = 0

        source_token_count = len(src_dictionary)
        self.src_padding_idx = src_dictionary.pad()
        self.embed_tokens = Embedding(source_token_count, encoder_embed_dim, self.src_padding_idx)

        self.lstm = LSTM(
            input_size=encoder_embed_dim,
            hidden_size=encoder_lstm_hidden_size,
            num_layers=encoder_lstm_num_layers,
            dropout=0,
            bidirectional=encoder_lstm_bidirectional
        )

        self.output_units = encoder_lstm_hidden_size
        if encoder_lstm_bidirectional:
            self.output_units *= 2
        
        self.reordering = SoftReordering(encoder_embed_dim, self.reordering_window_size)

    def forward(self, src_tokens, src_lengths):
        # Code for saving the checkpoints from Colab to Google Drive
        if self.training:
            filename_best = './checkpoints/checkpoint_best.pt'
            filename_last = './checkpoints/checkpoint_last.pt'
            if os.path.isfile(filename_best):
                best_stamp = os.stat(filename_best).st_mtime
                if best_stamp != self.best_stamp:
                    self.best_stamp = best_stamp
                    print("Best Stamp: ", self.best_stamp)
                    upload_best()
            if os.path.isfile(filename_last):
                last_stamp = os.stat(filename_last).st_mtime
                if last_stamp != self.last_stamp:
                    self.last_stamp = last_stamp
                    print("Last Stamp: ", self.best_stamp)
                    upload_last()

        bsz, seqlen = src_tokens.size()

        # Apply embedding layer to sequences
        x = self.embed_tokens(src_tokens)   # bsz x seqlen x embed_dim
        
        # Apply input dropout
        x = F.dropout(x, p=self.encoder_embed_dropout, training=self.training)
        
        # Apply Soft Reordering
        x = self.reordering(x)

        # Transpose input from bsz x seqlen x embed_dim to seqlen x bsz x embed_dim
        # for packing and unpacking the input through LSTM layers
        x = x.transpose(0, 1)

        # Pack input into a Packed Sequence
        packed_inputs = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # Apply LSTM layers to the packed input
        if self.encoder_lstm_bidirectional:
            state_size = 2 * self.encoder_lstm_num_layers, bsz, self.encoder_lstm_hidden_size
        else:
            state_size = self.encoder_lstm_num_layers, bsz, self.encoder_lstm_hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outputs, (final_hiddens, final_cells) = self.lstm(packed_inputs, (h0, c0))

        # Unpack LSTM outputs
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0)
        
        # Apply output dropout
        x = F.dropout(x, p=self.encoder_lstm_out_dropout, training=self.training)

        # Combine the bidirectional hidden states and cell states into unidirectional states
        if self.encoder_lstm_bidirectional:
            final_hiddens = reshape_bidirectional_lstm_hiddens(final_hiddens, self.encoder_lstm_num_layers, bsz)
            final_cells = reshape_bidirectional_lstm_hiddens(final_cells, self.encoder_lstm_num_layers, bsz)

        # print("X Size: ", x.size())
        # print("Final Hidden Size: ", final_hiddens.size())
        # print("Final Cell Size: ", final_cells.size())
        return {
            'encoder_out': (x, final_hiddens, final_cells)
        }
    
    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class NPMT_Decoder(FairseqDecoder):
    def __init__(
        self, tgt_dictionary, decoder_embed_dim, decoder_lstm_hidden_size, decoder_out_embed_dim,
        decoder_lstm_num_layers, decoder_embed_dropout, decoder_lstm_out_dropout, encoder_output_units
    ):
        super().__init__(tgt_dictionary)
        self.decoder_embed_dropout = decoder_embed_dropout
        self.decoder_lstm_out_dropout = decoder_lstm_out_dropout
        self.decoder_lstm_num_layers = decoder_lstm_num_layers
        self.decoder_lstm_hidden_size = decoder_lstm_hidden_size

        target_token_count = len(tgt_dictionary)
        tgt_padding_idx = tgt_dictionary.pad()
        self.embed_tokens = Embedding(target_token_count, decoder_embed_dim, tgt_padding_idx)

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
        # print("Prev Output Tokens Size: ", prev_output_tokens)
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

        # initialize previous states
        prev_hiddens = [encoder_hiddens[i] for i in range(self.decoder_lstm_num_layers)]
        prev_cells = [encoder_cells[i] for i in range(self.decoder_lstm_num_layers)]
        if self.encoder_hidden_proj is not None:
            prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
            prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
        input_feed = x.new_zeros(bsz, self.decoder_lstm_hidden_size)

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.decoder_lstm_out_dropout, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            out = F.dropout(out, p=self.decoder_lstm_out_dropout, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.decoder_lstm_hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        attn_scores = None

        # project back to size of vocabulary
        if hasattr(self, 'additional_fc'):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.decoder_lstm_out_dropout, training=self.training)
        x = self.fc_out(x)
        
        # return x
        return x, attn_scores

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def reshape_bidirectional_lstm_hiddens(outs, num_layers, bsz):
    out = outs.view(num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
    return out.view(num_layers, bsz, -1)


@register_model_architecture('npmt', 'npmt_iwslt_de_en')
def npmt_iwslt_de_en(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.reordering_window_size = getattr(args, 'reordering_window_size', 7)
    args.encoder_lstm_hidden_size = getattr(args, 'encoder_lstm_hidden_size', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)


@register_task('load_dataset')
class LoadDataset(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('data', nargs='+', help='path to data directory')
        parser.add_argument('-s', '--source-lang', metavar='SRC', default='de',
                            help='source language')
        parser.add_argument('-t', '--target-lang', metavar='TGT', default='en',
                            help='target language')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        """
        src_dict[0] = <Lua heritage> = tgt_dict[0]
        src_dict[1] = <pad> = tgt_dict[1]
        src_dict[2] = </s> = tgt_dict[2]
        src_dict[3] = <unk> = tgt_dict[3]
        """
        print('| dictionary [{}]: {} types'.format(args.source_lang, len(src_dict)))
        print('| dictionary [{}]: {} types'.format(args.target_lang, len(tgt_dict)))
        return cls(args, src_dict, tgt_dict)
    
    def load_dataset(self, split, combine=False, **kwargs):
        src_dataset = None
        tgt_dataset = None
        
        data_path = self.args.data[0]
        
        src, tgt = self.args.source_lang, self.args.target_lang
        
        src_file = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, src))
        tgt_file = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, tgt))
        
        # Check for source file
        if IndexedDataset.exists(src_file):
            src_dataset = IndexedDataset(src_file, fix_lua_indexing=True)
            print('| {} {} examples'.format(src_file, len(src_dataset)))
        else:
            raise FileNotFoundError('Source file not found: {}'.format(src_file))
        
        # Check for target file
        if IndexedDataset.exists(tgt_file):
            tgt_dataset = IndexedDataset(tgt_file, fix_lua_indexing=True)
            print('| {} {} examples'.format(tgt_file, len(tgt_dataset)))
        else:
            raise FileNotFoundError('Target file not found: {}'.format(tgt_file))

        # src_dataset.sizes - Array of numbers representing the no. of tokens in each src sequence
        # tgt_dataset.sizes - Array of numbers representing the no. of tokens in each tgt sequence

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict
        )
    
    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict