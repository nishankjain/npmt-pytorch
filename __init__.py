import torch
import torch.nn as nn
import torch.nn.functional as F
from .SimpleLSTMDecoder import SimpleLSTMDecoder
from .SimpleLSTMEncoder import SimpleLSTMEncoder

from fairseq import options, utils
from fairseq.modules import AdaptiveSoftmax
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqModel,
    register_model,
    register_model_architecture
)
import os, itertools
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


@register_model('npmt')
class NPMTModel(FairseqModel):
    def __init__(self, encoder, decoder):
        return super().__init__(encoder, decoder)
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--max-segment-len', type=int, metavar='N', help='maximum segment length in the output')
        parser.add_argument('--num-lower-win-layers', type=int, metavar='N', help='reorder layer')
        parser.add_argument('--use-win-middle', action='store_true', help='reorder layer with window centered at t')
        parser.add_argument('--dec-unit-size', type=int, metavar='N', help='number of hidden units per layer in decoder (uni-directional)')
        parser.add_argument('--word-weight', type=float, metavar='D', help='Use word weight')
        parser.add_argument('--lm-weight', type=float, metavar='D', help='external lm weight')
        parser.add_argument('--lm-path', type=str, metavar='STR', help='external lm path')
        parser.add_argument('--use-resnet-enc', action='store_true', help='use resnet connections in enc')
        parser.add_argument('--use-resnet-dec', action='store_true', help='use resnet connections in dec')
        parser.add_argument('--npmt-dropout', type=float, metavar='D', help='npmt dropout factor')
        parser.add_argument('--rnn-mode', type=float, metavar='D', help='or GRU')
        parser.add_argument('--use-cuda', action='store_true', help='use cuda')
        parser.add_argument('--beam', type=int, metavar='N', help='beam size')
        parser.add_argument('--group-size', type=int, metavar='N', help='group size')
        parser.add_argument('--use-accel', action='store_true', help='use C++/CUDA acceleration')
        parser.add_argument('--conv-kW-size', type=int, metavar='N', help='kernel width for temporal conv layer')
        parser.add_argument('--conv-dW-size', type=int, metavar='N', help='kernel stride for temporal conv layer')
        parser.add_argument('--num-lower-conv-layers', type=int, metavar='N', help='num lower temporal conv layers')
        parser.add_argument('--num-mid-conv-layers', type=int, metavar='N', help='num mid temporal conv layers')
        parser.add_argument('--num-high-conv-layers', type=int, metavar='N', help='num higher temporal conv layers')
        parser.add_argument('--win-attn-type', type=str, metavar='STR', help='ori: original')
        parser.add_argument('--reset-lrate', action='store_true', help='True reset learning rate after reloading')
        parser.add_argument('--use-nnlm', action='store_true', help='True use a separated RNN')
        parser.add_argument('--kwidth', type=int, metavar='N', help='Window width for reordering layer')
        parser.add_argument('--nenclayer', type=int, metavar='N', help='Number of bi-directional GRU encoder layers')
        parser.add_argument('--nhid', type=int, metavar='N', help='Number of hidden units per encoder layer (bi-directional)')
        parser.add_argument('--dropout-src', type=float, metavar='D', help='dropout on source embeddings')
        parser.add_argument('--dropout-tgt', type=float, metavar='D', help='dropout on target embeddings')
        parser.add_argument('--dropout-out', type=float, metavar='D', help='dropout on decoder output')
        parser.add_argument('--dropout-hid', type=float, metavar='D', help='dropout between layers')
        parser.add_argument('--dropout', type=float, metavar='D', help='Overall dropout')
        parser.add_argument('--batchsize', type=int, metavar='N', help='Batch Size')
        parser.add_argument('--optim', type=str, metavar='STR', help='Optimizer')
        # parser.add_argument('--lr', type=float, metavar='D', help='Learning Rate')
        parser.add_argument('--sourcelang', type=str, metavar='STR', help='Source Language')
        parser.add_argument('--targetlang', type=str, metavar='STR', help='Target Language')
        parser.add_argument('--datadir', type=str, metavar='STR', help='Pre-processed data directory')
        parser.add_argument('--model', type=str, metavar='STR', help='Model to be used for train/dev/test')
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the encoder embeddings',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        parser.add_argument(
            '--encoder-dropout', type=float, default=0.1,
            help='encoder dropout probability',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )

    @classmethod
    def build_model(cls, args, task):
        # encoder = None
        # decoder = None
        # return cls(encoder, decoder)
        encoder = SimpleLSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout
        )
        decoder = SimpleLSTMDecoder(
            dictionary=task.target_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            dropout=args.decoder_dropout
        )
        model = cls(encoder, decoder)

        # Print the model architecture.
        print(model)

        return model


@register_model_architecture('npmt', 'npmt_iwslt_de_en')
def npmt_iwslt_de_en(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.max_segment_len = getattr(args, 'max_segment_len', 6)                      # maximum segment length in the output
    args.num_lower_win_layers = getattr(args, 'num_lower_win_layers', 0)            # reorder layer
    args.use_win_middle = getattr(args, 'use_win_middle', True)                     # reorder layer with window centered at t
    args.dec_unit_size = getattr(args, 'dec_unit_size', 512)                        # number of hidden units per layer in decoder (uni-directional)
    args.word_weight = getattr(args, 'word_weight', 0.5)                            # Use word weight.
    args.lm_weight = getattr(args, 'lm_weight', 0.0)                                # external lm weight.
    args.lm_path = getattr(args, 'lm_path', "")                                     # external lm path.
    args.use_resnet_enc = getattr(args, 'use_resnet_enc', False)                    # use resnet connections in enc
    args.use_resnet_dec = getattr(args, 'use_resnet_dec', False)                    # use resnet connections in dec
    args.npmt_dropout = getattr(args, 'npmt_dropout', 0.5)                          # npmt dropout factor
    args.rnn_mode = getattr(args, 'rnn_mode', 'LSTM')                               # or GRU
    args.use_cuda = getattr(args, 'use_cuda', True)                                 # use cuda
    args.beam = getattr(args, 'beam', 10)                                           # beam size
    args.group_size = getattr(args, 'group_size', 512)                              # group size
    args.use_accel = getattr(args, 'use_accel', False)                              # use C++/CUDA acceleration
    args.conv_kW_size = getattr(args, 'conv_kW_size', 3)                            # kernel width for temporal conv layer
    args.conv_dW_size = getattr(args, 'conv_dW_size', 2)                            # kernel stride for temporal conv layer
    args.num_lower_conv_layers = getattr(args, 'num_lower_conv_layers', 0)          # num lower temporal conv layers
    args.num_mid_conv_layers = getattr(args, 'num_mid_conv_layers', 0)              # num mid temporal conv layers
    args.num_high_conv_layers = getattr(args, 'num_high_conv_layers', 0)            # num higher temporal conv layers
    args.win_attn_type = getattr(args, 'win_attn_type', 'ori')                      # ori: original
    args.reset_lrate = getattr(args, 'reset_lrate', False)                          # True reset learning rate after reloading
    args.use_nnlm = getattr(args, 'use_nnlm', False)                                # True use a separated RNN
    args.kwidth = getattr(args, 'kwidth', 7)                                        # Window width for reordering layer
    args.nenclayer = getattr(args, 'nenclayer', 2)                                  # Number of bi-directional GRU encoder layers
    args.nhid = getattr(args, 'nhid', 256)                                          # Number of hidden units per encoder layer (bi-directional)
    args.dropout_src = getattr(args, 'dropout_src', 0)                              # dropout on source embeddings
    args.dropout_tgt = getattr(args, 'dropout_tgt', 0)                              # dropout on target embeddings
    args.dropout_out = getattr(args, 'dropout_out', 0)                              # dropout on decoder output
    args.dropout_hid = getattr(args, 'dropout_hid', 0)                              # dropout between layers
    args.dropout = getattr(args, 'dropout', 0.5)                                    # Overall dropout
    args.batchsize = getattr(args, 'batchsize', 32)                                 # Batch Size
    args.optim = getattr(args, 'optim', 'adam')                                     # Optimizer
    args.lr = getattr(args, 'lr', 0.001)                                            # Learning Rate
    args.sourcelang = getattr(args, 'sourcelang', 'de')                             # Source Language
    args.targetlang = getattr(args, 'targetlang', 'en')                             # Target Language
    args.datadir = getattr(args, 'datadir', 'data-bin/iwslt16.tokenized.de-en')     # Pre-processed data directory
    args.model = getattr(args, 'model', 'npmt')                                     # Model to be used for train/dev/test
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
    # args.encoder_dropout = getattr(args, 'encoder_dropout', 0.1)



@register_task('load_dataset')
class LoadDataset(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', metavar='SRC', default='de',
                            help='source language')
        parser.add_argument('-t', '--target-lang', metavar='TARGET', default='en',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions',type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions',  type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary',type=int,
                            help='amount to upsample primary dataset')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        return cls(args, src_dict, tgt_dict)
    
    def load_dataset(self, split, combine=False, **kwargs):
        # Load a given dataset
        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None
        
        src_datasets = []
        tgt_datasets = []
        data_paths = self.args.data
        for dk, data_path in enumerate(data_paths):
            print(dk, data_path)
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                print("k: ", k)
                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                    print("prefix",prefix)
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                    print("prefix2",prefix)
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} {} examples'.format(k, data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break
        print('here')
        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )
    
    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict