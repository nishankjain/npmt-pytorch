import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import AdaptiveSoftmax
from fairseq.models import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model,
    register_model_architecture,
)

@register_model('npmt')
class NPMTModel(FairseqModel):
    def __init__(self, encoder, decoder):
        return super().__init__(encoder, decoder)
    
    @staticmethod
    def add_args(parser):
        return super().add_args(parser)