import torch
import torch.nn as nn
import torch.autograd as autograd

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

import tokenizers
from tokenizers import Tokenizer, normalizers, pre_tokenizers, models, trainers
from tokenizers.normalizers import NFD, StripAccents, Lowercase, NFC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

import math
from pathlib import Path


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):

        """

        :param x: of shape (batch_size, seq_len)
        :return:
        """
        x = self.embedding(x) * math.sqrt(self.d_model)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # step 1 pe
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):

            for i in range(0, d_model, 2):

                pe[pos, i] = math.sin(pos / math.exp(i / d_model * math.log(10000.0)))
                pe[pos, (i + 1)] = math.cos(pos / math.exp(i / d_model * math.log(10000.0)))

        # Adding batch dimension
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # and here we also tell the model that we dont want to learn the positional encoding, because they are fixed
        # they will be always the same and they will not be leart during the whole training process
        # so we will use requires_grad_(False), this will make this self.pe[:, :x.shape[1], :] whole thing not learnt
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)

        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.d_v = d_model // h

        # Linear Transformation
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask, dropout: nn.Dropout):

        """

        :param query: of shape (batch_size, h, seq_len, d_k)
        :param key: of shape (batch_size, h, seq_len, d_k)
        :param value: of shape (batch_size, h, seq_len, d_v)
        :param mask:
        :param dropout: is an object of the Dropout Class
        :return: (attention_scores @ value), attention_scores
        """

        d_k = query.shape[-1]
        d_v = value.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 1, -1e9)
            # attention_scores.masked_fill_(mask == 0, -1e9)

        # attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):

        """

        :param q: of shape (batch_size, seq_len, d_model)
        :param k: of shape (batch_size, seq_len, d_model)
        :param v: of shape (batch_size, seq_len, d_model)
        :param mask:
        :return:
        """

        # step 1 Linear Transformation
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # step 2 Multi Heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        # transpose
        query = query.transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)
        key = key.transpose(1, 2)

        value = value.view(value.shape[0], value.shape[1], self.h, self.d_v)
        value = value.transpose(1, 2)

        # step 3 output, self.attention_scores
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # step 4 concat
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(x.shape[0], -1, self.d_model)

        # step 5 linear transformation
        output = self.w_o(x)

        return output


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.features = features
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        print(f" this is the shape of x {x.shape}")
        print(f"this is the shape of self.alpha, {self.alpha.shape}")

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        x = self.alpha * (x - mean) / (std + self.eps) + self.bias

        return x


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.features = features
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):

        output = sublayer(x)
        output = self.dropout(output)
        x = x + output
        x = self.norm(x)

        return x


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # Linear Transformation
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        output = self.linear_1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear_2(output)

        return output


class EncoderBlock(nn.Module):

    def __init__(self,
                 features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.dropout_self = nn.Dropout(dropout)
        self.norm_self = LayerNormalization(features)

        self.feed_forward_block = feed_forward_block
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = LayerNormalization(features)

        # self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        output_self = self.self_attention_block(x, x, x, src_mask)
        output_self = self.dropout_self(output_self)
        x = x + output_self
        x = self.norm_self(x)

        # x = self.residual_connections[1](x, self.feed_forward_block)
        output_ff = self.feed_forward_block(x)
        output_ff = self.dropout_ff(output_ff)
        x = x + output_ff
        x = self.norm_ff(x)

        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, src_mask):

        for layer in self.layers:

            x = layer(x, src_mask)

        x = self.norm(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self,
                 features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float = 0.1):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.dropout_self = nn.Dropout(dropout)
        self.norm_self = LayerNormalization(features)

        self.cross_attention_block = cross_attention_block
        self.dropout_cross = nn.Dropout(dropout)
        self.norm_cross = LayerNormalization(features)

        self.feed_forward_block = feed_forward_block
        self.dropout_ff = nn.Dropout(dropout)
        self.norm_ff = LayerNormalization(features)

        # self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        # x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        output_self = self.self_attention_block(x, x, x, tgt_mask)
        output_self = self.dropout_self(output_self)
        x = x + output_self
        x = self.norm_self(x)

        # x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,src_mask))
        output_cross = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        output_cross = self.dropout_cross(output_cross)
        x = x + output_cross
        x = self.norm_cross(x)

        # x = self.residual_connections[2](x, self.feed_forward_block)
        output_ff = self.feed_forward_block(x)
        output_ff = self.dropout_ff(output_ff)
        x = x + output_ff
        x = self.norm_ff(x)

        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        for layer in self.layers:

            x = layer(x, encoder_output, src_mask, tgt_mask)

        x = self.norm(x)

        return x


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, tgt_vocab_size: int) -> None:
        super().__init__()

        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size

        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        x = self.proj(x)

        x = nn.functional.log_softmax(x, dim=-1)

        return x


class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)

        return src

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return tgt

    def project(self, x):
        # (batch, seq_len, vocab_size)

        x = self.projection_layer(x)
        return x


def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # # Create the encoder blocks
    # encoder_blocks = []
    # for _ in range(N):
    #     encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    #     feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    #     encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
    #     encoder_blocks.append(encoder_block)
    #
    # # Create the decoder blocks
    # decoder_blocks = []
    # for _ in range(N):
    #     decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    #     decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
    #     feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    #     decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
    #                                  feed_forward_block, dropout)
    #     decoder_blocks.append(decoder_block)
    #
    # # Create the encoder and decoder
    # encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    # decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # encoder
    encoder = Encoder(d_model,
                      nn.ModuleList([EncoderBlock(d_model,
                                                  MultiHeadAttentionBlock(d_model, h, dropout),
                                                  FeedForwardBlock(d_model, d_ff, dropout),
                                                  dropout) for _ in range(N)]))

    # decoder
    decoder = Decoder(d_model,
                      nn.ModuleList([DecoderBlock(d_model,
                                                  MultiHeadAttentionBlock(d_model, h, dropout),
                                                  MultiHeadAttentionBlock(d_model, h, dropout),
                                                  FeedForwardBlock(d_model, d_ff, dropout),
                                                  dropout) for _ in range(N)]))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder,
                              decoder,
                              src_embed,
                              tgt_embed,
                              src_pos,
                              tgt_pos,
                              projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():

        if p.dim() > 1:

            nn.init.xavier_uniform_(p)

    return transformer
