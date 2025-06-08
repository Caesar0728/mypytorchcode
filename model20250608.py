import torch
import torch.nn as nn
import torch.autograd as autograd

import tokenizers
from tokenizers import Tokenizer, pre_tokenizers, normalizers, trainers, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFC, StripAccents, Lowercase, NFD
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel

import math
from pathlib import Path


class InputEmbeddings(nn.Module):
    """
    d_model in the paper means the embedding vector size,
    the embedding matrix is of shape (vocab_size * d_model)

    self.embedding 里面会把seq_len个tokens/indexes转化成onehot vector

    ###### 实际操作 ######
    nn.Embedding 这里传入的是一个句子的token，比方说一个句子为i like you， seq_len=3，转化成tokens可能是[31， 28， 8989]
    传给nn.Embedding是传[31， 28， 8989],
    而并非传入[31， 28， 8989]转成one-hot之后的结果，不是传入一个size为 seq_len * vocab_size（3 * vocab_size）的one-hot矩阵,
    31， 28， 8989这些数字都是索引，对于输入的每个单词索引，
    nn.Embedding 层会查找embedding matrix 中(embedding matrix的shape为vocab_size * d_model)
    对应的行的向量(shape 为1 * d_model)作为embedding vector嵌入向量。
    eg. 比如说 I 对应的 index 是上面的 31， 那久拿着 31 去查 embedding matrix 的第31行，
    这个第31行的向量就是 I 这个词 对应的 embedding vector嵌入向量
    # 因为从一个一维的数字索引转换成一个二维的embedding vector， 所以就等于增加了一个维度
    # 因此输出的结果x的shape 为 (batch_size, seq_len, d_model)

    ###### 理解操作 ######
    个人觉得， 先把这个[31， 28， 8989]转成one-hot之后的方式更利于理解， 因为可以设计到矩阵的相乘， 具体怎么做 如下：
    step 1 将[31， 28， 8989]转化成one-hot matrix， 即 3 * 10000 的matrix（因为总共有10000个词）
    第一行的第31个位置是1， 其余位置的值是0， 同理第二行的第28个位置是1， 其余位置的值是0，第三行的第8989个位置是1， 其余位置的值是0，
    所以得到的one-hot matrix的shape 是 (seq_len, vocab_size)
    step 2 将 这个one-hot matrix 乘以 embedding matrix
    得到 (seq_len, vocab_size) * (vocab_size * d_model) = (seq_len, d_model)
    ###### 这个与上面通过index去查找对应行的效果是完全一致的 ######

    当然这里只考虑了一句话， 实际上一个batch里面有多句话，因此实际情况中， 我们有batch_size
    就变成了 (batch_size，seq_len) -> (batch_size，seq_len, vocab_size) -> 乘以 embedding matrix
    -> (batch_size，seq_len, d_model)
    ###### 理解操作 ######

    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        # 这里的d_model就是embedding_size
        self.d_model = d_model

        # 就是vocab_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """

        :param x: of shape batch_size * seq_len
        :return:
        """
        # 通过内置nn.Embedding 来生成embedding_matrix
        # The reason we increase the weights of embedding values by math.sqrt(d_model)
        # before the addition is to make the positional encoding relatively smaller.
        # This means the original meaning in the embedding vector won’t be lost when we add them together.

        # My hypothesis is that the authors tried rescaling the embeddings by various numbers
        # (as they certainly did with attention), and this particular rescaling happened to work,
        # because it made the embeddings much bigger than the positional encodings (initially).
        # The positional encodings are necessary, but they probably shouldn't be as "loud" as the words themselves.

        # embedding matrix中的每个元素通常都很小， 这可能会导致在模型的前向传播过程中， 梯度小时或爆炸的问题
        # 为了缓解这个问题， 我们可以对embedding向量进行缩放， 即将其乘以一个缩放因子(math.sqrt(d_model))。
        # 在gpt模型中， 这个缩放因子是embedding向量的维度的平方根。这个缩放因子的作用是将embedding向量的范围调整到较大的范围，
        # 使得梯度变化更明显， 从而提高训练的稳定性和速度。同时，这种缩放方式不会改变embedding向量的方向， 因此在语义表示上并不会有影响。

        # 将单词嵌入表示乘以一个数值，这个数值是d_model的平方根，这样可以使得单词嵌入表示的值域范围更大，更有利于模型学习。
        x = self.embedding(x) * math.sqrt(self.d_model)

        return x


class PositionalEncoding(nn.Module):
    """
    seq_len is the lenth of the sequence

    the shape of embeddings is calculated in InputEmbedding class:
    (batch_size, seq_len, d_model)

    and the positional encoding matrix has the same shape
    (batch_size, max_seq_len, d_model)

    and seq_len <= max_seq_len
    so that's why we need only part of self.pe[:, :x.shape[1], :] if x is batch first
    otherwise we need only part of self.pe[:x.shape[0], :, :] if x is seq_len first
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1, batch_first: bool = True) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = max_seq_len
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        # step 1 pe
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):

            for i in range(0, d_model, 2):
                # 原公式 pos / (10000 ^ (2i / d_model)) = pos * (10000 ^ (-2i / d_model))
                pe[pos, i] = math.sin(pos / math.exp(i / d_model * math.log(10000.0)))
                pe[pos, (i + 1)] = math.cos(pos / math.exp(i / d_model * math.log(10000.0)))

        # step 2 Adding batch dimension
        if batch_first:

            pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        else:

            pe = pe.unsqueeze(0).transpose(0, 1)  # (seq_len, 1, d_model)

        # step 3 register_buffer
        # Register the positional encoding as a buffer
        # 增加一个属性， 并且是直接调用的属性， 不需要重新跑的属性
        self.register_buffer('pe', pe)

    def forward(self, x):

        if self.batch_first:

            # 因为self.pe 的shape是(batch_size, max_seq_len, d_model)

            # 而这里的x是前面说的 embeddings它的shape是(batch_size, seq_len, d_model)
            # x.shape[1] = seq_len

            # 而seq_len <= max_seq_len

            # 所以要根据x的shape 把positional_encoding的部分加上去， 因此有self.pe[:, :x.shape[1], :]

            # 这里使用了Variable函数将位置编码矩阵转换为一个不去计算梯度的张量（设置requires_grad = False），
            # 这样可以避免在模型训练中对位置编码矩阵进行梯度更新
            x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch_size, seq_len, d_model)

        else:

            # 因为如果 x的shape是(seq_len, batch_size d_model)
            # 那么self.pe 的shape就是(max_seq_len, batch_size d_model)
            # x.shape[0] = seq_len
            # 而seq_len <= max_seq_len
            # 所以要根据x的shape 把positional_encoding的部分加上去， 因此有self.pe[:x.shape[0], :, :]

            x = x + (self.pe[:x.shape[0], :, :]).requires_grad_(False)  # (seq_len, batch_size, d_model)

        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    """
    we do Multi Head Attention(splitting heads) along the embedding dimension, which is the d_model dimension.
    which means that each head we will have access to the full sentence
    but a different part of the embedding of each word
    因为这里是将d_model拆成8个头， 每个头对应的部分是d_k， 但是前面的batch_size, seq_len都没有改变，
    也就是说这里只有每个单词的embedding的维度发生了改变/发生了操作， 其他都不变， 因此整个句子都是能使用到的
    因此这里说we will have access to the full sentence
    """

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"
        # in the paper, it's assumed that d_k = d_v = d_model / h
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

        :param query: of shape (batch_size, num_heads, seq_len, d_k)
        :param key: of shape (batch_size, num_heads, seq_len, d_k)
        :param value: of shape (batch_size, num_heads, seq_len, d_v)
        :param mask:
        :param dropout: is an object of the Dropout Class
        :return: (attention_scores @ value), attention_scores
        """

        d_k = query.shape[-1]
        d_v = value.shape[-1]

        # 这里key.transpose(-2, -1)意思是保持了batch_size和heads_num的维度， 剩下的seq_len和depth=d_k的维度进行转置
        # Q @ K.T = (batch_size, num_heads, seq_len, d_k) * (batch_size, num_heads, seq_len, d_k).transpose(-2, -1)
        #         = (batch_size, num_heads, seq_len, d_k) * (batch_size, num_heads, d_k, seq_len)
        #         = (batch_size, num_heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # step 1 mask 是一个与 attention_scores shape相同的tensor
            # step 2 masked_filled_是在attention_scores上进行修改，
            # 因为前面用了"masked_filled_", 而不是"masked_filled"，所以是直接在 attention_scores上进行修改
            # step 3 masked_filled_的第一个参数是一个条件tensor。 这整个tensor的值只有True和False，
            # 当这个条件tensor为True的时候， 才会用这个函数当中的下一个参数的值， 进行填充
            # step 4 对attention_scores上对应mask为True的位置上的元素进行 指定值的填充
            # eg.mask == 1,指的是对位置上的值等于"1"的值， 使用"-1e9"进行替换， 把1 替换成-1e9

            ######
            # 这里要注意入参 mask是怎么来的， 如何定义的
            ######

            # 这里省略了encoder_padding_mask，但是最后的效果是一样的
            # 比如说算出来的attention_scores 是等于 q @ k.T / math.sqrt(d_k)
            # 然后再去对encoder_padding_mask增加维度， 并乘以-1e9之后再加回去attention_scores
            # 那么最后如果mask里面是有0的话（根据老弓的课是1）， 将其变成-1e9
            # 当将这个数加回去attention_scores后还是-1e9， 那还不如直接将attention_scores的对应位置设为-1e9

            # 这里又一个点要注意， 就是mask的shape需要和attention_scores的shape一致才可以

            attention_scores.masked_fill_(mask == 1, -1e9)
            # attention_scores.masked_fill_(mask == 0, -1e9)

        # attention_scores = attention_scores.softmax(dim=-1)
        # softmax(dim=-1)是为了对最后那个维度求softmax，batch_size * num_heads * seq_len * seq_len
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)

        if dropout is not None:
            # this dropout is nn.Dropout, is a layer, not a value
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
        # q @ self.w_q = (batch , seq_len, d_model) * (d_model, d_model) --> (batch , seq_len, d_model)
        query = self.w_q(q)

        # k @ self.w_k = (batch , seq_len, d_model) * (d_model, d_model) --> (batch , seq_len, d_model)
        key = self.w_k(k)

        # v @ self.w_v = (batch , seq_len, d_model) * (d_model, d_model) --> (batch , seq_len, d_model)
        value = self.w_v(v)
        # 上面的操作都和论文中的一致， 对输入做Q/K/V线性变换

        # 论文中不会做下面的"分头"， 就是下面的view和transpose操作
        # step 2 Multi Heads 先对字的vector分头
        # (batch , seq_len, d_model) --> (batch_size, seq_len, heads_num, depth=d_k)
        # 这个维度拆分是直接可以完成的， 因为只是最后一个维度的切分
        # we want to keep the first dimension which is the batch_size
        # and also we want to keep the second dimension which is the seq_len
        # 分头后的形状， 但还不是我们想要的形状， 我们需要的是batch_size, heads_num, seq_len, depth

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        # transpose
        # we want a shape of (batch, num_heads, seq_len, d_k) from (batch, seq_len, num_heads, d_k),
        # so we need to do transpose 因此多加一步transpose
        # and then transpose(1, 2) will get batch_size, heads_num, seq_len, depth

        # transpose完后的shape的理解是， 我门希望的到每个batch当中的每一个head， 都能包含"整句话"的"每一个字"的"一部分"信息
        # 所以有
        # (batch_size(每一个batch)， self.h(每一个头)， seq_len(整句话的每一个字)， self.d_k(每一个字的某一个头的信息))

        # the transpose means that each head we will have access to the full sentence(seq_len)
        # but a different part(d_k) of the embedding of each word, that is (num_heads * seq_len * d_k)
        query = query.transpose(1, 2)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)
        key = key.transpose(1, 2)

        value = value.view(value.shape[0], value.shape[1], self.h, self.d_v)
        value = value.transpose(1, 2)

        # 而是将step 1中的线性变换做不同的 8 次， 就是有 8套的不同的(Wqi, Wki, Wvi) i = 0, 1, 2...7
        # 并计算8 套不同的attention_scores
        # 这样理解是 从8 个方向去理解同一句原始的输入的句子，相当于提问8个问题来获取这个句子8个方面的信息
        # 这种解释在吴恩达的视屏当中解释的非常清楚
        # https://www.bilibili.com/video/BV1ev4y1U7j2?spm_id_from=333.788.videopod.episodes&vd_source=c7d149649e5042695dd51cb1f0324fa0&p=189

        # step 3 output, self.attention_scores
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        # 这里的x的是 attention_score @ V
        # shape是:
        # (batch, num_heads, seq_len, seq_len) * (batch, num_heads, seq_len, d_v) = (batch, num_heads, seq_len, d_v)

        # step 4 concat
        # 接头的时候，
        # 1)变回(batch, seq_len， num_heads, d_k)
        # 所以有 step 1 (batch, h, seq_len, d_v) --> (batch, seq_len, h, d_v)

        # 下面这几步只是分头后的反向操作， 来去获得分头前的形状
        # 原来操作：
        # 1先分头
        # 2再transpose
        # 现在逆向操作
        # 1先transpose
        # 2再合并

        # batch_size, heads_num, seq_len, d_k --> batch_size, seq_len, heads_num, d_k
        # 这一步操作只是变回分头后的形状
        x = x.transpose(1, 2)

        # 2) 作存储操作， 即保留先有的位置上元素信息， 通过contiguous() 实现
        # 对transpose后的x使用contiguous
        # 使用contiguous方法后返回新Tensor，重新开辟了一块内存，这个内存的位置与x就不会共享了，
        # 并使用照transpose后的x的按行优先一维展开的顺序存储底层数据。
        # 很明显这个新的 按行优先一维展开的顺序存储底层数据 与原来transpose之前的x 按行优先一维展开的顺序存储底层数据的不一样的
        # 假设一开始x = [[1, 2, 3],
        #              [4, 5, 6]]
        # 按行优先一维展开的顺序存储底层数据 [1, 2, 3, 4, 5, 6]
        # x.tranpose(0, 1) = [[1, 4],
        #                     [2, 5],
        #                     [3, 6]]
        # 按行优先一维展开的顺序存储底层数据 [1, 4, 2, 5, 3, 6]
        # 这两个按行优先一维展开的顺序存储底层数据是完全不一样的
        x = x.contiguous()

        # 3) view
        # 相对分头，这里做接头的操作
        x = x.view(x.shape[0], -1, self.d_model)
        # 上面可以直接写成 x = x.view(x.shape[0], x.shape[1], self.d_model)

        # step 5 linear transformation
        # (batch_size, seq_len, d_model) * (d_model, d_model) --> batch_size, seq_len, d_model
        output = self.w_o(x)

        return output


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.features = features
        self.eps = eps

        # nn.Parameter()是一个特殊的类，用于创建可训练的参数。
        # 这些参数在模型训练过程中会自动更新，并且具有自动计算梯度的能力。
        self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        # print(f" this is the shape of x {x.shape}")
        # print(f"this is the shape of self.alpha, {self.alpha.shape}")

        # 这里用keepdim的原因是， 按照dim=-1算完统计值后，就会把这个dimension去掉了
        # 因此要用keepdim=True把这个dimension保留
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)

        # eps is to prevent dividing by zero or when std is very small
        x = self.alpha * (x - mean) / (std + self.eps) + self.bias

        return x

    class ResidualConnection(nn.Module):

        def __init__(self, features: int, dropout: float = 0.1) -> None:
            super().__init__()

            self.features = features
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            """

            :param x:
            :param sublayer: is the previous layer, which is a layer object like class()
            :return:
            """
            #########
            # FIX：可以把sublayer直接写成previous layer
            # the sublayer is the previous layer
            # sublayer(x)相当于previous layer的output
            # 比方说这里previous layer的output可以写成MultiHeadAttentionBlock.forward(x, x, x, scr_mask)
            # 但这里又会出现一个问题， scr_mask 还没定义
            #########

            # this is the paper's order
            ######
            # 这里修改一下， 方便理解
            # sublayer作为class name被传入， 还不是一个instance， sublayer()才是一个instance
            # if isinstance(sublayer(), MultiHeadAttentionBlock):
            #
            #     output = sublayer(x, x, x, mask)

            # step1 sublayer 值得是multihead那一层
            output = sublayer(x)

            # step2 然后再做dropout
            output = self.dropout(output)

            # step3 最后做残差加回x
            x = x + output

            # step4 对residual net出来的结果做layer normalization
            x = self.norm(x)

            return x


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int = 512, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu') -> None:
        super().__init__()

        self.d_model = d_model

        # FFN 接受来自self-attention的输出作为输入（实际是过完residual+layerNorm之后的输出）
        # 并通过一个带有relu的激活函数的两层全联接网络对输入进行更加复杂的非线性变换。
        # 实验证明， 这一非线性变换会对模型最终的兴盛产生十分重要的影响
        # 因此增大FFN的neuron数目有利于提升最终的结果质量， 因此FFN的维度一般逼self-attention的层的维度更大
        # dim_feedforward=2048
        self.dim_feedforward = dim_feedforward

        self.linear_1 = nn.Linear(d_model, dim_feedforward)  # w1 and b1
        if activation == 'sigmoid':

            self.activation = nn.Sigmoid()

        elif activation == 'softmax':

            self.activation = nn.Softmax(dim=-1)

        else:

            self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, d_model)  # w2 and b2

    def forward(self, x):
        """

        :param x: of shape batch_size * seq_len * d_model
        :return:
        """

        # step 1 batch_size, seq_len, d_model --> batch_size, seq_len, dim_feedforward
        output = self.linear_1(x)

        output = self.activation(output)
        output = self.dropout(output)

        # step 2 batch_size, seq_len, dim_feedforward --> batch_size, seq_len, d_model
        output = self.linear_2(output)
        # return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

        return output


class TransformerEncoderLayer(nn.Module):

    # 这里只是一个encoder的block
    # 整个encoder的部分是有N个这样的EncoderBlock

    def __init__(self,
                 features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float = 0.1) -> None:
        # 这里的features在后面调用的时候都会用统一的一个值 d_model,
        # 所以在计算self_attention_block的LayerNormalization的时候
        # 或 feed_forward_block的LayerNormalization的时候 都用同一个features=d_model就可以了
        super().__init__()

        ######
        # 要注意后面是怎么调用的
        # 这里的入参是class， 不是object

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


# 参考原代码
# class EncoderBlock(nn.Module):
#
#     def __init__(self,
#                  self_attention_block: MultiHeadAttentionBlock,
#                  feed_forward_block: FeedForwardBlock,
#                  dropout: float = 0.1):
#
#         super().__init__()
#
#         ######
#         # 要注意后面是怎么调用的
#         # 这里的入参是class， 不是object
#         self.self_attention_block = self_attention_block
#         self.feed_forward_block = feed_forward_block
#
#         self.dropout = nn.Dropout(dropout)
#         self.norm = LayerNormalization()
#
#         # 要做两个residual_connections 所以这里是range 2
#         self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
#
#         # 与foward中对应， 是否需要加LayerNormalization
#         # self.layer_norms = nn.ModuleList(LayerNormalization() for _ in range(2))
#
#     def forward(self, x, src_mask):
#
#         """
#
#         :param x:
#         :param src_mask: : encoder_padding_mask
#         :return:
#         """
#
#         # MultiHead attension and residual
#         # 这里要用lambda x: self.self_attention_block(x, x, x, src_mask) 是因为
#         # self.residual_connections[0]在隐式调用forward时的入参是一个x和一个sublayer， sublayer是一个类
#         # 正常情况下的写法应该是
#         # self.residual_connections[0](x, self.self_attention_block)
#
#         # 但在self.residual_connections[0]的forward函数当中， 又是直接调用sublayer(x)的，
#         # 因此， 对于self.residual_connections[0]的forward函数，
#         # 实际上是需要用到sublayer.forward(x, x, x, src_mask)的结果，
#         # 所以这里彩这样写（具体为什么这个能work不太清楚）
#         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
#
#         ######
#         # lambda x: self.self_attention_block(x, x, x, src_mask) 这一段不好理解， 我们可以直接在这里重写这一层的residual_connect
#         # x = x + self.dropout(self.norm(self.self.self_attention_block(x, x, x, src_mask)))
#
#         ######
#         # 这里是不是应该加一个LayerNormalization层？
#         # x = self.layer_norms[0](x)
#
#         # Feed forward
#         x = self.residual_connections[1](x, lambda x: self.feed_forward_block)
#
#         ######
#         # 这里是不是应该加一个LayerNormalization层？
#         # x = self.layer_norms[1](x)
#
#         return x


class TransformerEncoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, src_mask):

        for layer in self.layers:

            x = layer(x, src_mask)

        x = self.norm(x)

        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 features: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float = 0.1):
        super().__init__()

        # 这里的features在后面调用的时候都会用统一的一个值 d_model,
        # 所以在计算self_attention_block的LayerNormalization的时候
        # 或 cross_attention_block的LayerNormalization的时候 都用同一个d_model就可以了
        # 或 feed_forward_block的LayerNormalization的时候 都用同一个features=d_model就可以了

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
        # 这里第一步要做的是对decoder做self attention， 这里不会涉及到encoder的输出，
        # 是tgt与自己的attention计算， 所以这里使用的全是tgt， 以及使用tgt_mask
        # 这里的x 就是tgt
        output_self = self.self_attention_block(x, x, x, tgt_mask)
        output_self = self.dropout_self(output_self)
        x = x + output_self
        x = self.norm_self(x)
        # x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,src_mask))
        # 这里第二步要做的是tgt与 encoder_output的交互， encoder_output既作为key传入， 也作为value传入
        # 这里的x 就是tgt经过上一个self_attention_block输出的结果
        # 而encoder_output是经过TransformerEncoder（N= 6个TransformerEncoderLayer作用之后的输出）
        # 然后把x=tgt 以及encoder_output 传入cross_attention_block
        # 这里会对x=tgt 以及encoder_output分别做linear transformation， 得到对应的q k和v
        # 可以理解为q 是tgt经过线性变换w_q作用之后的结果
        # 可以理解为k和v 是encoder_output分别经过线性变换w_k与w_v作用之后的结果

        # 这里要重点讲一下mask， 这里传入的是src_mask， 而不是tgt_mask
        # 原因如下：
        # 1。我们现在是在做tgt 与 src的交互， 我门希望用由src中的encoder_output所生成的k 与 由tgt所生成的q计算出attention_scores
        # 2 然后用由src中的encoder_output所生成的v进行1中attention_scores的加权来表达这个"由tgt所生成的q"
        # 我们不希望用到v中作为padding的，没有意义的vi去表达这里的q，
        # 因此，我们需要把由encoder_output所生成的某些vi给mask掉；
        # 所以， 对应encoder_output， 我们需要用encoder_mask 即src_mask而不是tgt_mask
        # tgt_mask只能mask掉tgt的部份vectors， 而不能mask掉encoder_output的部份vectors

        # 所以这就是为什么要使用 src_mask

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


class TransformerDecoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        for layer in self.layers:
            # 因为要调用TransformerDecoderLayer的forward函数， 所以这里的inputs与TransformerDecoderLayer的forward的inputs完全一致，
            x = layer(x, encoder_output, src_mask, tgt_mask)

        x = self.norm(x)

        return x


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, tgt_vocab_size: int) -> None:
        super().__init__()

        self.d_model = d_model
        # 这里的tgt_vocab_size指的是target_vocab_size， 后面可以改名字
        self.tgt_vocab_size = tgt_vocab_size

        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, tgt_vocab_size)
        x = self.proj(x)

        # final result we use the log_softmax
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
        # input:(batch, seq_len)
        # output size: (batch, seq_len, d_model)

        # the shape must be batch_size * seq_len, NOT batch_size * seq_len * src_vocab_size
        # NOTE：这里传入的是一个句子长度对应的每个词的token的列表， 并不是根据这个列表转化的one-hot矩阵

        # step 1 input embedding
        src = self.src_embed(src)

        # step 2 input positional encoding
        src = self.src_pos(src)

        # step 3 encoder
        # 这里是调用了TransformerEncoder Class里面的forward函数
        # 进而调用了TransformerEncoderLayer Class里面的forward函数
        src = self.encoder(src, src_mask)

        return src

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # input:(batch, seq_len)
        # output size: (batch, seq_len, d_model)

        # the shape must be batch_size * seq_len, NOT batch_size * seq_len * tgt_vocab_size
        # NOTE：这里传入的是一个句子长度对应的每个词的token的列表， 并不是根据这个列表转化的one-hot矩阵
        tgt = self.tgt_embed(tgt)

        # step 2 input positional encoding
        tgt = self.tgt_pos(tgt)

        # step 3 decoder
        # 这里是调用了TransformerDecoder Class里面的forward函数
        # 进而调用了TransformerDecoderLayer Class里面的forward函数
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

    # step 1 create the embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # step 2 Positional encoding layer
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

    # Step 3 encoder
    encoder = TransformerEncoder(d_model,
                                 nn.ModuleList(
                                     [
                                         TransformerEncoderLayer(d_model,
                                                                 MultiHeadAttentionBlock(d_model, h, dropout),
                                                                 FeedForwardBlock(d_model, d_ff, dropout),
                                                                 dropout) for _ in range(N)
                                     ]
                                 )
                                 )

    # step 4 decoder
    decoder = TransformerDecoder(d_model,
                                 nn.ModuleList(
                                     [
                                         TransformerDecoderLayer(d_model,
                                                                 MultiHeadAttentionBlock(d_model, h, dropout),
                                                                 MultiHeadAttentionBlock(d_model, h, dropout),
                                                                 FeedForwardBlock(d_model, d_ff, dropout),
                                                                 dropout) for _ in range(N)
                                     ]
                                 )
                                 )

    # step 5 Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # step 6 Create the transformer
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



