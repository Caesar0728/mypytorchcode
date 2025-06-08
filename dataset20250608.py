import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    """
    since this class is inherited from the Dataset Class, so we have to define __len__ and __getitem__ methods
    # torch.utils.data.Dataset 是 PyTorch 提供的一个抽象类，
    # 用户需要继承这个类并实现 __len__() 和 __getitem__() 方法，以定义自己的数据集。
    # 这个类通常用于 自定义数据集的加载和处理，广泛适用于各种深度学习任务。
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # 下面还要定义句首字符SOS的token和句尾字符EOS的token，及padding项的token
        # '[SOS]', '[EOS]', '[PAD]'可以只用tokenizer_src里面的，
        # 因为在tgt里面， 出现这三个的话， 他们的顺序也会是一样的， 对应的token_id也是一样的
        # 当然最好用两套， src_sos_token， src_eos_token，src_pad_token，
        # tgt_sos_token， tgt_eos_token，tgt_pad_token，

        #######
        # 先跑成功了后面再加两套
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

        ######
        # 源代码里面只用了一个self.seq_len， 貌似是假定src和tgt当中的seq_len是一样的，除非这个seq_len指的是max_seq_len
        # 否则个人感觉还是要区分src_seq_len与tgt_seq_len

        # 使用tokenizer.token_to_id()函数时，你需要提供一个token作为输入，然后它会返回该token对应的ID。
        # 如果输入的token不在tokenizer的词汇表中，该函数可能会返回一个特殊的ID（如[UNK]未知词的ID）或抛出一个错误，
        # 这取决于tokenizer的具体实现。

        # the indexes of tgt_sos_token, tgt_eos_token, tgt_pad_token
        # are the same as
        # sos_token, eos_token, pad_token accordingly, so we don't have to define them again
        # but for completeness, we define them
        self.tgt_sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.tgt_eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.tgt_pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # step 1 input target pair
        src_target_pair = self.ds[index]
        # 这里self.ds[index]是一个字典， {id:'1', 'translation': {'en': '...', 'it': '...'}}

        # step 2 read the input adn read the target
        # 下面把src_text句子通过tokenizer拆分成token，然后在映射成token_id；
        # 同理，把tgt_text句子通过tokenizer拆分成token，然后在映射成token_id；
        src_text = src_target_pair['translation'][self.src_lang]
        # encoding = tokenizer.encode("hi, this is chen shen")
        # print(encoding.tokens)  # 输出 ['hi', ',', 'this', 'is', 'chen', 'shen']
        # print(encoding.ids)     # 输出对应的词汇表 ID [12, 4, 8, 13, 11, 15]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # 这里会返回src_text当中， 每一个字词对应的在字典里头的id， 这里会返回一个array
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # padding each sentence, -2 means sos and eos tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # - 2 是把 SOS和EOS减去

        # We will only add <sos>, and <eos> only on the label， 只会加一项
        # 这里就是相当于把 sos i like you eos 拆成sos i like you 和 i like you eos之后， 其中一个句子的长度
        # 一个叫做target_input一个叫做target_real
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        # - 1 是把 SOS减去， 训练的时候是不会带有EOS的， 训练中句子长度，是整个句子长度-1， 减去最后一位

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # 下面要做的是encoder_input， decoder_input， decoder_target拼接出来
        # 下面[]里面的每一个元素都是单维度的tensor， 按dim=0维度进行concat， 会得到一个'列tensor'， 然后再进行 维度增加

        # Add <s> and </s> token
        # 把一个句子的tokens全部拼接起来
        # sos_token
        # enc_input_tokens
        # eos_token
        # padding_tokens 有可能有的padding项

        # encoder_input torch.Size([seq_len])
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ], dim=0
        )

        # decoder的input 是没有eos_token的

        # Add only <s> token
        # 这部分相当于视频里的target_input
        # sos
        # dec_input_tokens
        # padding_tokens 有可能有的padding项

        # 这里的padding方式没错
        # sos
        # dec_input_tokens
        # 到上面这一步都不会产生疑问
        # target inputs
        # 1 sos 我喜欢你pad -- 我 喜 欢 你 eos pad
        # 2 sos 我要去旅游     我 要 去 旅 游   eos

        # decoder_input torch.Size([seq_len])
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ], dim=0
        )

        # decoder的label是没有sos_token的

        # Add only </s> token
        # 这部分相当于是我们视频里的target_real
        # dec_input_tokens
        # padding_tokens 有可能有的padding项
        # eos

        # label的shape为 torch.Size([seq_len])
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ], dim=0
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input == self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input == self.pad_token).unsqueeze(0).unsqueeze(0).int() | look_ahead_mask(
                decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def look_ahead_mask(size):
    # 因为这个causalmask要和tgt_mask取并集， 并加回去query @ keky.Transpose(1, 2) 这个整体上面
    # 这个整体的shape 是 batch_size * seq_len * seq_len,
    # 因此我们这里要先把mask的shape定义成1 * seq_len * seq_len,

    # diagonal=1这个参数是指不保留diagonal=1的值
    # diagonal=2是指不保留diagonal轴上再往上数一条斜对角线上的值， 实在不理解可以自行切换print出来看看
    # 这里的意思就是， 构建一个(1, size, size)的1矩阵， 然后把对角线及以下的元素全部变为0

    #  构造一个三维的单位tensor， 然后将这个tensor的最后两个维度形成的矩阵变成上三角矩阵， 即对角线以上的数据都保留
    #  diagonal是把对角线及以下的数也抹去变成0

    ###### 这里的causal_mask就是look_ahead_padding_mask ######
    # 这里的代码
    # (decoder_input != self.pad_token).unsqueeze(0).int() | causal_mask(decoder_input.size(0))
    # 按笔记中的理解是取并集的， 即用 ｜，
    # 具体怎么写 可以参考tokenizers_lib解释如何使用tokenizers库.py的主注释部分 ########超级有用#########
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)

    # mask == 1 是把这个tensor到过来， 不等于1 的位置就变成0， 等于1的位置就变成1
    # tensor([[[0., 1., 1.],
    #          [0., 0., 1.],
    #          [0., 0., 0.]],
    #
    #         [[0., 1., 1.],
    #          [0., 0., 1.],
    #          [0., 0., 0.]]])

    # tensor([[[ False, True, True],
    #          [ False, False, True],
    #          [ False,  False,  False]],
    #
    #         [[ False, True, True],
    #          [ False,  False, True],
    #          [ False,  False,  False]]])
    return mask == 1
    # 这里就会形成一个True/False， 由于mask矩阵对角线及以下的元素全为0，
    # 所以mask == 1就会使得这个新矩阵对角线及以下的元素全为False， 其余的全为Ture
