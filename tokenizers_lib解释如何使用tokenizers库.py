# step 1
from tokenizers import Tokenizer  # 导入Tokenizer类

from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents, NFC
# normalizers模块通过执行诸如转换为小写、移除特殊字符、处理缩写、消除重音符号等操作，对文本进行规范化处理
# 1.NFD 将字符拆解 “â” (U+00E2) -> “a” (U+0061) + “̂” (U+0302) 即箭头左边的â被拆分成“a” (U+0061)和“̂” (U+0302)， 变得到需要的a
# 2.Lowercase规范器负责将文本中所有的大写字母转换成小写字母。
# 3.StripAccents规范器负责去掉所有的重音，在使用StripAccents规范器之前，需要使用NFD规范器以保持一致性。
# 一下面这个"Café Noël"为例，
# 通过NFD()后就得到Cafe* Noe!l， 这里由于没法表示两个e头顶上的特殊Accents符号，就用*和!表示，
# 然后再通过StripAccents()就会把这两个特殊Accents去掉， 得到Cafe Noel
# 然后再通过Lowercase()就会把大写字母转成小写字母 得到cafe noel
# 然后再通过NFC()组合回去变成原来NFD()之前的格式，虽然转回去了， 但中间的特殊Accents符号被去掉了，大写也变成小写了。
# normalizers就是做这样一个处理的过程
input_text = "Café Noël"
# normalizer = normalizers.StripAccents()
normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase(), NFC()])
normalized_text = normalizer.normalize_str(input_text)

print("原始文本:", input_text)
print("规范化后的文本:", normalized_text)

from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit, Punctuation, WhitespaceSplit
# 在 tokenizers 库中，pre_tokenizers 用于对文本进行预分词处理，
# 即在实际进行分词之前，先按照某种规则对文本进行初步的分割。
# 它的作用是将文本划分为基本的子词或子符号，方便后续的编码器进行更细粒度的分词操作。
# 1.Whitespace()按标点符号和空格作为分隔符， 对句子进行切分
# 2.Punctuation()只按标点符号作为分隔符， 对句子进行切分
# 3.WhitespaceSplit()只按空格作为分隔符， 对句子进行切分
# 4.CharDelimiterSplit()按字符作为分隔符， 对句子进行切分， 一个字符为单位， 是字符！
pre_tokenizer = Whitespace()
pre_tokenizer_normalized_text = pre_tokenizer.pre_tokenize_str(normalized_text)
print("原始文本:", normalized_text)
print("规范化后的文本:", pre_tokenizer_normalized_text)

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

# 目标：我们需要构建一个分词器Tokenizer()， 这个分词器是根据某一个分词模型BPE() 通过训练BpeTrainer， 得到的。
# 1.tokenizers import Tokenizer, Tokenizer()是一个空的未被定义未被训练的分词器；
# 2.tokenizers.models import BPE, 训练一个分词器，可以从 tokenizers.models 模块中选择一种分词模型， 例如BPE
# 3.tokenizers.trainers import BpeTrainer 因为我们选了BPE类型的分词模型， 那么对应的我们需要用BpeTrainer来训练BPE分词模型
# 4.如果在models中选择了WordLevel， 那么对应的在trainers当中我门就会选择WordLevelTrainer

# 创建 WordLevel 模型的 tokenizer
# step 1 初始化分词器
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

# 设置预分词器, 根据标点符号和空格先将样本中的句子进行拆分， 上面有解释如何拆分
tokenizer.pre_tokenizer = Whitespace()

# 构造训练器
# step 3 初始化一个 WordLevelTrainer
# WordLevelTrainer() 是 Hugging Face tokenizers 库中的一个类，用于训练 WordLevel 模型。
# WordLevel 是基于词汇表（vocabulary）的分词器，直接将每个词映射为词汇表中的一个索引。
# WordLevelTrainer 用于从大量的文本数据中生成这个词汇表。

# substep1
# 限定词汇表大小为10000， 也就是新的一个词袋中单词数量不能超过10000

# substep2
# min_frequency=1， 就是训练的这些句子当中， 词出现的频率要大于等于1
# 由于这里的样本很少， 暂时设置为1，
# 正常情况下我们有大量的句子的时候， 这个min_frequency可以适当的调高， 比如说min_frequency=2
# 这样可以减少生僻字进入词袋

# substep3
# 自定义特殊标记 special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
# 注意这里的特殊标记 [UNK] 和 UNK是不一样的， 具体怎么定义要看实际情况
trainer = WordLevelTrainer(
    vocab_size=10000,  # 限定词汇表大小为10000
    min_frequency=1,  # 词频至少为2的词语才会被加入词汇表
    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]  # 自定义特殊标记
)
# vocab_size 是用于指定分词器词汇表的大小，即生成的词汇表中可以包含的 最大词汇数量。
# vocab_size=10000 表示分词器的词汇表最多包含 10,000 个不同的 token（子词单元或字符组合）。
# 如果训练样本当中的句子中， 所有不一样的词的数量超过10000，那么训练器会选择频率最高的前 10,000 个 token 构建词汇表

# min_frequency (int, optional):
# 指定词语出现在数据中最小的频率。
# 如果某个词的频率低于 min_frequency，它将不会被包括在词汇表中。
# 这可以帮助过滤掉一些罕见的词，防止词汇表过于庞大。

# special_tokens (List[str], optional):
# 特殊标记的列表，如 ""[UNK]", "[PAD]", "[SOS]", "[EOS]"", 这些标记在处理文本任务时往往是必须的。
# 可以自定义哪些特殊标记应包括在词汇表中。


# 训练文本数据
texts_old = ["Café Noël",
             "hi Chen Shen.",
             "this is caesar, how are you?",
             "I love programming.",
             "how old are you this year?",
             "Tokenizers are awesome!"]

# 对每一个样本作 normalizers处理
normalized_texts = []
for text in texts_old:
    normalized_text = normalizer.normalize_str(text)
    normalized_texts.append(normalized_text)
print(normalized_texts)

# 在每个句子前加上 [SOS]，后面加上 [EOS]

with_special_tokens_texts = [f'[SOS] {text} [EOS]' for text in normalized_texts]
print(with_special_tokens_texts)

tokenizer.train_from_iterator(with_special_tokens_texts, trainer)
print("######")
print(tokenizer.get_vocab())
encoded_batch_1 = tokenizer.encode_batch(['[UNK]', '[PAD]', '[SOS]', '[EOS]'])

for encoded in encoded_batch_1:
    print("Tokens:", encoded.tokens)
    print("Token IDs:", encoded.ids)

encoded_batch_2 = tokenizer.encode_batch(with_special_tokens_texts)

for encoded in encoded_batch_2:
    print("Tokens:", encoded.tokens)
    print("Token IDs:", encoded.ids)

# step 6 测试分词器
encoding = tokenizer.encode("hi, this is chen shen")
print(encoding.tokens)  # 输出 ['hi', ',', 'this', 'is', 'chen', 'shen']
print(encoding.ids)     # 输出对应的词汇表 ID [16, 11, 8, 18, 15, 23]
# 训练的时候会根据特殊标记和样本句子出现了的单词进行排序，从而每个单词包括特殊标记都有他们自己对应的id
# 从结果来开， 是先特殊标记然后再对符号， 再对单词这样的排序


# 对文本进行编码
encoded_texts = [tokenizer.encode(text) for text in with_special_tokens_texts]

# 找到最长的句子长度
max_len = max(len(encoded.ids) for encoded in encoded_texts)

# 填充到最长长度
padded_encoded_texts = []
for encoded in encoded_texts:

    # 对每个句子进行不同长度的填充，
    padding_length = max_len - len(encoded.ids)

    # 填充 [PAD] 的 token id
    padded_ids = encoded.ids + [tokenizer.token_to_id("[PAD]")] * padding_length
    padded_tokens = encoded.tokens + ["[PAD]"] * padding_length

    padded_encoded_texts.append({"ids": padded_ids, "tokens": padded_tokens})

# 查看填充后的结果
for i, encoded in enumerate(padded_encoded_texts):
    print(f"Sentence {i+1}:")
    print("Token IDs:", encoded["ids"])
    print("Tokens:", encoded["tokens"])







# step 2 按词为单位划分句子
from tokenizers.models import WordLevel
# WordLevel(vocab, unk_token="UNK")
# 这个方法有两个参数， 一个是vocab 一个是unk_token，
# vocab是指整个单词库的字典，就是整个词袋， 其形式如 单词：id，
# 这个词袋里面需要包含"UNK" 或者unk_token 指定的那个词，
# 比如说我指定orange作为unk_token， 那么在一个新的句子中， 如果出现了一个词袋里面没有的单词，
# 那么就会把这个单词替换成unk_token指定的单词，"UNK"或者是指定的"orange"，
# 并且它的对应的id就是这个"UNK"对应的ID， 或者是orange对应的恶id

# step 3 按空格作为分割符
# 如果不加这个pre_tokenizer， 就是不拿空格作为分隔符， 那么输入的整个句子就会被视作一个单词， 出来的必定是UNK
from tokenizers.pre_tokenizers import Whitespace

# step 4 定义一个trainer
from tokenizers.trainers import WordLevelTrainer
# WordLevelTrainer() 是 Hugging Face tokenizers 库中的一个类，用于训练 WordLevel 模型。
# WordLevel 是基于词汇表（vocabulary）的分词器，直接将每个词映射为词汇表中的一个索引。
# WordLevelTrainer 用于从大量的文本数据中生成这个词汇表。

# 参数详解
# WordLevelTrainer() 的构造函数接受以下参数：
#
# vocab_size (int, optional):
# 用于指定词汇表的大小。默认情况下，WordLevelTrainer 将尝试从数据中提取所有唯一词语。
# 如果设置了 vocab_size，则会选择最常用的词语，构建一个大小为 vocab_size 的词汇表。

# min_frequency (int, optional):
# 指定词语出现在数据中最小的频率。
# 如果某个词的频率低于 min_frequency，它将不会被包括在词汇表中。
# 这可以帮助过滤掉一些罕见的词，防止词汇表过于庞大。

# special_tokens (List[str], optional):
# 特殊标记的列表，如 ""[UNK]", "[PAD]", "[SOS]", "[EOS]"", 这些标记在处理文本任务时往往是必须的。
# 可以自定义哪些特殊标记应包括在词汇表中。

#
# # 首先， 要有一个简单的词汇表
vocab = {"hello": 0,
         "world": 1,
         "[UNK]": 2,
         "I": 3,
         "like": 4,
         "an": 5,
         "apple": 6,
         "[PAD]": 7,
         "[SOS]": 8,
         "[EOS]": 9}

# 然后， 创建一个 WordLevel 分词器
# 这里指定unk_token是"[UNK]"这个词
tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
# 用空格作为分隔符
tokenizer.pre_tokenizer = Whitespace()

# encoding是一个类， 有tokens和ids等属性， 我们只要知道它划分了哪些tokens， 和这些tokens对应的ids
encoding = tokenizer.encode("I like orange")
print("#############333#####")
print(tokenizer.get_vocab_size())
print(tokenizer.get_vocab())
print(type(encoding))  # encoding是一个类，
print(encoding.tokens)  # ['I', 'like', 'UNK'] 我们只要知道它划分了哪些tokens，
print(encoding.ids)     # [3, 4, 2] 这些tokens对应的ids, 这里由于句子中orange不在词袋vocab中， 所以被当成是unk_token，
# "I like orange"这句话就相当于"I like UNK"，
# 那么新的这句话对应的tokens就是['I', 'like', 'UNK']， 对应的ids[3, 4, 2]



########超级有用#########
# #############
# print("Training Part")
# # 上面部分只是直接划分token和找出词袋中token对应的id
#
#
# import torch
# import torch.nn as nn
# import tokenizers
# from tokenizers import Tokenizer
# from tokenizers.models import WordLevel
# from tokenizers.trainers import WordLevelTrainer
# from tokenizers.pre_tokenizers import Whitespace
# # 下面是根据实际句子样本，来训练一个划分token的方式
#
# # step 1 初始化分词器
# tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
#
# # step 2 如果不加这个pre_tokenizer， 就是不拿空格作为分隔符， 那么输入的整个句子就会被视作一个单词， 出来的必定是UNK
# tokenizer.pre_tokenizer = Whitespace()
#
# # step 3 初始化一个 WordLevelTrainer
# # substep1
# # 限定词汇表大小为10000， 也就是新的一个词袋中单词数量不能超过10000
#
# # substep2
# # min_frequency=1， 就是训练的这些句子当中， 词出现的频率要大于等于1
# # 由于这里的样本很少， 暂时设置为1，
# # 正常情况下我们有大量的句子的时候， 这个min_frequency可以适当的调高， 比如说min_frequency=2
# # 这样可以减少生僻字进入词袋
#
# # substep3
# # 自定义特殊标记 special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
# # 注意这里的特殊标记 [UNK] 和 UNK是不一样的， 具体怎么定义要看实际情况
# trainer = WordLevelTrainer(
#     vocab_size=10000,  # 限定词汇表大小为10000
#     min_frequency=1,  # 词频至少为2的词语才会被加入词汇表
#     special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]  # 自定义特殊标记
# )
#
# # step 4 准备训练数据生成器
# # 一般情况下样本库都是一个句子接一个句子这样的， 所以要一个接一个的去训练
# # 那么这个时候就要生成一个生成器， 不用return用yield
# def data_generator():
#     yield "hi chen shen,"
#     yield "this is caesar, how are you?"
#     yield "how old are you this year?"
#
# # step 5 训练分词器， length这个参数感觉没什么用
# tokenizer.train_from_iterator(data_generator(), trainer, length=1)
#
# # step 6 测试分词器
# encoding = tokenizer.encode("hi, this is chen shen")
# print(encoding.tokens)  # 输出 ['hi', ',', 'this', 'is', 'chen', 'shen']
# print(encoding.ids)     # 输出对应的词汇表 ID [12, 4, 8, 13, 11, 15]
# # 训练的时候会根据特殊标记和样本句子出现了的单词进行排序，从而每个单词包括特殊标记都有他们自己对应的id
# # 从结果来开， 是先特殊标记然后再对符号， 再对单词这样的排序
# sos_token = torch.tensor([1], dtype=torch.int64)
# print(torch.tensor(encoding.ids))
# encode_input = torch.cat(
#     [
#         sos_token,
#         torch.tensor(encoding.ids)
#     ], dim=0)
# print(encode_input)
# print(encode_input.shape)
# print(encode_input.unsqueeze(0).unsqueeze(0).shape)  # (1, 1, 7)
#
# # causal_mask = torch.tril(torch.ones(1, 7, 7))
# # 保留对角线以上的元素（不包含对角线）， 其余都为0
# look_ahead_padding_mask = torch.triu(torch.ones(1, 7, 7), diagonal=1)
# # 这个causal_mask 就是笔记里面的look_ahead_padding_mask
# print(look_ahead_padding_mask)
#
# # 然后把这个look_ahead_padding_mask转换成True/False 为元素的causal_mask
# causal_mask = (look_ahead_padding_mask == 1)
# print(causal_mask)
#
# # 假设pad_token就是15
# pad_token = 15
# # encoder_decoder_padding_mask 与 encoder_padding_mask一致
# # 而encoder_padding_mask是通过encode_input == pad_token 得到的（找出encode_input哪一个位置是pad_token）
# encoder_decoder_padding_mask = (encode_input == pad_token)
# # 增加一个batch_size的维度
# encoder_decoder_padding_mask = encoder_decoder_padding_mask.unsqueeze(0)
# print(encoder_decoder_padding_mask)
# print(encoder_decoder_padding_mask | causal_mask)

########超级有用#########




# from tokenizers import Tokenizer
# from tokenizers.models import WordLevel
# from tokenizers.trainers import WordLevelTrainer
# from tokenizers.pre_tokenizers import Whitespace
#
# # 初始化分词器
# tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
#
# # 指定特殊 token
# special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]
# trainer = WordLevelTrainer(vocab_size=30000, special_tokens=special_tokens)
#
# # 使用 Whitespace 预分词器
# tokenizer.pre_tokenizer = Whitespace()
#
# # 训练数据，包含 [CLS] 和 [SEP]
# texts = ['[CLS] Hello world [SEP]', '[CLS] I love programming. [SEP]', '[CLS] Tokenizers are awesome! [SEP]']
#
# # 训练 tokenizer
# tokenizer.train_from_iterator(texts, trainer)
#
# # 对文本进行编码
# encoded_texts = [tokenizer.encode(text) for text in texts]
#
# # 找到最长的句子长度
# max_len = max(len(encoded.ids) for encoded in encoded_texts)
#
# # 填充到最长长度
# padded_encoded_texts = []
# for encoded in encoded_texts:
#     padding_length = max_len - len(encoded.ids)
#     # 填充 [PAD] 的 token id
#     padded_ids = encoded.ids + [tokenizer.token_to_id("[PAD]")] * padding_length
#     padded_tokens = encoded.tokens + ["[PAD]"] * padding_length
#     padded_encoded_texts.append({"ids": padded_ids, "tokens": padded_tokens})
#
# # 查看填充后的结果
# for i, encoded in enumerate(padded_encoded_texts):
#     print(f"Sentence {i+1}:")
#     print("Token IDs:", encoded["ids"])
#     print("Tokens:", encoded["tokens"])

############################################
############################################
############################################
#                下面是实操
############################################
############################################
############################################

import torch
from tokenizers import Tokenizer

from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents, NFC

from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit, Punctuation, WhitespaceSplit

from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase(), NFC()])
tokenizer.pre_tokenizer = Whitespace()
trainer = WordLevelTrainer(vocab_size=10000, min_frequency=1, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
# vocab_size=10000 限定词汇表大小为10000
# min_frequency=1 词频至少为2的词语才会被加入词汇表
# special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"] 自定义特殊标记


# 训练文本数据
texts_old = ["Café Noël",
             "hi Chen Shen.",
             "this is caesar, how are you?",
             "I love programming.",
             "how old are you this year?",
             "Tokenizers are awesome!"]


def get_all_sentences(texts):

    for item in texts:

        yield item


# Normalize the input texts
normalized_texts = [tokenizer.normalizer.normalize_str(text) for text in texts_old]
print(normalized_texts)

# train the tokenizer
tokenizer.train_from_iterator(get_all_sentences(normalized_texts), trainer)

print("######")
print(tokenizer.get_vocab())

# 要在train完tokenizer之后， 再在每个句子前加上 [SOS]，后面加上 [EOS]
with_special_tokens_texts = [f'[SOS] {text} [EOS]' for text in normalized_texts]
print(with_special_tokens_texts)

# step 6 测试分词器
encoding = tokenizer.encode("[SOS] hi, this is chen shen [EOS]")
print(encoding.tokens)  # 输出 ['[SOS]', 'hi', ',', 'this', 'is', 'chen', 'shen', '[EOS]']
print(encoding.ids)     # 输出对应的词汇表 ID [2, 16, 11, 8, 18, 15, 23, 3]
# 训练的时候会根据特殊标记和样本句子出现了的单词进行排序，从而每个单词包括特殊标记都有他们自己对应的id
# 从结果来开， 是先特殊标记然后再对符号， 再对单词这样的排序


# 对文本进行编码
encoded_texts = [tokenizer.encode(text) for text in with_special_tokens_texts]
print(f'encoded_texts {encoded_texts}')

# 找到最长的句子长度
max_len = max(len(encoded.ids) for encoded in encoded_texts)

# 填充到最长长度
padded_encoded_texts = []
for encoded in encoded_texts:

    # 对每个句子进行不同长度的填充，
    padding_length = max_len - len(encoded.ids)

    # 填充 [PAD] 的 token id
    padded_ids = encoded.ids + [tokenizer.token_to_id("[PAD]")] * padding_length
    padded_tokens = encoded.tokens + ["[PAD]"] * padding_length

    padded_encoded_texts.append({"ids": padded_ids, "tokens": padded_tokens})

# 查看填充后的结果
for i, encoded in enumerate(padded_encoded_texts):
    print(f"Sentence {i + 1}:")
    print("Token IDs:", encoded["ids"])
    print("Tokens:", encoded["tokens"])



# import torch
# import torch.nn as nn
#
# import tokenizers
# from tokenizers import Tokenizer
# from tokenizers import normalizers
# from tokenizers.normalizers import NFD, StripAccents, Lowercase, NFC
# from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.models import WordLevel
# from tokenizers.trainers import WordLevelTrainer
#
#
# tokenizer = Tokenizer(WordLevel(unk_token="[UNK"))
# tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase(), NFC()])
# tokenizer.pre_tokenizer = Whitespace()
# trainer = WordLevelTrainer(vocab_size=10000, min_frequency=1, special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"])
#
#
# def get_all_sentences(texts):
#
#     for item in texts:
#
#         yield item
#
#
# # 训练文本数据
# texts_old = ["Café Noël",
#              "hi Chen Shen.",
#              "this is caesar, how are you?",
#              "I love programming.",
#              "how old are you this year?",
#              "Tokenizers are awesome!"]
#
# normalized_texts = [tokenizer.normalizer.normalize_str(text) for text in texts_old]
# print(normalized_texts)
#
# tokenizer.train_from_iterator(get_all_sentences(normalized_texts), trainer)
# print(tokenizer.get_vocab())
#
# with_special_tokens_texts = [f'[SOS] {text} [EOS]' for text in normalized_texts]
# print(with_special_tokens_texts)
#
#
# print(tokenizer.encode(with_special_tokens_texts[0]).tokens)
# print(tokenizer.encode(with_special_tokens_texts[0]).ids)
#
# # 对文本进行编码
# encoded_texts = [tokenizer.encode(text) for text in with_special_tokens_texts]
# print(f'encoded_texts {encoded_texts}')
#
# max_seq_len = max(len(encoded_text) for encoded_text in encoded_texts)
# print(max_seq_len)
#
# padded_encoded_texts = []
#
# for encoded_text in encoded_texts:
#
#     seq_len = len(encoded_text.ids)
#
#     padding_len = max_seq_len - seq_len
#
#     padded_text_ids = encoded_text.ids + [tokenizer.token_to_id("[PAD]")] * padding_len
#     padded_text_tokens = encoded_text.tokens + ["[PAD]"] * padding_len
#
#     padded_encoded_texts.append({"ids": padded_text_ids, "tokens": padded_text_tokens})
#
# for i, padded_text in enumerate(padded_encoded_texts):
#
#     print(f"padding tokens: {padded_text['tokens']}")
#     print(f"padding ids: {padded_text['ids']}")





