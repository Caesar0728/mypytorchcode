import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, look_ahead_mask
from configuration import get_config, get_weights_file_path, latest_weights_file_path
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

from pathlib import Path


def get_all_sentences_old(dataset, language):

    # 在train_from_iterator中需要用iterator因此用了yield
    for item in dataset:
        # 因为这里的dataset是一个字典， 这个字典有两个keys， 一个是id， 另一个是translation
        # 而translation这个key对应的value还是一个字典， 这个字典又包含两个keys
        # 一个key是一门语言， 如 'en'； 另外一个key也是一门语言， 如'it'
        # 因此这里还要再套一个[language]的key才能得到，这门语言的句子

        yield item['translation'][language]


def get_all_sentences(dataset, language):
    for item in dataset:
        # 因为这里的dataset是一个字典， 这个字典有两个keys， 一个是id， 另一个是translation
        # 而translation这个key对应的value还是一个字典， 这个字典又包含两个keys
        # 一个key是一门语言， 如 'en'； 另外一个key也是一门语言， 如'it'
        # 因此这里还要再套一个[language]的key才能得到，这门语言的句子
        
        # 确保获取正确的字段
        if 'translation' in item and language in item['translation']:
            text = item['translation'][language]
            if isinstance(text, str) and text.strip():  # 确保是有效字符串
                yield text


def get_or_build_tokenizer(configuration, dataset, language):

    # 这里是一门语言生成一个tokenizer_en.json文件， 通过这个文件名来获取这门语言的tokenizer（如果有这样一个被训练过的tokenizer
    tokenizer_path = Path(configuration['tokenizer_file'].format(language))
    # 这里的configuration['tokenizer_file']指的就是"tokenizer_{0}.json"这个string
    # 然后通过.format(language)来把{0}替换成language，并生成language的json文件，如下面例子
    # print("tokenizer_{0}.json".format('en'))  # tokenizer_en.json

    if not Path.exists(tokenizer_path):
        # 这里要参考tokenizers_lib这个自建类说明会更清晰一些
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()  # 这里是tokenizer.pre_tokenizer 而不是原来的tokenizer.pre_tokenizers, 没有‘s’
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)

        # 保存tokenizer并将其保存再tokenizer_path的路径下
        tokenizer.save(str(tokenizer_path))

    else:
        # else的情况就是说这么语言的tokenizer已经有了， 那么只要读取就可以
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(configuration):

    # step 1. 通过load_dataset读取数据
    # step 2 通过random_split划分train_dataset_raw和val_dataset_raw
    # step 3 通过get_or_build_tokenizer生成tokenizer_src， tokenizer_tgt
    # step 4 通过BilingualDataset这个类生成train_dataset和val_dataset
    # step 5 通过DataLoader生成train_dataloader和val_dataloader

    # load_dataset()函数中的一些参数：
    # 1.path：参数path表示数据集的名字或者路径。
    # 2.name：参数name表示数据集中的子数据集，当一个数据集包含多个数据集时，就需要这个参数;
    # opus_books 包含了很多小的数据集， 每个小的数据集都有自己的名称， 如：
    # ['ca-de', 'ca-en', 'ca-hu', 'ca-nl', 'de-en', 'de-eo', 'de-es', 'de-fr', 'de-hu', 'de-it',
    # 'de-nl', 'de-pt', 'de-ru', 'el-en', 'el-es', 'el-fr', 'el-hu', 'en-eo', 'en-es', 'en-fi',
    # 'en-fr', 'en-hu', 'en-it', 'en-nl', 'en-no', 'en-pl', 'en-pt', 'en-ru', 'en-sv', 'eo-es',
    # 'eo-fr', 'eo-hu', 'eo-it', 'eo-pt', 'es-fi', 'es-fr', 'es-hu', 'es-it', 'es-nl', 'es-no',
    # 'es-pt', 'es-ru', 'fi-fr', 'fi-hu', 'fi-no', 'fi-pl', 'fr-hu', 'fr-it', 'fr-nl', 'fr-no',
    # 'fr-pl', 'fr-pt', 'fr-ru', 'fr-sv', 'hu-it', 'hu-nl', 'hu-no', 'hu-pl', 'hu-pt', 'hu-ru',
    # 'it-nl', 'it-pt', 'it-ru', 'it-sv']
    # 现在做的是英文和意大利文的翻译模型， 所以要用到'en-it'， 那么就加上这个参数name='en-it'
    # 3.split='train'就是train数据集里面的全部样本， split='train[：10%]'就是train数据集里面的前10%的样本

    # EXAMPLE:
    # dataset2 = load_dataset(path='opus_books', name='en-it', split='train')
    # print(dataset2.column_names)
    # print(dataset2[100])

    # dataset2[100]输出的结果是一个字典， 这个字典包含两个keys 1个是id， 1个是translation，
    # translation这个key对应的value是一个字典， 包含两种语言的两个key是，如 'en' 和 'it'，
    # 这两个keys有分别对应两种语言的句子作为各自的value
    dataset_raw = load_dataset(path=f"{configuration['datasource']}",
                               name=f"{configuration['lang_src']}-{configuration['lang_tgt']}",
                               split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(configuration, dataset_raw, configuration['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(configuration, dataset_raw, configuration['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    # train_test_split in pytorch we use random_split
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, configuration['lang_src'], configuration['lang_tgt'],
                                configuration['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, configuration['lang_src'], configuration['lang_tgt'],
                              configuration['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][configuration['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][configuration['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=configuration['batch_size'], shuffle=True)
    # 这样做的结果是 将train_dataset 每batch_size个样本合成一个batch， 因此会增加一个batch的维度
    # DataLoader 会根据 batch_size 进行批次划分。
    # 对于 batch_size=8，每次迭代时，DataLoader 会将八个样本打包在一起，因此对于每个批次，张量的形状会增加一个"batch批次"维度。

    # 根据上面batch_size=configuration['batch_size'] = 8这个参数的设定
    # for batch in train_dataloader
    # 果 train_dataset['encoder_input'] 的形状是 torch.Size([seq_len])（即每个样本有 seq_len 个元素），这里是1个样本
    # 当 batch_size=8 时，返回的批次中，每个 batch['encoder_input'] 包含 8 个样本，
    # 每个样本的形状是 torch.Size([seq_len])。因此 batch['encoder_input'] 的形状将变为 [batch_size=8, seq_len]，其中：
    # 8 是批次大小，即有 8 个样本。
    # seq_len 是每个样本的原始张量形状。
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(configuration, src_vocab_size, tgt_vocab_size):

    # step 1 define device
    # step 2 get_ds
    # step 3 get_model
    # step 4 optimizer
    # step 5 loss function

    model = build_transformer(src_vocab_size,
                              tgt_vocab_size,
                              configuration['seq_len'],
                              configuration['seq_len'],
                              configuration['d_model'])

    return model


def greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):

    tgt_sos_token = tokenizer_tgt.token_to_id("[SOS]")
    tgt_eos_token = tokenizer_tgt.token_to_id("[EOS]")

    # recompute the encoder_output and reuse it for every token we get from the decoder
    encoder_output = model.encode(encoder_input, encoder_mask)
    # 这里是调用model.encode, 然后通过encode调用Encoder class里面的forward函数，
    # 这个forward函数会通过调用EncoderBlock类里面的forward去计算attention
    # 但是在Encoder class里面的forward函数里面结束了forward之后， 最后是用了LayerNormalization的， 所以出来的结果是标准化处理过的
    # 所以这里的encoder_output是经过标准化处理过之后的结果

    # initialize the decoder input with the tgt_sos_token
    decoder_input = torch.tensor(tgt_sos_token).view(1, 1).type_as(encoder_input).to(device)
    # decoder_input is of shape (batch_size, seq_len)
    print("step 1, decoder_input before while loop")
    print(decoder_input.shape)
    print(decoder_input)
    # 进入while loop之前， 这里的decoder_input只是一个长度为1的tgt_sos_token， 只会提供"[SOS]"
    # 所以上面对应的是 [batch_size=1, seq_len=1]
    # tensor([[2]]) 对应的就是tgt_sos_token

    while True:

        if decoder_input.size(1) == max_len:
            break

        # build mask for the decoder_input
        decoder_mask = look_ahead_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
        # def look_ahead_mask() 输出的结果look_ahead_mask(decoder_input.size(1))是一个True/False元素的tensor，
        # encoder_mask 是来自batch的， 而batch是一个字典，
        # 对应字典的key是encoder_mask， 对应的值encoder_mask是一个元素为1/0的tensor
        # 因此经过type_as()函数之后look_ahead_mask(decoder_input.size(1))这个True/False元素的tensor会变成int类型
        # 最后的输出结果就变成了对应位置True/False元素为1/0的tensor
        print("step 1.5 decoder_output before transformer")
        print(decoder_input.shape)
        print(decoder_input)
        # 在进入decode之前， 这个decoder_input是没做任何处理的， 所以shape还是和while loop之前的一致
        # 进入while loop之前， 这里的decoder_input只是一个长度为1的tgt_sos_token， 只会提供"[SOS]"
        # 所以上面对应的是 [batch_size=1, seq_len=1]
        # tensor([[2]]) 对应的就是tgt_sos_token

        # calculate the decoder_output
        # 这里出来的结果是
        decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        print("step 2 decoder_output after transformer")
        print(decoder_output.shape)  # (batch_size, seq_len * d_model)
        print("######")
        # 在进入decode值个函数之后， 他其实是经历了embedding， positional_decoding和decoder这三部分
        # 在embedding处理完之后原来的[batch_size=1, seq_len=1]shape就会变成[batch_size=1, seq_len=1，d_model]
        # 这里比原来是增加了embedding的维度，即d_model

        # 这里首先需要注意的：decoder_input直到这一步是没有经过layer_normalization的， 而encoder_output是经过了layer_normalization的
        # 这里调用了model.decode， 是会调用Decoder Class里面的forward方法，在结束了for loop之后是调用了self.norm对结果进行标准化处理
        # 在for loop里面，依然用的变量还是decoder_input（没有标准化过的）
        # forloop里面调用的是DecoderBlock里面的forward函数， 这个forward函数有三步，
        # 第一步是经过self attention， 这一步进去self.self_attention_block(x, x, x, tgt_mask)的时候是没做标准化处理的
        # 是在这一步最后， 计算完residual之和后才做layer_normalization的

        # 第二部是经过cross attention，
        # 这一步self.cross_attention_block(input_cross, encoder_output, encoder_output, src_mask)中，
        # input_cross是上一步self attention的结果， 是做了layer_normalization的， 同时encoder_output是传入的变量也是做了layer_normalization的
        # 所以在计算corss attention的时候， 输入变量input_cross与encoder_output都是标准化过的结果
        # 然后这一步的最后， 计算完residual之和后又做了layer_normalization， 然后这个结果被传入到第三部 residual_connection中

        # 第三部的residual connection是调用了ResidualConnection class中的forward函数， 在函数的最后一步中也是做了layer normalization的
        # 所以最后这个decoder_output的结果也是标准化之后的结果

         # calculate proj_output
        # proj_output = model.project(decoder_output)
        # prob = proj_output[:, -1]

        # predict the last token's probability distribution
        # prob = model.project(decoder_output[:, -1, :])
        # 而这里取了最后一个单词，
        # project function output (d_model, tgt_vocab_size)
        prob = model.project(decoder_output[:, -1])

        # prob已经是最后一个单词的概率分布了，
        # 所以prob的shape 是 1 * tgt_vocab_size了， 所以下面的dim是用1
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.tensor(next_word.item()).view(1, 1).type_as(encoder_input).to(device)
        ], dim=1)
        # decoder_input 是一个 1 * seq_len的tensor所以在concat的时候要用dim=1来concat
        print("step 3 decoder_input after concatenating")
        print(decoder_input.shape)
        # 这个shape 还是 [batch_size=1， seq_len]
        # 但是这里的seq_len比上一轮while loop的迭代的seq_len长1位，
        # 因为我们把预测那个最后一个字的token concat 回去给到上一轮的预测结果句子， 因此会多一位
        # NOTE：decoder_input的shape始终是 [batch_size=1， seq_len]， 并不是[batch_size=1， seq_len， d_model]
        # 只是每一轮seq_len都会增加1位， 预测的值的token的长度就是1位
        print(decoder_input)
        print("######")

        if next_word == tgt_eos_token:
            break

        # to remove the batch dimension, since we one have one sample for validation
        decoder_input = decoder_input.squeeze(0)

    return decoder_input
    print(f"final step decoder input after squeeze(0) {decoder_input.squeeze(0).shape}")
    print(decoder_input.squeeze(0))
    # 这里squeeze(0)就是把batch_size这个dimension去掉， 但是应该不会有太大的差别，
    # 因为在validation的时候，一般batch_size就是1， 所以，没有太大的影响


def run_validation(model,
                   validation_dataset,
                   tokenizer_src,
                   tokenizer_tgt,
                   max_len,
                   device,
                   print_msg,
                   global_state,
                   writer=None,
                   num_examples=2):

    # 说明model是在evaluation的状态， 同时会停止例如batch_norm的操作， 不实在训练
    model.eval()
    # 在 PyTorch 中，model.train() 和 model.eval() 是两种模式切换方法，用于控制模型在不同阶段（训练或评估/推理）下的行为。它们的主要区别体现在某些特定的层（如 Dropout 和 BatchNorm）的工作方式上。
    #
    # 1. model.train()：
    # model.train() 会将模型设置为 训练模式。在这个模式下，模型中的一些层会以训练时的方式运行，特别是：
    #
    # Dropout：在训练过程中，Dropout 层会随机地“丢弃”一部分神经元，以防止过拟合。
    # BatchNorm：BatchNorm 层在训练时会计算当前批次的均值和方差，并更新其内部的均值和方差的移动平均值（用于推理阶段）。
    # 总结：model.train() 告诉模型现在处于训练阶段，模型的参数可以被更新，并且特定层会根据训练的需求调整其行为。
    #
    # 2. model.eval()：
    # model.eval() 会将模型设置为 评估模式。在这个模式下，模型中的一些层的行为会发生变化，尤其是：
    #
    # Dropout：在评估时，Dropout 层会停止丢弃神经元，所有神经元都会参与计算。
    # BatchNorm：BatchNorm 层在评估模式下，会使用训练过程中积累的均值和方差，而不再计算当前批次的均值和方差。
    count = 0
    source_texts = []
    expected = []
    predicted = []

    console_width = 80
    # console_width = 80 与tqdm一起使用， 用于限制在控制台上输出的行宽，通常以 80 个字符为标准宽度。
    # 这在格式化输出、显示进度条、控制台日志、以及保持整齐的输出时非常有用。

    with torch.no_grad():

        for batch in validation_dataset:

            count += 1

            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # 下面要做的是validation不是training，
            # step1 所以流程是通过前一个单词推断下一个单词，
            # step2 把这个推断出来的单词和上一个单词一并使用， 并推断出下一个单词，
            # step3 由于1和2的机制， 所以这里用到的padding_mask只是look_ahead_mask
            # 而且这里只会使用一次的encoder_input去计算encoder_output，
            # 上面每一个单词的推断都是用同一个的这个encoder_output

            model_output = greedy_decode(model, encoder_input, encoder_mask,tokenizer_src, tokenizer_tgt, max_len, device)
            # decoder的部分不是在这里计算的， 而是在greedy_decode里面去计算， 所以这里不需要decoder_input

            # 这里要关注以下greedy_decode的输出结果的shape是什么， 这里的输出结果是decoder_input，它的shape是(1)
            # 没有batch_size这个dimension只有seq_len这个dimension

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            # 检查以下这里为什么要加个0
            # 这里的batch是在batch_iterator里面读取出来的
            # 而batch_iterator是来自val_dataloader的， 这里的val_dataloader他是一个DataLoader()
            # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
            # 这里的batch_size按validation传入的值是1， 当然也可以是其他值， 1， 2，3， 4，5， 6。。。
            # val_dataset 的值是BilingualDataset()这个类的实例， 是一个个的字典，
            # 而这一个个的字典中， src_text这个key对应的是一个句子， 是一个string

            # 但经过了DataLoader之后， 它的工作机制如下：

            # DataLoader 的批处理行为：
            #
            # 当你设置 batch_size=1 时，DataLoader 会把每个数据项放在一个列表或张量中，即便它的大小是 1。
            # 这样，DataLoader 会返回一个批次，而不是原始的单个元素。
            # 对于一个字符串 'hello world'，DataLoader 会将其打包为一个单元素的列表 ['hello world']，以满足批处理的要求。

            # batch_size 的作用：
            # 当 batch_size 大于 1 时，DataLoader 会返回多个数据项的批次。
            # 如果 batch_size=1，
            # 虽然每批只有一个数据项，但 DataLoader 仍然将其封装在一个批次的形式中，这就导致原本是字符串的值变成了一个列表。

            # EXample：
            # 假设val_dataset里面包含了四个字典的结果， 每个字典都有src_text这个key， 以及对应的value， value以string的格式存储
            # dict1 = {'src_text':'a'}
            # dict2 = {'src_text':'b'}
            # dict3 = {'src_text':'c'}
            # dict4 = {'src_text':'d'}

            # 然后假设这里batch_size=2
            # 那么就会出现一个batch有两个字典， 一共两个batch,
            # 假设batch_1 包含dict1 和dict2， batch_2包含dict3 和dict4

            # 按照DataLoader的工作原理， 出来的src_text
            # 对应的value就会是 batch_1: 'src_text'：['a', 'b'], batch_1: 'src_text'：['c', 'd']
            # 'src_text' key对应的value就是一个list， 这里是因为做了批次处理

            # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
            #
            # for batch in train_dataloader:
            #     print(batch)

            # output
            # [{'src_text': ['a', 'b']}]
            # [{'src_text': ['c', 'd']}]

            # 所以上面的结果batch已经是一个batch_size为1的batch了，
            # 那么batch['src_text']对应的就是一个长度为1的列表
            # 因此要通过batch['src_text'][0]查询它的值

            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            # .detach() 的作用：
            # 将一个张量（tensor）转换为 NumPy 格式时，通常需要先调用.detach()，
            # 尤其是当这个张量是在计算图中产生的并且有梯度（即 requires_grad=True 时）。

            # .detach() 的作用是创建一个与原张量共享内存的副本，但该副本不再需要梯度，也不会参与计算图的构建。换句话说，它与计算图断开了联系。
            # 在使用 .detach() 后，PyTorch 不会跟踪该张量上的任何操作。

            # model_output是一个1 * tgt_vocab_size的 被转化成token数字的tensor
            # 要先把model_output.detach().cpu().numpy()
            # .numpy():
            # 作用：将 PyTorch 张量转换为 NumPy 数组。
            # 原因：在 PyTorch 中，张量和 NumPy 数组可以无缝地相互转换，但前提是张量位于 CPU 上。
            # .cpu():
            # 作用：将张量从 GPU 转移到 CPU。
            # 原因：很多时候你需要将张量转换为 NumPy 数组，而 NumPy 只支持 CPU 上的操作。
            # .detach():
            # 作用：将张量从计算图中分离出来，停止对其进行梯度跟踪。
            # 原因：在 PyTorch 中，张量默认是与计算图相关联的，尤其是在训练过程中，当你需要进行反向传播时。
            # 这意味着 PyTorch 会跟踪所有与该张量相关的操作。
            # 如果你不需要对这个张量再进行梯度计算（例如，在进行模型推理或保存张量的值时），可以使用 .detach() 来分离张量。
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_output_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break


def train_model(configuration):

    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
        # else "mps" if torch.backends.mps.is_available() \
        # else "cpu"
    print("Using device:", device)

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    elif (device == 'mps'):
        print(f"Device name: <mps>")

    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")

    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{configuration['datasource']}_{configuration['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(configuration)
    model = get_model(configuration,
                      tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    # Tensorboard, 不懂tensorboard的代码也没关系， 只是拿来做试图的
    writer = SummaryWriter(configuration['experiment_name'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = configuration['preload']
    model_filename = latest_weights_file_path(configuration) if preload == 'latest' else get_weights_file_path(configuration, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, configuration['num_epochs']):

        torch.cuda.empty_cache()
        model.train()
        # 这是因为 train() 是 PyTorch nn.Module 类中的一个内置方法，用于将模型切换到训练模式。
        # 所有继承自 nn.Module 的子类都会自动继承这个方法。
        # train() 方法会将模型设置为训练模式，启用 dropout 和 batch normalization 等特性。
        # 如果你没有在类中覆盖 train() 方法，默认会继承 nn.Module 的实现。

        # 这里的train并不是一个训练的方法， 还是会通过loss_fn.backward()完成training的

        # 使用 tqdm 的代码会提供一个可视化的进度条，让你能够随时了解训练的进展，方便监控训练过程。
        # 不使用 tqdm 则不会显示进度条，只是普通的迭代。对短时间的任务可能无影响，但对长时间的任务来说，没有反馈可能会影响体验。
        # batch_iterator = train_dataloader
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        # train_dataloader 来自get_ds里面的train_dataset
        # train_dataloader = DataLoader(train_dataset, batch_size=configuration['batch_size'], shuffle=True)
        # train_dataloader 包含了八个train_dataset的样本
        # 而这里的train_dataset又是BilingualDataset 类的一个实例， 一个train_dataset是__getitem__方法当中返回的一个字典

        # 这里的batch_iterator 就相当于 train_dataloader，
        # 是batch-size=8个train_dataset的迭代器，
        # 也就是batch-size=8个 字典的迭代器

        # 那么 for batch in batch_iterator: 就是在遍历这 8 个字典
        for batch in batch_iterator:

            print(batch.shape)
            encoder_input = batch['encoder_input'].to(device)  # batch_size * seq_len
            # 得到batch_size * seq_len这样的结果是由于上面
            # train_dataloader = DataLoader(train_dataset, batch_size=configuration['batch_size'], shuffle=True)
            # 下面的注释。简单说就是：
            # 原来train_dataset是一个样本的字典， 对应的train_dataset['encoder_input']是一个形如tensor.Size([seq_len])的tensor，
            # 由于batch_size=configuration['batch_size']这个的设定， 会把batch_size个样本打包放一起
            # 那么就在原来单个样本的基础上增加一个batch_size的维度，便有了batch_size * seq_len。
            decoder_input = batch['decoder_input'].to(device)  # batch_size * seq_len
            encoder_mask = batch['encoder_mask'].to(device)  # batch_size * 1 * 1 *  seq_len,
            # 在没有batch之前， encoder_mask在计算的时候是做了unsqueeze(0),unsqueeze(0)两次操作， 使得其shape为1 * 1 *  seq_len,
            # 然后加上了batch这个dimension之后就变成了 batch_size * 1 * 1 * seq_len
            decoder_mask = batch['decoder_mask'].to(device)
            # 在没有batch之前， decoder_mask在计算的时候是与look_ahead_mask做了一次并集的处理，
            # 这里的look_ahead_mask的shape是 1 * seq_len * seq_len，
            # 然后加上了batch这个dimension之后就变成了 batch_size * 1 * seq_len * seq_len

            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # CrossEntropyLoss 要求：
            # 输入 logits 的形状是 (N, C)，其中 N 是样本数量，C 是类别数。
            # 输入标签的形状是 (N)，其中 N 是样本数量，对应每个样本的类别标签。
            # 因此，tensor_a.view(-1, vocab_size) 和 tensor_label.view(-1) 的形状是匹配的，符合 CrossEntropyLoss 的要求。

            # set_postfix 相当于print
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            # optimizer.zero_grad() (set_to_none=False)： 梯度被重置为零，梯度张量保留。
            # optimizer.zero_grad(set_to_none=True)： 梯度被设置为 None，释放内存并在反向传播时重新分配梯度张量。
            # 对于内存敏感的任务，使用 set_to_none=True 可以提高效率，而默认的 zero_grad() 更适合常见的使用场景。

            global_step += 1

        # saving the model
        # Save the model at the end of every epoch
        # 每一个epoch保存一次模型
        model_filename = get_weights_file_path(configuration, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':

    configuration = get_config()
    train_model(configuration)

