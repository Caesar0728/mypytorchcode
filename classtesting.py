# # class MyClass:
# #
# #     class_variable = 10
# #
# #     @classmethod
# #     def f(cls, x):
# #         cls.class_variable += x
# #         print(f"this is class variable {cls.class_variable}")
# #         print("this is a class method")
# #
# #
# # MyClass.f(10)
# # m1 = MyClass()
# # m1.f(20)
#
# class Animal:
#     @classmethod
#     def speak(cls):
#         print("Animal speaks")
#
#
# class Dog(Animal):
#     @classmethod
#     def speak(cls):
#         print("Dog barks")
#
#
# class Cat(Animal):
#     @classmethod
#     def speak(cls):
#         print("Cat meows")
#
#
# # 调用子类的类方法
# Dog.speak()
# Cat.speak()

from collections import defaultdict


####### PART I #######
class WXPClass1:

    def __init__(self, y=0, m=0, d=0):

        self.year = y
        self.month = m
        self.day = d
        # self.date_dict = defaultdict(int)  # 默认值value 为 0的空字典


    def out(self):
        print(f"{self.year}-{self.month}-{self.day}")


    @classmethod
    def fromstring(cls, daystring):

        y, m, d = daystring.split('-')

        date1 = cls(int(y), int(m), int(d))
        # 下面如果直接print(cls.year)是会报错的，
        # 因为在调用fromstring时还没到实例化这一步， 就是最后return date1
        # 所以这个print(cls.year)会报错， 因为这个类尚未北实例化， 因此cls.year不存在
        # 连__init__函数都没运行过， 哪来的cls.year属性
        # print(cls.year)

        # 但是如果是print(date1.year)就不会报错了， 因为date1是已经被实例化了，
        # 通过cls(int(y), int(m), int(d))间接调用__init__函数进行实例化 所以不会报错
        # print(date1.year)

        # 下一不这里为什么不报错呢？
        # 是因为这里实际上是给这个类的year属性赋值
        # cls.year = int(y) + 1
        # print(cls.year)

        # 最后这一步是简介调用了__init__进行了实例化， 然后后面调用out函数就没问题了
        # 同时如果上面那个赋值cls.year = int(y) + 1实现完了之后， 然后在运行的return date1
        # 那么就会覆盖掉了原来的赋值， 有2024 + 1 = 2025 变回 2024
        return date1

print("from WPXCLass1")
date2 = WXPClass1.fromstring('2024-12-26')
# 下面这一步运行out函数明显就是运行了__init__函数的间接调用实例化了之后，
# 才可以调用这个类当中的out函数
date2.out()


# ####### PART II #######
class WXPClass2:

    def __init__(self, y=0, m=0, d=0):

        # self.year = y
        # self.month = m
        # self.day = d
        # self.date_dict = defaultdict(int)  # 默认值value 为 0的空字典
        self.date_dict = {"year": y,
                          "month": m,
                          "day": d}  # 默认值value 为 0的空字典


    # def out(self):
    #     print(f"{self.year}-{self.month}-{self.day}")

    def __getitem__(self, key):
        # 获取token对应的索引，如果不存在则返回未知词的索引
        return self.date_dict.get(key, 9999)

    def out1(self):
        # 如果没有上面的__getitem__那么这样写会报错的
        # 只能写 print(f" year is {self.date_dict['year'] + 1}")
        print(f" year is {self['year'] + 1}")

    def out2(self):
        # 如果没有上面的__getitem__那么这样写会报错的
        # 只能写 print(f" year is {self.date_dict['year'] + 1}")
        print("without using __getitem__")
        print(f" year is {self.date_dict['year'] + 2}")

    @classmethod
    def fromstring(cls, day_string):

        y, m, d = day_string.split('-')

        date1 = cls(int(y), int(m), int(d))
        # 下面如果直接print(cls.year)是会报错的，
        # 因为在调用fromstring时还没到实例化这一步， 就是最后return date1
        # 所以这个print(cls.year)会报错， 因为这个类尚未北实例化， 因此cls.year不存在
        # 连__init__函数都没运行过， 哪来的cls.year属性
        # print(cls.year)

        # 但是如果是print(date1.year)就不会报错了， 因为date1是已经被实例化了，
        # 通过cls(int(y), int(m), int(d))间接调用__init__函数进行实例化 所以不会报错
        # print(date1.year)

        # 下一不这里为什么不报错呢？
        # 是因为这里实际上是给这个类的year属性赋值
        # cls.year = int(y) + 1
        # print(cls.year)

        # 最后这一步是简介调用了__init__进行了实例化， 然后后面调用out函数就没问题了
        # 同时如果上面那个赋值cls.year = int(y) + 1实现完了之后， 然后在运行的return date1
        # 那么就会覆盖掉了原来的赋值， 有2024 + 1 = 2025 变回 2024
        return date1

print("from WPXCLass2")
date2 = WXPClass2.fromstring('2024-12-26')
# 下面这一步运行out函数明显就是运行了__init__函数的间接调用实例化了之后，
# 才可以调用这个类当中的out函数
date2.out1()
date2.out2()

# ####### PART III #######
#
# from collections import defaultdict, Counter
#
#
# class Vocab:
#
#     ###### 这个类里头几个点需要注意的 ######
#     ###### 1. 通过@classmethod避开实例化函数__init__, 然后在return的时候通过cls()间接实例化，
#     ###### 这个实例化后的object具有两个属性 1）self.idx_to_token 一个列表 2）self.token_to_idx 一个字典
#     ###### def __len__(self)是针对 self.idx_to_token 这个列表用的，
#     ###### def __getitem__(self, token): 是针对self.token_to_idx 这个字典用的
#
#     ###### 可以通过 比较 def convert_tokens_to_ids(self, tokens): 和 def convert_ids_to_tokens(self, indices): 看出区别
#     ###### 由于定义了def __getitem__(self, token): 所以 在 def convert_tokens_to_ids(self, tokens):中能够直接return self[token]
#     ###### 而在def convert_ids_to_tokens(self, indices): 只能用self.idx_to_token[index]， 因为前面没有定义类似于针对dict的__getitem__ 中 get函数 的方法
#     ###### 所以这里只能写全self.idx_to_token[index]， 而不能写self[index],
#     ###### 如果self.idx_to_token[index]这样写self[index]， 就相当于这个self被当做成了self.token_to_idx然后套用__getitem__,
#     ###### 去找“index”这个key， 而这个index是一个int， 不是一个token(string)那显然会报错。
#
#     def __init__(self, tokens=None):
#         self.idx_to_token = list()
#         self.token_to_idx = dict()
#
#         if tokens is not None:
#             if "<unk>" not in tokens:
#                 tokens = tokens + ["<unk>"]
#             for token in tokens:
#                 self.idx_to_token.append(token)
#                 self.token_to_idx[token] = len(self.idx_to_token) - 1
#             self.unk = self.token_to_idx['<unk>']
#
#     @classmethod
#     def build(cls, text, min_freq=1, reserved_tokens=None):
#         token_freqs = defaultdict(int)
#         for sentence in text:
#             for token in sentence:
#                 token_freqs[token] += 1
#         uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
#         uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
#         return cls(uniq_tokens)
#
#     # 在已经生成词汇表和词典的基础上
#     # 我们可以对词汇表和词典进行各项操作 ↓
#     def __len__(self):
#         # 返回词汇表的大小
#         return len(self.idx_to_token)
#
#     def __getitem__(self, token):
#         # 获取token对应的索引，如果不存在则返回未知词的索引
#         return self.token_to_idx.get(token, self.unk)
#
#     def convert_tokens_to_ids(self, tokens):
#         # 将token列表转换为索引列表（也就是将文字进行编码）
#         # 这里可以这样写self[token]是因为上面写了__getitem__函数， 能够对self.token_to_idx直接调用，
#         # 所以不用写成self.token_to_idx[token],
#         # 通过__getitem__的get函数，self[token]直接等价与self.token_to_idx.get(token, self.unk)
#         # 当然也可以直接写成 return [self.token_to_idx.get(token, self.unk) for token in tokens]
#         return [self[token] for token in tokens]
#
#     def convert_ids_to_tokens(self, indices):
#         # 将索引列表转换为token列表（也就是根据编码、找到相应的token）
#
#         return [self.idx_to_token[index] for index in indices]


####### PART IV #######
from collections import defaultdict, Counter


class Vocab:
    """
    可以同时接纳Token和text两种类型的数据
    对原始文字数据，调用build方法，进行分词、并完成词频筛选
    对Token数据，使用init中的流程，完成添加未知词、词汇表构建并根据词汇表进行编码
    建好词汇表后，再调用单独的方法来进行编码
    """

    def __init__(self, tokens=None):
        # init的输入参数是Token
        # 注意！这里的Token要求是一个【包含所有token的list】
        # 也就是这个列表里只能有token本身，不能再包含其他内容或者其他层次
        # 比如，一个list中包含了多个句子，每个句子都是按照token的方式排列的
        # 那这个list就不属于【包含所有token的list】，而是包含句子的list

        # 构建两个变量，一个idx_to_token，一个是token_to_idx
        # idx_to_token是列表，包含了数据集中所有的单词
        # token_to_idx是词汇表，是不重复的单词 + 索引构成的结果
        self.idx_to_token = list()
        self.token_to_idx = dict()

        # 如果输入了tokens（Tokens不为None）
        # 就直接进行未知词操作

        if tokens is not None:
            # 如果tokens中不包含"<unk>"这个词（未知词），则添加"<unk>"
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            # 遍历tokens，将每个token添加到idx_to_token列表，并在token_to_idx字典中映射其索引
            # 基于添加了未知词的Tokens，直接创造出列表 + 词汇表
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            # 设置未知词的索引，将未知的词设置为一个单独的属性self.unk
            self.unk = self.token_to_idx['<unk>']

    # 调用魔法命令classmethod，这个命令允许我们在不进行实例化的情况下使用类中的方法
    # build的输入参数与Vocab本身的init完全不同，因此我们可以运行它被单独调用
    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        # build，此时输入的参数有4个
        # cls是Vocab这个类本身，这魔法命令classmethod的要求
        # 有了cls就可以在不进行实例化的情况下直接调用build功能
        # text是需要构建词汇表和词典的文本，在这个文本上我们可以直接开始进行词频筛选
        # 注意！这个文本的范围很广泛，只要不是单一token list，都可以被认为是文本（见下面的详细说明）
        # min_freq是我们用于筛选的最小频率，低于该频率阈值的词会被删除
        # reserved_token是我们可以选择性输入的"通用词汇表"，假设text本身太短词太少的话
        # reserved_token可以帮助我们构建更大的词典、从而构建更大的词向量空间
        # 以上4个参数中只有text是必填的

        # 创建一个defaultdict字典，用于统计每个单词的出现频率
        token_freqs = defaultdict(int)
        # 遍历文本中的每个句子，统计每个单词的出现次数
        # 其中，单词使用变量token来代表
        for sentence in text:
            for token in sentence:
                # 不断保存到字典中的是——
                # 以token（词本身）作为键、词出现的频率作为值的键值对
                token_freqs[token] += 1

        # 创建一个空列表uniq_tokens，用于存储"<unk>"和输入用来保底的reserved_tokens
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])

        # 将token_freqs中保存的词和词频进行循环
        # 除了"<unk>"之外，过滤掉出现次数少于min_freq的词
        # 并将没有被过滤掉的词打包到一个列表中
        # 这个列表uniq_tokens就是过滤后的Tokens列表
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        # 将过滤后的Tokens列表放入cls，也就是Vocab类中
        # 这个Token进入到Vocab类之后，会触发init，开始进入init中的流程
        # 因此，只要调用build方法，就可以从text构建一组token、并将这组token放入Vocab类
        # 这是这个类的“递归”所在，我们可以调用类中的方法来创造类所需的数据类型
        # 并在该方法的最后重启这个类
        return cls(uniq_tokens)

        ####### 重点重点重点 #######
        # 这里是丢一个uniq_tokens的列表进行进行实例化， __init__的间接调用， 即__init__(uniq_tokens)， 并生成一个object
