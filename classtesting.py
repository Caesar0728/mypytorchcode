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
