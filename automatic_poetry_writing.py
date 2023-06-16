# 机 构：中国科学院大学
# 程序员：李浩东
# 时 间：2023/5/1 19:20

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 超参数定义
data_path = 'data/tang.npz'
seq_len = 60  # 根据序列长度重新划分无空格数据集
batch_size = 16
EPOCH = 3
lr = 1e-3
hidden_dim = 256
vocab_size = 8293
embedding_dim = 128
LSTM_layers = 3  # 隐藏层层数
max_gen_len = 100  # 生成唐诗的最长长度

# 数据处理
datas = np.load(data_path, allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
# print(data)
# print(ix2word)
# print(word2ix)


class Poem():
    def __init__(self):
        # 初始化函数，将过滤掉数据中的'</s>'的结果存储在self.no_space_data中
        self.no_space_data = self.filter_space()
        print(len(self.no_space_data))

    # 获取数据的输入序列和标签序列
    def __getitem__(self, idx):
        # 通过输入索引idx，将数据拆分成长度为config.seq_len的序列
        text = self.no_space_data[idx * seq_len:(idx + 1) * seq_len]
        # 标签序列相比输入序列向后移动了一个字符
        label = self.no_space_data[idx * seq_len + 1:(idx + 1) * seq_len + 1]
        # 将输入序列和标签序列转为torch.Tensor
        text = torch.from_numpy(np.array(text)).long()
        label = torch.from_numpy(np.array(label)).long()
        # 返回输入序列和标签序列
        return text, label

    # 获取数据可以生成的序列个数
    def __len__(self):
        # 无标签数据总长度/序列长度，即为序列个数
        return int(len(self.no_space_data) / seq_len)

    # 过滤数据中的空格，并返回过滤后的结果
    def filter_space(self):
        # 把二维的tensor拼接成一维并转回numpy
        splicing_data = torch.from_numpy(data).view(-1).numpy()
        # 过滤数值为8292的元素，即空格
        no_space_data = []
        for i in splicing_data:
            if i != 8292:
                no_space_data.append(i)
        # 返回过滤后的数据
        return no_space_data

# 构建模型
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)  # 嵌入层
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3)  # LSTM层
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, 2048), nn.ReLU(inplace=True), nn.Linear(2048, self.vocab_size))  # 全连接层

    def forward(self, input, hidden = None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.fc(output.view(seq_len * batch_size, -1))
        return output, hidden


potryModel = PoetryModel(vocab_size, embedding_dim, hidden_dim)  # 创建模型对象
optimizer = torch.optim.Adam(potryModel.parameters(), lr=lr, weight_decay=1e-3)  # 优化器
criterion = nn.CrossEntropyLoss()  # 损失函数

# 模型训练
def train(model, dataloader):
    model.train()
    total_batch = 0
    for epoch in range(EPOCH):
        train_loss = 0.0
        train_loader = tqdm(dataloader)
        for i, data in enumerate(train_loader, 0):
            # print(i)
            # print(data)
            inputs, labels = data[0], data[1]
            labels = labels.view(-1)  # 将labels展平成(batch_size*seq_len)的形状，以便计算损失
            optimizer.zero_grad()
            outputs, hidden = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 100 == 0:
                print("    Test--->Epoch:%d,  Batch:%3d,  Loss:%.8f" % (epoch + 1, i + 1, loss.item()))
            total_batch += 1

# 生成古诗
def generate_poem(model, start_words):
    results = list(start_words)  # results用于存储生成的文本
    start_words_len = len(start_words)
    # 将开始符号<START>转化为对应的索引，然后转化为张量并 reshape 为(1,1)，作为模型的输入
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    # 最开始的隐状态初始为0矩阵
    hidden = torch.zeros((2, LSTM_layers * 1, 1, hidden_dim), dtype=torch.float)
    # hidden = None
    model.eval()
    with torch.no_grad():
        for i in range(max_gen_len):
            output, hidden = model(input, hidden)
            # 读取输入的第一句,如果在给定的句首中，input 为句首中的下一个字
            if i < start_words_len:
                w = results[i]
                input = input.data.new([word2ix[w]]).view(1, 1)
            # 生成后面的句子,否则将 output 作为下一个 input 进行
            else:
                top_index = output.data[0].topk(1)[1][0].item()  # 从输出中选择概率最高的下一个词语
                w = ix2word[top_index]
                results.append(w)
                input = input.data.new([top_index]).view(1, 1)
            if w == '<EOP>':
                del results[-1]
                break
    results = ' '.join(i for i in results)
    return results  # 生成的文本


# 生成藏头诗
def generate_acrostic_poem(model, start_words_list):
    results_total = []
    for start_words in start_words_list:
        results = list(start_words)  # results用于存储生成的文本
        start_words_len = len(start_words)
        # 将开始符号<START>转化为对应的索引，然后转化为张量并 reshape 为(1,1)，作为模型的输入
        input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
        # 最开始的隐状态初始为0矩阵
        hidden = torch.zeros((2, LSTM_layers * 1, 1, hidden_dim), dtype=torch.float)
        # hidden = None
        model.eval()
        with torch.no_grad():
            for i in range(max_gen_len):
                output, hidden = model(input, hidden)
                # 读取输入的第一句,如果在给定的句首中，input 为句首中的下一个字
                if i < start_words_len:
                    w = results[i]
                    input = input.data.new([word2ix[w]]).view(1, 1)
                # 生成后面的句子,否则将 output 作为下一个 input 进行
                else:
                    top_index = output.data[0].topk(1)[1][0].item()  # 从输出中选择概率最高的下一个词语
                    w = ix2word[top_index]
                    results.append(w)
                    input = input.data.new([top_index]).view(1, 1)

                if w == '<EOP>':
                    del results[-1]
                    break
        results = ' '.join(i for i in results)
        results_total.append(results)
    return results_total


# 保存模型
def save_param(model, path):
    torch.save(model.state_dict(), path)  # 保存网络里的参数

# 加载模型
def load_param(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    poem = Poem()
    # 生成数据迭代器
    poem_loader = DataLoader(poem, batch_size=batch_size, shuffle=True, num_workers=2)
    print('-----------------------------开始训练-----------------------------------')
    train(potryModel, poem_loader)

    save_param(potryModel, 'potryModel.pth')
    load_param(potryModel, 'potryModel.pth')

    print('-----------------------------开始测试-----------------------------------')
    results = generate_poem(potryModel, '白日依山尽')
    print('生成古诗：', results)

    results = generate_acrostic_poem(potryModel, ["壮", "丽", "北", "京"])
    print('生成藏头诗：')
    for item in results:
        print(item)