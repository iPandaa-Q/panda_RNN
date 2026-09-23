import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import math

# 数据准备
JSON_DIR = "./chinese-poetry/御定全唐詩/json"

def is_regular_poem(text):
    """
    判断一首诗是否全部由 5 言或全部由 7 言组成。
    是则返回True，否则False。
    """
    sentences = re.split(r"[，。？！、；：]", text)  # 以这些为依据把古诗分句
    sentences = [s.strip() for s in sentences if s.strip()]  # 去掉空的
    if len(sentences) < 2:
        return False
    lengths = [len(s) for s in sentences]  # 取长度，后续筛选
    return all(l == 5 for l in lengths) or all(l == 7 for l in lengths)

def load_poems(json_dir, max_poems=100000):
    poems = []
    if not os.path.exists(json_dir):
        return poems

    files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    for fname in files:
        filepath = os.path.join(json_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data

        for item in items:
            paragraphs = item.get("paragraphs", [])
            if not paragraphs:
                continue

            # 去掉作者署名：破折号后面的内容全部丢掉
            clean_paragraphs = []
            for p in paragraphs:
                if "——" in p:
                    p = p.split("——")[0]
                clean_paragraphs.append(p)

            # 拼接古诗
            text = "".join(clean_paragraphs).strip()

            # 去掉无关符号啥的
            text = re.sub(r"[a-zA-Z0-9\s]", "", text)

            # 太长和太短的都不要
            if len(text) < 10 or len(text) > 200:
                continue

            # 五言/七言过滤
            if not is_regular_poem(text):
                continue

            poems.append(text)
            if len(poems) >= max_poems:
                return poems
    return poems
poems = load_poems(JSON_DIR, )

# 创建词表，模型可以预测的所有类别
all_text = "".join(poems)

# 统计所有字符出现次数
counter = Counter(all_text)

# 只保留出现次数 >= MIN_FREQ 的字符
MIN_FREQ = 50
chars = sorted([c for c, cnt in counter.items() if cnt >= MIN_FREQ])
# 只会保留最常用的 3000 个汉字
MAX_VOCAB = 3000
chars = chars[:MAX_VOCAB]

PAD_TOKEN = "<PAD>"  # 填充符
SOP_TOKEN = "<SOP>"  # 开始位置
EOP_TOKEN = "<EOP>"  # 结束位置
UNK_TOKEN = "<UNK>"  # 未知字符

# 四个字符也加到词表中
for tok in [PAD_TOKEN, SOP_TOKEN, EOP_TOKEN, UNK_TOKEN]:
    if tok not in chars:
        chars.append(tok)

char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(chars)  # 词表容量，也就是模型可以预测的所有类别的总数

PAD_IDX = char2idx[PAD_TOKEN]
SOP_IDX = char2idx[SOP_TOKEN]
EOP_IDX = char2idx[EOP_TOKEN]
UNK_IDX = char2idx[UNK_TOKEN]

class PoetryDataset(Dataset):
    def __init__(self, poems, char2idx, fixed_len=64):
        self.fixed_len = fixed_len
        self.samples = []
        for poem in poems:
            ids = [char2idx.get(c, UNK_IDX) for c in poem]
            if len(ids) < 5:
                continue

            # 输入和目标错开一位
            # 自回归（Autoregressive）预测
            x = [SOP_IDX] + ids
            y = ids + [EOP_IDX]

            if len(x) > fixed_len:
                x = x[:fixed_len]
                y = y[:fixed_len]
            else:
                pad_len = fixed_len - len(x)
                x = x + [PAD_IDX] * pad_len
                y = y + [PAD_IDX] * pad_len

            self.samples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

FIXED_LEN = 64
dataset = PoetryDataset(poems, char2idx, fixed_len=FIXED_LEN)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

class CharLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入到门的权重
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))

        # 隐状态到门的权重
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))

        # 两个偏置
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # 均匀初始化
        stdv = 1.0 / math.sqrt(self.hidden_dim) if self.hidden_dim > 0 else 0
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x, hidden=None):
        B, T, _ = x.shape
        H = self.hidden_dim

        if hidden is None:
            h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]

            # 把x_t和h分别过权重，再加偏置
            gates = x_t @ self.W_ih.T + self.b_ih + h @ self.W_hh.T + self.b_hh

            # 按PyTorch顺序切分：i, f, g, o
            i, f, g, o = gates.chunk(4, dim=1)

            i = torch.sigmoid(i)  # 输入门
            f = torch.sigmoid(f)  # 遗忘门
            g = torch.tanh(g)  # 候选记忆
            o = torch.sigmoid(o)  # 输出门

            c = f * c + i * g
            h = o * torch.tanh(c)

            outputs.append(h.unsqueeze(1))

        out = torch.cat(outputs, dim=1) 
        return out, (h, c)

# 以下是验证代码，证明手写和调用效果类似
torch.manual_seed(42)

input_size = 8
hidden_size = 16
B, T = 2, 5

# 创建两个实现
lstm_ref = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
lstm_manual = CharLSTM(input_size, hidden_size)

# 把参考权重复制到手写实现
with torch.no_grad():
    lstm_manual.W_ih.copy_(lstm_ref.weight_ih_l0)
    lstm_manual.W_hh.copy_(lstm_ref.weight_hh_l0)
    lstm_manual.b_ih.copy_(lstm_ref.bias_ih_l0)
    lstm_manual.b_hh.copy_(lstm_ref.bias_hh_l0)

# 输入相同数据
x = torch.randn(B, T, input_size)

# 前向传播
with torch.no_grad():
    out_ref, (h_ref, c_ref) = lstm_ref(x)
    out_manual, (h_manual, c_manual) = lstm_manual(x)

# 对比
print("输出对比")
print(f"out      形状: ref={out_ref.shape}, manual={out_manual.shape}")
print(f"out      max diff: {(out_ref - out_manual).abs().max().item():.2e}")
print(f"h_n      max diff: {(h_ref[0] - h_manual).abs().max().item():.2e}")
print(f"c_n      max diff: {(c_ref[0] - c_manual).abs().max().item():.2e}")

# 逐时间步对比
print("逐时间步对比")
for t in range(T):
    diff = (out_ref[:, t, :] - out_manual[:, t, :]).abs().max().item()
    print(f"t={t}: max diff = {diff:.2e}")