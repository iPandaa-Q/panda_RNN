import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

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
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        assert embed_dim == hidden_dim, "权重绑定需要 embed_dim == hidden_dim"

        # nn.Embedding等价于：把one-hot向量乘以一个可学习的矩阵
        # padding_idx=PAD_IDX：告诉 Embedding 层 <PAD> 对应的向量永远是 0，且不参与梯度更新
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # 手动初始化 Embedding权重为小值
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # 把PAD行清零
        with torch.no_grad():
            self.embed.weight[PAD_IDX].fill_(0)
        
        # 调包使用
        self.lstm = nn.LSTM(
            input_size=embed_dim,  # 嵌入维度，把离散的字符索引映射成稠密向量
            hidden_size=hidden_dim,  # hidden_size=hidden_dim：隐藏状态维度
            num_layers=num_layers,  # 堆叠 2 层 LSTM
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # 权重绑定
        self.fc.weight = self.embed.weight

    def forward(self, x, hidden=None):
        emb = self.embed(x)  # (B, T)->(B, T, E)
        """
            hidden 是一个元组 (h_n, c_n)：
            h_n：最后一层的最后时间步隐状态，形状 (num_layers, B, H)
            c_n：最后一层的最后时间步记忆单元，形状 (num_layers, B, H)
        """

        out, hidden = self.lstm(emb, hidden)  # (B, T, E)->(B, T, H)
        out = self.dropout(out)  # (B, T, H)
        logits = self.fc(out)  # (B, T, H)->(B, T, V)
        return logits, hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharLSTM(vocab_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)  # 加了一个标签平滑

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 依旧学习率衰减


use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler('cuda') if use_amp else None



"""
下面是训练代码了
"""

EPOCHS = 80
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if use_amp:
            # 前向：autocast自动混合精度
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            # 反向：放大loss，防止FP16梯度下溢
            # 缩放因子完全内置在GradScaler对象里，作为它的内部状态自动维护
            scaler.scale(loss).backward()

            # 裁剪前先把梯度缩回真实值
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            # 另外，scaler.step(optimizer)在更新参数前会检查梯度里有没有inf/nan
            # 有就跳过这一步，避免污染模型；没有就正常更新
            # scaler.update() 根据检查结果动态调整缩放因子，保证FP16训练的稳定性
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        else:
            # CPU路径
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "poem_lstm.pth")

"""
生成代码改动有一些大
Top-K+重复惩罚+屏蔽UNK
"""

def generate(model, start_char, length=50, temperature=1.5,
             top_k=10, repetition_penalty=1.5):
    model.eval()
    chars = [start_char]
    generated_tokens = [char2idx.get(start_char, UNK_IDX)]
    input_idx = torch.tensor([[generated_tokens[0]]], device=device)  # 把字符转为索引作为输入
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_idx, hidden)
            logits = logits[:, -1, :] / temperature  # 取最后一个时间步

            # 屏蔽<UNK>，不让它出现在生成结果里
            logits[0, UNK_IDX] = -1e9

            # 重复惩罚
            for idx in set(generated_tokens):
                logits[0, idx] -= repetition_penalty

            # Top-K 采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)  # 返回概率最高的top_k个值的分数和索引
                probs = torch.softmax(top_k_logits, dim=-1)  # 再计算概率
                sampled = torch.multinomial(probs, 1).item()  # 从中随机抽取，这个sample是索引
                next_idx = top_k_indices[0, sampled].item()  
            else:
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()

            next_char = idx2char[next_idx]
            if next_char == EOP_TOKEN:
                break
            chars.append(next_char)
            generated_tokens.append(next_idx)
            input_idx = torch.tensor([[next_idx]], device=device)

    return "".join(chars)

print("生成示例：")
for start in ["<SOP>", "<SOP>", "<SOP>", "<SOP>"]:
    print(f"以 {start} 开头 ")
    print(generate(model, start, length=50, temperature=0.8))
    print()