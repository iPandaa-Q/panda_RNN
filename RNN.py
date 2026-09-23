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
def load_poems(json_dir, max_poems=50000):
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
            text = "".join(paragraphs).strip()
            text = text.replace("——", "，")
            text = re.sub(r"[a-zA-Z0-9\s]", "", text)
            if len(text) < 10 or len(text) > 200:
                continue
            poems.append(text)
            if len(poems) >= max_poems:
                return poems
    return poems
poems = load_poems(JSON_DIR, max_poems=50000)

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
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.2):
        super().__init__()
        # nn.Embedding 等价于：把 one-hot 向量乘以一个可学习的矩阵
        # padding_idx=PAD_IDX：告诉 Embedding 层 <PAD> 对应的向量永远是 0，且不参与梯度更新
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
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
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

EPOCHS = 80
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "poem_lstm.pth")

def generate(model, start_char, length=50, temperature=1.0):
    model.eval()
    chars = [start_char]
    input_idx = torch.tensor([[char2idx.get(start_char, UNK_IDX)]], device=device)  # 把字符转为索引作为输入
    hidden = None
    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_idx, hidden)
            logits = logits[:, -1, :] / temperature  # 取最后一个时间步
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx2char[next_idx]
            if next_char == EOP_TOKEN:
                break
            chars.append(next_char)
            input_idx = torch.tensor([[next_idx]], device=device)

    return "".join(chars)
print("生成示例：")
for start in ["春", "月", "山", "风"]:
    print(f"以 {start} 开头 ")
    print(generate(model, start, length=50, temperature=0.8))
    print()