import os
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Literal

# ======================== 模型结构（与训练时一致） =========================
# 请确保这些类定义与你训练时使用的完全一致
class SimpleCNENTokenizer:
    def __init__(self, vocab_size: int = 16000, min_freq: int = 1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self.token2id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
    def _tokenize_line(self, text: str) -> List[str]:
        text = str(text).strip()
        if not text: return []
        tokens = []
        buff = []
        def flush_buff():
            if buff:
                tokens.append("".join(buff))
                buff.clear()
        for ch in text:
            if ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf"):
                flush_buff()
                tokens.append(ch)
            elif ch.isspace():
                flush_buff()
            elif ch.isalnum() or ch in ["_", "-"]:
                buff.append(ch)
            else:
                flush_buff()
                tokens.append(ch)
        flush_buff()
        return tokens
    def encode(self, text: str, max_len: int = 1024) -> List[int]:
        toks = ["[CLS]"] + self._tokenize_line(text) + ["[SEP]"]
        ids = [self.token2id.get(tok, self.token2id["[UNK]"]) for tok in toks]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids
    @property
    def pad_id(self): return self.token2id["[PAD]"]

def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]
    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n_heads - closest_power_of_2]
        slopes += extra
    return torch.tensor(slopes).float()

class MultiheadSelfAttentionALiBi(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer("slopes", _get_alibi_slopes(num_heads).view(1, num_heads, 1, 1), persistent=False)
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, v, k = qkv[0], qkv[2], qkv[1]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        pos = torch.arange(T, device=x.device)
        rel = pos[None, :] - pos[:, None]
        rel = rel.unsqueeze(0).unsqueeze(0)
        alibi = self.slopes * rel
        attn_scores = attn_scores + alibi
        mask = attn_mask[:, None, None, :]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj_drop(self.out(out))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttentionALiBi(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x, attn_mask):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class TinyTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int = 16000, d_model: int = 128, num_layers: int = 2, num_heads: int = 4,
                 mlp_ratio: int = 4, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.max_len = max_len
    def forward(self, input_ids, attention_mask):
        x = self.token_emb(input_ids)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attention_mask)
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        return pooled

class TinyTransformerForClassification(nn.Module):
    def __init__(self, num_labels: int, vocab_size: int = 16000, d_model: int = 128, num_layers: int = 2,
                 num_heads: int = 4, mlp_ratio: int = 4, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.encoder = TinyTransformerEncoder(vocab_size, d_model, num_layers, num_heads, mlp_ratio, max_len, dropout)
        self.classifier = nn.Linear(d_model, num_labels)
    def forward(self, input_ids, attention_mask):
        pooled = self.encoder(input_ids, attention_mask)
        logits = self.classifier(pooled)
        return logits, pooled

def convert_to_onnx(
    model_pt_path: str,
    preprocessor_dir: str,
    onnx_path: str,
    max_len: int = 1024,
):
    """
    加载 PyTorch 模型，并将其转换为 ONNX 格式。
    """
    device = torch.device("cpu")
    le_path = os.path.join(preprocessor_dir, "label_encoder.pkl")
    tokenizer_path = os.path.join(preprocessor_dir, "tokenizer.pkl")

    if not os.path.exists(le_path) or not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"未找到预处理器文件。请确保已运行 `save_preprocessors.py`。")
    
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
    num_labels = len(le.classes_)
    print(f"LabelEncoder 已加载，类别数量为 {num_labels}。")
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.token2id)
    print(f"分词器已加载，词表大小为 {vocab_size}。")

    if max_len <= 0:
        raise ValueError("max_len 必须大于 0")
    
    model = TinyTransformerForClassification(
        num_labels=num_labels, vocab_size=vocab_size, d_model=128, num_layers=2,
        num_heads=4, mlp_ratio=4, max_len=max_len
    ).to(device)
    model = model.float()

    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"未找到 PyTorch 模型文件: {model_pt_path}")
    
    model.load_state_dict(torch.load(model_pt_path, map_location=device))
    model.eval()

    dummy_input_ids = torch.randint(0, vocab_size, (1, max_len)).to(device)
    dummy_attention_mask = torch.ones(1, max_len, dtype=torch.long).to(device)

    try:
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits', 'pooled_output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'seq_len'},
                'attention_mask': {0: 'batch_size', 1: 'seq_len'}
            }
        )
        print(f"\n模型成功转换为 ONNX 格式，并保存到: {onnx_path}")
        print("ONNX 模型输入: ['input_ids', 'attention_mask']")
        print("ONNX 模型输出: ['logits', 'pooled_output']")
    except Exception as e:
        print(f"模型转换失败: {e}")

if __name__ == "__main__":
    max_len = 1500
    model_pt_path = "test_model/result/tiny_student_state_dict.pt"
    preprocessor_folder = "preprocessors"
    onnx_path = "test_model/result/tiny_student_state_dict.onnx"
    
    try:
        convert_to_onnx(
            model_pt_path,
            preprocessor_folder,
            onnx_path,
            max_len=max_len,
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
