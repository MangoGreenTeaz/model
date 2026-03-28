import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional, Any, Dict

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

    def load_vocab(self, path: str) -> None:
        import json

        with open(path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)

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

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _extract_tensor_from_state_value(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value.dequantize() if value.is_quantized else value

    if isinstance(value, dict):
        if "int8" in value and "scale" in value:
            int8_tensor = value["int8"]
            scale_tensor = value["scale"]
            if torch.is_tensor(int8_tensor) and torch.is_tensor(scale_tensor):
                return int8_tensor.float() * scale_tensor.float()

        preferred_keys = (
            "tensor",
            "value",
            "data",
            "param",
            "weight",
            "bias",
            "qweight",
        )
        for key in preferred_keys:
            if key in value:
                tensor = _extract_tensor_from_state_value(value[key])
                if tensor is not None:
                    return tensor

        for nested in value.values():
            tensor = _extract_tensor_from_state_value(nested)
            if tensor is not None:
                return tensor

    return None


def _normalize_state_dict(raw_checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("checkpoint 不是 dict，无法解析 state_dict。")

    candidates = (
        "state_dict",
        "model_state_dict",
        "student_state_dict",
        "model",
        "net",
    )
    state_obj: Any = raw_checkpoint
    for key in candidates:
        if key in raw_checkpoint and isinstance(raw_checkpoint[key], dict):
            state_obj = raw_checkpoint[key]
            break

    if not isinstance(state_obj, dict):
        raise ValueError("无法从 checkpoint 中定位 state_dict。")

    normalized: Dict[str, torch.Tensor] = {}
    for name, value in state_obj.items():
        tensor = _extract_tensor_from_state_value(value)
        if tensor is None:
            raise ValueError(f"参数 `{name}` 不是 Tensor，也无法从嵌套结构中提取 Tensor。")
        normalized[name] = tensor

    return normalized


def _infer_model_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    inferred: Dict[str, int] = {}

    emb_key = "encoder.token_emb.weight"
    if emb_key in state_dict and state_dict[emb_key].ndim == 2:
        inferred["vocab_size"] = int(state_dict[emb_key].shape[0])

    cls_w_key = "classifier.weight"
    if cls_w_key in state_dict and state_dict[cls_w_key].ndim == 2:
        inferred["num_labels"] = int(state_dict[cls_w_key].shape[0])

    return inferred


def convert_to_onnx(
    model_pt_path: str,
    vocab_json_path: str,
    onnx_path: str,
    max_len: int = 1024,
):
    """
    加载 PyTorch 模型，并将其转换为 ONNX 格式。
    """
    device = torch.device("cpu")

    if not os.path.exists(vocab_json_path):
        raise FileNotFoundError(f"未找到 vocab.json: {vocab_json_path}")
    
    total_steps = 4

    with tqdm(total=total_steps, desc="ONNX conversion pipeline", unit="step") as pbar:
        tokenizer = SimpleCNENTokenizer()
        tokenizer.load_vocab(vocab_json_path)
        vocab_size = len(tokenizer.token2id)
        print(f"词表已加载，词表大小为 {vocab_size}。")
        pbar.update(1)

        if max_len <= 0:
            raise ValueError("max_len 必须大于 0")

        if not os.path.exists(model_pt_path):
            raise FileNotFoundError(f"未找到 PyTorch 模型文件: {model_pt_path}")
        raw_checkpoint = torch.load(model_pt_path, map_location=device)
        state_dict = _normalize_state_dict(raw_checkpoint)
        inferred_dims = _infer_model_dims_from_state_dict(state_dict)

        effective_vocab_size = inferred_dims.get("vocab_size", vocab_size)
        effective_num_labels = inferred_dims.get("num_labels")
        if effective_num_labels is None:
            raise ValueError("无法从 checkpoint 推断 num_labels（缺少 classifier.weight）。")

        if effective_vocab_size != vocab_size:
            print(
                f"警告: tokenizer 词表({vocab_size}) 与 checkpoint 词表({effective_vocab_size}) 不一致，"
                f"将按 checkpoint 词表构建模型。"
            )
            if vocab_size > effective_vocab_size:
                raise ValueError(
                    "tokenizer 词表大于 checkpoint embedding 词表，推理时可能发生越界。"
                    "请使用与该 checkpoint 配套的 vocab.json。"
                )

        model = TinyTransformerForClassification(
            num_labels=effective_num_labels, vocab_size=effective_vocab_size, d_model=128, num_layers=2,
            num_heads=4, mlp_ratio=4, max_len=max_len
        ).to(device)
        model = model.float() # 默认 PyTorch 即导出 FP32
        pbar.update(1)

        model.load_state_dict(state_dict)
        model.eval()

        dummy_input_ids = torch.randint(0, effective_vocab_size, (1, max_len)).to(device)
        dummy_attention_mask = torch.ones(1, max_len, dtype=torch.long).to(device)
        pbar.update(1)

        try:
            _ensure_parent_dir(onnx_path)
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
            print(f"\n基础模型(FP32)成功转换为 ONNX 格式，暂存/保存到: {onnx_path}")
            print("ONNX 模型输入: ['input_ids', 'attention_mask']")
            print("ONNX 模型输出: ['logits', 'pooled_output']")
            pbar.update(1)
        except Exception as e:
            print(f"模型转换失败: {e}")

if __name__ == "__main__":
    max_len = 1500
    model_pt_path = "test_zl/260327_1500fp32/model/tiny_student_state_dict.pt"
    vocab_json_path = "test_zl/260327_1500fp32/soft_labels/vocab.json"
    onnx_path = "tiny_student_state_dict_fp32.onnx"
    
    try:
        convert_to_onnx(
            model_pt_path,
            vocab_json_path,
            onnx_path,
            max_len=max_len
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
