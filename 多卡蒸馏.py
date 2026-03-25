import os
import math
import time
import json
import random
from typing import List, Tuple, Optional, Sequence, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

# ======================== 全局配置 =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_GPU = torch.cuda.is_available() and torch.cuda.device_count() > 1

def _get_available_gpus() -> List[str]:
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

def _log_device_config():
    if torch.cuda.is_available():
        if MULTI_GPU:
            print(f"Multi-GPU mode: {torch.cuda.device_count()} devices -> {_get_available_gpus()}")
        else:
            print("Single GPU mode: cuda:0")
    else:
        print("CPU mode")

def _wrap_dataparallel(model: nn.Module) -> nn.Module:
    if MULTI_GPU:
        return nn.DataParallel(model)
    return model

def _state_dict_for_save(model: nn.Module) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()

def _load_state_dict_safely(model: nn.Module, state_dict: dict):
    model_keys = list(model.state_dict().keys())
    model_has_prefix = any(k.startswith("module.") for k in model_keys)
    state_has_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if model_has_prefix and not state_has_prefix:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif state_has_prefix and not model_has_prefix:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _encode_texts(embedder: Any, texts: List[str], batch_size: int, desc: str) -> np.ndarray:
    if MULTI_GPU:
        target_devices = _get_available_gpus()
        print(f"Multi-process encode on devices: {target_devices}")
        pool = embedder.start_multi_process_pool(target_devices=target_devices)
        try:
            chunk_size = max(1000, batch_size * 20)
            chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            embeddings_list = []
            for chunk in tqdm(chunks, desc=desc):
                emb = embedder.encode_multi_process(chunk, pool, batch_size=batch_size)
                embeddings_list.append(emb)
            if not embeddings_list:
                return np.empty((0, 0))
            return np.concatenate(embeddings_list, axis=0)
        finally:
            embedder.stop_multi_process_pool(pool)
    return embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)

# ======================== 简易分词器（中英文通用） =========================
# 规则：
# - 中文：逐字作为 token；
# - 英文/数字：按空白切分单词；
# - 特殊符号保留为单独 token；
# - 自动从训练数据构建词表，设定 vocab_size 上限；

class SimpleCNENTokenizer:
    def __init__(self, vocab_size: int = 16000, min_freq: int = 1):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self.token2id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def _tokenize_line(self, text: str) -> List[str]:
        text = str(text).strip()
        if not text:
            return []
        tokens = []
        buff = []
        def flush_buff():
            if buff:
                tokens.append("".join(buff))
                buff.clear()
        for ch in text:
            cat = ord(ch)
            if ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf"):  # CJK
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

    def build_vocab(self, texts: List[str]):
        from collections import Counter
        cnt = Counter()
        for t in tqdm(texts, desc="Building vocab"):
            cnt.update(self._tokenize_line(t))
        # 过滤低频并截断
        items = [tok for tok, c in cnt.items() if c >= self.min_freq]
        items = sorted(items, key=lambda x: (-cnt[x], x))
        keep = self.vocab_size - len(self.special_tokens)
        items = items[:max(0, keep)]
        # 组装词表
        self.token2id = {tok: i for i, tok in enumerate(self.special_tokens + items)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    @property
    def pad_id(self):
        return self.token2id["[PAD]"]

    @property
    def cls_id(self):
        return self.token2id["[CLS]"]

    @property
    def sep_id(self):
        return self.token2id["[SEP]"]

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = ["[CLS]"] + self._tokenize_line(text) + ["[SEP]"]
        ids = [self.token2id.get(tok, self.token2id["[UNK]"]) for tok in toks]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def save_vocab(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False)
            
    def load_vocab(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.token2id = json.load(f)
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)


# ======================== ALiBi 注意力偏置 =========================
# 参考: Press et al. 2021; 实现按 head 生成不同斜率的线性偏置。

def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    # 来自官方实现的近似，兼容任意 head 数
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
    return torch.tensor(slopes).float()  # [n_heads]

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
        q, v, k = qkv[0], qkv[2], qkv[1]  # [B, H, T, Dh]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,T]
        # ALiBi 偏置：-inf * (i<j) + slope * (j - i)
        # 构造位置差矩阵
        pos = torch.arange(T, device=x.device)
        rel = pos[None, :] - pos[:, None]  # [T,T]
        rel = rel.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
        alibi = self.slopes * rel  # [1,H,T,T]
        attn_scores = attn_scores + alibi
        # mask: 1 表示有效，0 表示 pad
        mask = attn_mask[:, None, None, :]  # [B,1,1,T]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [B,H,T,Dh]
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
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, input_ids, attention_mask):
        x = self.token_emb(input_ids)  # [B,T,C]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attention_mask)
        x = self.norm(x)
        # mean-pool（只对有效 token）
        mask = attention_mask.unsqueeze(-1)  # [B,T,1]
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1)
        pooled = summed / denom
        return pooled  # [B,C]

class TinyTransformerForClassification(nn.Module):
    def __init__(self, num_labels: int, vocab_size: int = 16000, d_model: int = 128, num_layers: int = 2,
                 num_heads: int = 4, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = TinyTransformerEncoder(vocab_size, d_model, num_layers, num_heads, mlp_ratio, dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        pooled = self.encoder(input_ids, attention_mask)
        logits = self.classifier(pooled)
        return logits, pooled

# ======================== 数据集封装（蒸馏用） =========================
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Any, tokenizer: SimpleCNENTokenizer, max_len: int,
                 teacher_logits: Optional[np.ndarray] = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = torch.tensor(self.tokenizer.encode(self.texts[idx], self.max_len), dtype=torch.long)
        # attention mask: 非PAD为1
        attn = (ids != self.tokenizer.pad_id).long()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        sample = {"input_ids": ids, "attention_mask": attn, "label": label}
        if self.teacher_logits is not None:
            sample["teacher_logits"] = torch.tensor(self.teacher_logits[idx], dtype=torch.float32)
        return sample

# ======================== 蒸馏损失（KL + CE 可切换） =========================
class DistillCriterion(nn.Module):
    def __init__(self, num_labels: int, use_ce: bool = True, kd_weight: float = 1.0, ce_weight: float = 1.0, T: float = 2.0):
        super().__init__()
        self.use_ce = use_ce
        self.kd_weight = kd_weight
        self.ce_weight = ce_weight
        self.T = T
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, labels, teacher_logits=None):
        loss = 0.0
        if teacher_logits is not None:
            # KL(student || teacher) with temperature
            s_log_probs = F.log_softmax(student_logits / self.T, dim=-1)
            t_probs = F.softmax(teacher_logits / self.T, dim=-1)
            kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (self.T ** 2)
            loss = loss + self.kd_weight * kl
        if self.use_ce:
            loss = loss + self.ce_weight * self.ce(student_logits, labels)
        return loss

# ======================== 教师：BGEM3 + 简单 MLP 分类头 =========================
# 为了与现有流程对齐：先用 BGEM3 抽取句向量，再训练一个 MLP 分类器作为教师，
# 然后用教师 logits 作为软标签去蒸馏学生 TinyTransformer。

from sentence_transformers import SentenceTransformer

class MLPTeacher(nn.Module):
    def __init__(self, in_dim: int, num_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_labels)
        )

    def forward(self, x):
        return self.net(x)

# ======================== 主流程（跑通版） =========================
class Runner:
    def __init__(self, train_csv_path: str, test_csv_path: str, val_csv_path: Optional[str] = None,
                 model_name_or_path: str = "model/bge-m3", max_len: Optional[int] = None,
                 vocab_size: int = 16000, batch_size: int = 32, soft_label_dir: str = "soft_labels_0",report_save_path: str = "report", model_save_path: str = "model",
                 data_mode: str = "separate", full_csv_path: Optional[str] = None,
                 full_train_ratio: float = 0.7, full_val_ratio: float = 0.2, full_test_ratio: float = 0.1,
                 full_shuffle: bool = True, full_split_save_dir: Optional[str] = None,
                 full_per_label_base_count: int = 10000):
        if max_len is None:
            raise ValueError("max_len 需要在入口处设置")
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.val_csv_path = val_csv_path
        self.model_name_or_path = model_name_or_path
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.soft_label_dir = soft_label_dir
        self.report_save_path = report_save_path
        self.model_save_path = model_save_path

        self.data_mode = data_mode
        self.full_csv_path = full_csv_path
        self.full_train_ratio = full_train_ratio
        self.full_val_ratio = full_val_ratio
        self.full_test_ratio = full_test_ratio
        self.full_shuffle = full_shuffle
        self.full_split_save_dir = full_split_save_dir
        self.full_per_label_base_count = full_per_label_base_count
        os.makedirs(self.soft_label_dir, exist_ok=True)
        os.makedirs(self.report_save_path, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        self.train_logits_path = os.path.join(self.soft_label_dir, "train_logits_split.npy")
        self.val_logits_path = os.path.join(self.soft_label_dir, "val_logits_split.npy")
        self.test_logits_path = os.path.join(self.soft_label_dir, "test_logits_split.npy")
        self.labels_path = os.path.join(self.soft_label_dir, "labels_split.npy")
        self.val_labels_path = os.path.join(self.soft_label_dir, "val_labels_split.npy")
        self.le_path = os.path.join(self.soft_label_dir, "le.pkl")
        self.vocab_path = os.path.join(self.soft_label_dir, "vocab.json")

    def _save_full_mode_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        if not self.full_split_save_dir:
            return
        os.makedirs(self.full_split_save_dir, exist_ok=True)
        train_path = os.path.join(self.full_split_save_dir, "train_split.csv")
        val_path = os.path.join(self.full_split_save_dir, "val_split.csv")
        test_path = os.path.join(self.full_split_save_dir, "test_split.csv")
        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
        print(f"full 模式切分数据已保存: {self.full_split_save_dir}")

    def _split_full_mode_per_label(self, df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ratio_sum = self.full_train_ratio + self.full_val_ratio + self.full_test_ratio
        if not np.isclose(ratio_sum, 1.0):
            raise ValueError("full 模式切分比例之和必须为 1.0")
        if self.full_train_ratio <= 0 or self.full_val_ratio <= 0 or self.full_test_ratio <= 0:
            raise ValueError("full 模式切分比例必须全部大于 0")
        if self.full_per_label_base_count <= 0:
            raise ValueError("full_per_label_base_count 必须大于 0")
        if 'scene_label' not in df_full.columns:
            raise ValueError("full 数据中缺少 scene_label 列")

        grouped = {}
        for idx, label in zip(df_full.index.tolist(), df_full['scene_label'].tolist()):
            grouped.setdefault(label, []).append(idx)

        base_n = self.full_per_label_base_count
        test_target = int(round(base_n * self.full_test_ratio))
        val_target = int(round(base_n * self.full_val_ratio))
        train_target = base_n - test_target - val_target
        if train_target < 0 or val_target < 0 or test_target < 0:
            raise ValueError("full 模式切分目标条数非法，请检查比例设置")

        tr_idx, va_idx, te_idx = [], [], []
        rng = random.Random(SEED)

        for label in sorted(grouped.keys()):
            label_indices = list(grouped[label])
            if self.full_shuffle:
                rng.shuffle(label_indices)

            available = len(label_indices)
            use_n = min(available, base_n)
            if available > base_n:
                print(f"警告：类别 {label} 样本数 {available} 超过 {base_n}，将仅使用前 {base_n} 条进行固定切分。")
            selected = label_indices[:use_n]

            # 优先保证 test 条数
            test_n = min(test_target, use_n)
            remain = use_n - test_n

            if use_n == base_n:
                # 达到基准条数时，严格按固定目标条数切分
                train_n = train_target
                val_n = val_target
            else:
                # 样本不足时，先满足 test，再按 train:val 比例分配剩余
                train_weight = self.full_train_ratio / (self.full_train_ratio + self.full_val_ratio)
                train_n = int(round(remain * train_weight))
                train_n = max(0, min(train_n, remain))
                val_n = remain - train_n

            test_items = selected[:test_n]
            train_items = selected[test_n:test_n + train_n]
            val_items = selected[test_n + train_n:test_n + train_n + val_n]

            te_idx.extend(test_items)
            tr_idx.extend(train_items)
            va_idx.extend(val_items)

            print(f"[full-fixed] label={label} total={available} used={use_n} -> train={len(train_items)} val={len(val_items)} test={len(test_items)}")

        if self.full_shuffle:
            rng.shuffle(tr_idx)
            rng.shuffle(va_idx)
            rng.shuffle(te_idx)

        train_df = df_full.loc[tr_idx].copy().reset_index(drop=True)
        val_df = df_full.loc[va_idx].copy().reset_index(drop=True)
        test_df = df_full.loc[te_idx].copy().reset_index(drop=True)
        return train_df, val_df, test_df

    def load_data(self):
        if self.data_mode not in ("separate", "full"):
            raise ValueError("data_mode 仅支持: separate, full")

        if self.data_mode == "separate":
            if not self.val_csv_path:
                raise ValueError("data_mode=separate 时必须提供 val_csv_path（验证集路径）")

            # 1. 读取训练集
            print(f"Loading train data from {self.train_csv_path}")
            df_train = pd.read_csv(self.train_csv_path)
            if 'scene_label' not in df_train.columns or 'MERGED_TEXT' not in df_train.columns:
                raise ValueError("Train CSV 必须包含 scene_label 与 MERGED_TEXT 列")
            df_train = df_train.dropna(subset=['scene_label', 'MERGED_TEXT'])
            self.X_train_texts = df_train['MERGED_TEXT'].astype(str).tolist()
            train_labels = df_train['scene_label'].astype(str).tolist()

            # 2. 读取验证集
            print(f"Loading val data from {self.val_csv_path}")
            df_val = pd.read_csv(self.val_csv_path)
            if 'scene_label' not in df_val.columns or 'MERGED_TEXT' not in df_val.columns:
                raise ValueError("Val CSV 必须包含 scene_label 与 MERGED_TEXT 列")
            df_val = df_val.dropna(subset=['scene_label', 'MERGED_TEXT'])
            self.X_val_texts = df_val['MERGED_TEXT'].astype(str).tolist()
            val_labels = df_val['scene_label'].astype(str).tolist()

            # 3. 读取测试集
            print(f"Loading test data from {self.test_csv_path}")
            df_test = pd.read_csv(self.test_csv_path)
            if 'scene_label' not in df_test.columns or 'MERGED_TEXT' not in df_test.columns:
                raise ValueError("Test CSV 必须包含 scene_label 与 MERGED_TEXT 列")
            df_test = df_test.dropna(subset=['scene_label', 'MERGED_TEXT'])
            self.X_test_texts = df_test['MERGED_TEXT'].astype(str).tolist()
            test_labels = df_test['scene_label'].astype(str).tolist()
        else:
            if not self.full_csv_path:
                raise ValueError("data_mode=full 时必须提供 full_csv_path")
            print(f"Loading full data from {self.full_csv_path}")
            df_full = pd.read_csv(self.full_csv_path)
            if 'scene_label' not in df_full.columns or 'MERGED_TEXT' not in df_full.columns:
                raise ValueError("Full CSV 必须包含 label 与 MERGED_TEXT 列")
            df_full = df_full.dropna(subset=['scene_label', 'MERGED_TEXT']).copy()
            df_full['MERGED_TEXT'] = df_full['MERGED_TEXT'].astype(str)
            df_full['scene_label'] = df_full['scene_label'].astype(str)

            train_df, val_df, test_df = self._split_full_mode_per_label(df_full)
            self.full_train_df = train_df
            self.full_val_df = val_df
            self.full_test_df = test_df

            self.X_train_texts = train_df['MERGED_TEXT'].tolist()
            self.X_val_texts = val_df['MERGED_TEXT'].tolist()
            self.X_test_texts = test_df['MERGED_TEXT'].tolist()
            train_labels = train_df['scene_label'].tolist()
            val_labels = val_df['scene_label'].tolist()
            test_labels = test_df['scene_label'].tolist()

        # 3. 处理标签编码
        # 如果存在 LabelEncoder 则加载，否则新建（保证增训时的标签一致性）
        le_loaded = False
        if os.path.exists(self.le_path):
            print(f"Loading existing LabelEncoder from {self.le_path}")
            import pickle
            try:
                with open(self.le_path, 'rb') as f:
                    self.le = pickle.load(f)
                le_loaded = True
            except Exception as e:
                print(f"警告：LabelEncoder 加载失败 ({e})，将重新构建。")

        if le_loaded:
            # 处理新出现的标签
            classes = list(getattr(self.le, "classes_", []))
            if not classes:
                raise ValueError("LabelEncoder classes_ 未初始化")
            known_labels = set(classes)
            
            # 过滤掉未知标签 (Train)
            mask_tr = [l in known_labels for l in train_labels]
            if not all(mask_tr):
                self.X_train_texts = [t for t, m in zip(self.X_train_texts, mask_tr) if m]
                train_labels = [l for l, m in zip(train_labels, mask_tr) if m]
                if self.data_mode == "full":
                    self.full_train_df = self.full_train_df.loc[mask_tr].reset_index(drop=True)
            self.y_train = self.le.transform(train_labels)

            # 过滤掉未知标签 (Val)
            mask_val = [l in known_labels for l in val_labels]
            if not all(mask_val):
                print(f"警告：验证集中发现 {len(mask_val) - sum(mask_val)} 条数据的标签不在旧 LabelEncoder 中，将被忽略。")
                self.X_val_texts = [t for t, m in zip(self.X_val_texts, mask_val) if m]
                val_labels = [l for l, m in zip(val_labels, mask_val) if m]
                if self.data_mode == "full":
                    self.full_val_df = self.full_val_df.loc[mask_val].reset_index(drop=True)
            self.y_val = self.le.transform(val_labels)

            # 过滤掉未知标签 (Test)
            mask_te = [l in known_labels for l in test_labels]
            if not all(mask_te):
                print(f"警告：测试集中发现 {len(mask_te) - sum(mask_te)} 条数据的标签不在旧 LabelEncoder 中，将被忽略。")
                self.X_test_texts = [t for t, m in zip(self.X_test_texts, mask_te) if m]
                test_labels = [l for l, m in zip(test_labels, mask_te) if m]
                if self.data_mode == "full":
                    self.full_test_df = self.full_test_df.loc[mask_te].reset_index(drop=True)
            self.y_test = self.le.transform(test_labels)

        else:
            self.le = LabelEncoder()
            # 所有的标签一起 fit
            # 所有的标签一起 fit
            all_labels = train_labels + val_labels + test_labels
            self.le.fit(all_labels)
            self.y_train = self.le.transform(train_labels)
            self.y_val = self.le.transform(val_labels)
            self.y_test = self.le.transform(test_labels)

        if self.data_mode == "full":
            self._save_full_mode_splits(
                self.full_train_df,
                self.full_val_df,
                self.full_test_df,
            )
            
        classes = list(getattr(self.le, "classes_", []))
        if not classes:
            raise ValueError("LabelEncoder classes_ 未初始化")
        self.num_labels = len(classes)
        print(f"数据载入完成：训练 {len(self.X_train_texts)}，验证 {len(self.X_val_texts)}，测试 {len(self.X_test_texts)}，类别数 {self.num_labels}")

    def train_teacher_and_save_logits(self, epochs: int = 50, lr: float = 1e-3, batch_size: int = 64):
        # 检查是否已存在 soft labels
        if os.path.exists(self.train_logits_path) and os.path.exists(self.test_logits_path) and os.path.exists(self.val_logits_path):
            print("Soft labels 已存在，跳过教师模型训练。")
            self.kd_train_logits = np.load(self.train_logits_path)
            self.kd_val_logits = np.load(self.val_logits_path)
            self.kd_test_logits = np.load(self.test_logits_path)
            self.y_train = np.load(self.labels_path).tolist()
            self.y_val = np.load(self.val_labels_path).tolist()
            # 重新加载 LabelEncoder
            import pickle
            try:
                with open(self.le_path, 'rb') as f:
                    self.le = pickle.load(f)
                self.num_labels = len(self.le.classes_)
                return
            except Exception as e:
                print(f"警告：Soft labels 存在但 LabelEncoder 加载失败 ({e})，将重新进行教师模型流程。")
                # 后面会自动走到 teacher 训练流程


        print("Soft labels 不存在，开始训练教师模型并生成 soft labels。")

        # 1) 提取 BGEM3 句向量
        embedder_device = "cpu" if MULTI_GPU else device
        embedder = SentenceTransformer(self.model_name_or_path, device=embedder_device)
        if self.max_len:
            embedder.max_seq_length = self.max_len
            
        print("提取 BGEM3 向量 (train)...")
        # 显存优化：降低 batch_size 防止 OOM
        encode_bs = 8
        emb_train = _encode_texts(embedder, self.X_train_texts, encode_bs, desc="Encoding train")
        print("提取 BGEM3 向量 (val)...")
        emb_val = _encode_texts(embedder, self.X_val_texts, encode_bs, desc="Encoding val")
        print("提取 BGEM3 向量 (test)...")
        emb_test = _encode_texts(embedder, self.X_test_texts, encode_bs, desc="Encoding test")
        self.emb_dim = emb_train.shape[1]

        Xtr = torch.tensor(emb_train, dtype=torch.float32)
        Ytr = torch.tensor(self.y_train, dtype=torch.long)
        Xval = torch.tensor(emb_val, dtype=torch.float32)
        Yval = torch.tensor(self.y_val, dtype=torch.long)
        Xte = torch.tensor(emb_test, dtype=torch.float32)
        Yte = torch.tensor(self.y_test, dtype=torch.long)

        # 2) 教师 MLP 分类器
        teacher = MLPTeacher(self.emb_dim, self.num_labels).to(device)
        teacher = _wrap_dataparallel(teacher)
        opt = torch.optim.Adam(teacher.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            teacher.train()
            loss_sum, correct, total = 0.0, 0, 0
            for bx, by in loader:
                bx = bx.to(device)
                by = by.to(device)
                opt.zero_grad()
                logits = teacher(bx)
                loss = crit(logits, by)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * bx.size(0)
                correct += (logits.argmax(-1) == by).sum().item()
                total += bx.size(0)
            print(f"[Teacher] Epoch {ep+1}/{epochs} loss={loss_sum/total:.4f} acc={correct/total:.4f}")

        # 3) 教师评估
        teacher.eval()
        with torch.no_grad():
            logits_tr = teacher(Xtr.to(device))
            logits_val = teacher(Xval.to(device))
            logits_te = teacher(Xte.to(device))
            
            val_preds = logits_val.argmax(-1)
            val_acc = (val_preds == Yval.to(device)).float().mean().item()

            preds = logits_te.argmax(-1)
            te_acc = (preds == Yte.to(device)).float().mean().item()
        print(f"[Teacher] Val acc={val_acc:.4f}, Test acc={te_acc:.4f}")

        # 生成并保存教师模型分类报告
        teacher_preds = preds.cpu().numpy()
        teacher_labels = Yte.cpu().numpy()
        report = classification_report(teacher_labels, teacher_preds, target_names=self.le.classes_, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        teacher_report_path = os.path.join(self.report_save_path, "teacher_cls_report.csv")
        _ensure_parent_dir(teacher_report_path)
        df_report.to_csv(teacher_report_path)
        print(f"教师模型分类报告: {teacher_report_path}")

        # 保存教师模型权重
        teacher_model_path = os.path.join(self.model_save_path, "teacher_model_state_dict.pt")
        _ensure_parent_dir(teacher_model_path)
        torch.save(_state_dict_for_save(teacher), teacher_model_path)
        print(f"教师模型权重已保存: {teacher_model_path}")

        # 4) 保存 soft labels 和其他必要信息
        self.kd_train_logits = logits_tr.cpu().numpy()
        self.kd_val_logits = logits_val.cpu().numpy()
        self.kd_test_logits = logits_te.cpu().numpy()
        np.save(self.train_logits_path, self.kd_train_logits)
        np.save(self.val_logits_path, self.kd_val_logits)
        np.save(self.test_logits_path, self.kd_test_logits)
        np.save(self.labels_path, self.y_train)
        np.save(self.val_labels_path, self.y_val)
        
        # 仅在 LabelEncoder 不存在或需要更新时保存(这里简单覆盖)
        import pickle
        with open(self.le_path, 'wb') as f:
            pickle.dump(self.le, f)
            
        print("Soft labels 和数据已保存。")
        # 清除教师模型，节省显存
        del teacher, embedder, Xtr, Ytr, Xval, Yval, Xte, Yte
        if device.type == 'cuda':
            torch.cuda.empty_cache()


    def train_student_with_logits(self, epochs: int = 40, lr: float = 2e-4, batch_size: int = 64,
                                 kd_weight: float = 1.0, ce_weight: float = 1.0, T: float = 2.0,
                                 d_model: int = 128, num_layers: int = 2, num_heads: int = 4,
                                 mlp_ratio: int = 4, resume_checkpoint: Optional[str] = None):
        # 确保 soft labels 已加载
        if not hasattr(self, 'kd_train_logits'):
            print("Soft labels 未加载，正在重新加载数据和logits。")
            self.load_data()
            self.train_teacher_and_save_logits()

        self.build_tokenizer()

        train_ds = TextDataset(self.X_train_texts, self.y_train, self.tokenizer, self.max_len, self.kd_train_logits)
        val_ds = TextDataset(self.X_val_texts, self.y_val, self.tokenizer, self.max_len, teacher_logits=None) # Val usually hard label
        test_ds = TextDataset(self.X_test_texts, self.y_test, self.tokenizer, self.max_len, self.kd_test_logits)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        student = TinyTransformerForClassification(
            num_labels=self.num_labels, vocab_size=len(self.tokenizer.token2id), d_model=d_model,
            num_layers=num_layers, num_heads=num_heads, mlp_ratio=mlp_ratio
        ).to(device)
        student = _wrap_dataparallel(student)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"正在从 {resume_checkpoint} 加载权重进行增训...")
            # 注意：需根据保存时的格式调整 key，这里假设加载的是 state_dict
            state_dict = torch.load(resume_checkpoint, map_location=device)
            _load_state_dict_safely(student, state_dict)

        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
        criterion = DistillCriterion(self.num_labels, use_ce=True, kd_weight=kd_weight, ce_weight=ce_weight, T=T)

        print("[Student] 开始训练（蒸馏）...")
        print("[Student] 开始训练（蒸馏）...")
        for ep in range(epochs):
            student.train()
            loss_sum, total = 0.0, 0
            correct = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                t_logits = batch["teacher_logits"].to(device)
                optimizer.zero_grad()
                s_logits, _ = student(input_ids, attn)
                loss = criterion(s_logits, labels, t_logits)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * input_ids.size(0)
                total += input_ids.size(0)
                correct += (s_logits.argmax(-1) == labels).sum().item()
            
            # Val Loop
            student.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    v_ids = vbatch["input_ids"].to(device)
                    v_attn = vbatch["attention_mask"].to(device)
                    v_labels = vbatch["label"].to(device)
                    v_logits, _ = student(v_ids, v_attn)
                    val_correct += (v_logits.argmax(-1) == v_labels).sum().item()
                    val_total += v_ids.size(0)
            val_acc = val_correct / val_total
            print(f"[Student] Epoch {ep+1}/{epochs} loss={loss_sum/total:.4f} train_acc={correct/total:.4f} val_acc={val_acc:.4f}")

        self.student = student
        self.test_ds = test_ds

    def build_tokenizer(self):
        self.tokenizer = SimpleCNENTokenizer(vocab_size=self.vocab_size)
        if os.path.exists(self.vocab_path):
            print(f"Loading existing vocab from {self.vocab_path}")
            self.tokenizer.load_vocab(self.vocab_path)
        else:
            # 基于全部文本构建词表，确保评测一致
            all_texts = self.X_train_texts + self.X_val_texts + self.X_test_texts
            self.tokenizer.build_vocab(all_texts)
            # 保存词表
            self.tokenizer.save_vocab(self.vocab_path)
            print(f"Vocab saved to {self.vocab_path}")

    def evaluate(self, model_save_dir: str = "test/result/model", report_save_dir: str = "test/result/report", save_precision: str = "fp32"):
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(report_save_dir, exist_ok=True)
        
        # 1) 分类指标
        loader = DataLoader(self.test_ds, batch_size=64)
        self.student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits, _ = self.student(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                preds = logits.argmax(-1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy().tolist())
        report = classification_report(all_labels, all_preds, target_names=self.le.classes_, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        cls_report_path = os.path.join(report_save_dir, "tiny_student_cls_report.csv")
        df_report.to_csv(cls_report_path)
        print(f"学生模型分类报告: {cls_report_path}")

        # 2) 模型大小（权重文件）
        precision = str(save_precision).lower()
        if precision not in ("fp32", "fp16", "int8"):
            raise ValueError("save_precision 仅支持: fp32, fp16, int8")

        model_path = None
        size_mb = None
        q_path = None
        q_size_mb = None

        if precision in ("fp32", "fp16"):
            model_path = os.path.join(model_save_dir, f"tiny_student_state_dict_{precision}.pt")
            state = {}
            for k, v in _state_dict_for_save(self.student).items():
                if v.dtype in (torch.float32, torch.bfloat16, torch.float16):
                    state[k] = v.float() if precision == "fp32" else v.half()
                else:
                    state[k] = v
            torch.save(state, model_path)
            print(f"学生模型权重({precision})已保存 -> {model_path}")



        # 4) 导出 int8 量化权重（可选）
        if precision == "int8":
            q_path = os.path.join(model_save_dir, "tiny_student_int8_weights.pt")
            q_state = {}
            for k, v in _state_dict_for_save(self.student).items():
                if v.dtype in (torch.float16, torch.float32, torch.bfloat16):
                    scale = v.abs().max() / 127.0 + 1e-8
                    q = torch.clamp((v/scale).round(), -128, 127).to(torch.int8)
                    q_state[k] = {"int8": q.cpu(), "scale": scale.cpu()}
                else:
                    q_state[k] = {"tensor": v.cpu()}
            torch.save(q_state, q_path)
            print(f"学生模型权重(int8)已保存 -> {q_path}")

        return {
            "report_csv": cls_report_path,
            "precision": precision,
            "fp_state_dict": model_path,
            "int8_state_dict": q_path
        }

# ======================== 入口（示例） =========================
if __name__ == "__main__":
    # 配置
    data_mode = "separate"  # separate=训练/验证/测试分开读入, full=整表切分
    train_file_path = "data/train_split.csv" # 训练集路径
    val_file_path = "data/val_split.csv"     # 验证集路径
    test_file_path = "data/test_split.csv"    # 测试集路径
    full_file_path = "data/0205_训练集_desample10000.csv" # 整表路径（仅 data_mode=full）
    
    max_len = 1500
    precision_choice = 1  # 1=fp32, 2=fp16, 3=int8
    precision_map = {1: "fp32", 2: "fp16", 3: "int8"}
    if precision_choice not in precision_map:
        raise ValueError("precision_choice 仅支持: 1(fp32), 2(fp16), 3(int8)")
    save_precision = precision_map[precision_choice]

    soft_label_dir = "test_zl/260313_1500fp32/soft_labels" # soft_label 保存路径
    dataset_save_dir = "test_zl/260313_1500fp32/dataset"  # full 模式切分数据保存路径
    model_save_path = "test_zl/260313_1500fp32/model"
    report_save_path = "test_zl/260313_1500fp32/report"
    full_train_ratio = 0.8
    full_val_ratio = 0.1
    full_test_ratio = 0.1
    full_shuffle = True
    full_per_label_base_count = 10000

    _log_device_config()
    runner = Runner(
        train_csv_path=train_file_path,
        test_csv_path=test_file_path,
        val_csv_path=val_file_path,
        model_name_or_path="model/bge-m3",
        max_len=max_len,
        vocab_size=16000,
        soft_label_dir=soft_label_dir,
        report_save_path=report_save_path,
        model_save_path=model_save_path,
        data_mode=data_mode,
        full_csv_path=full_file_path,
        full_train_ratio=full_train_ratio,
        full_val_ratio=full_val_ratio,
        full_test_ratio=full_test_ratio,
        full_shuffle=full_shuffle,
        full_split_save_dir=dataset_save_dir,
        full_per_label_base_count=full_per_label_base_count,
    )

    # 1) 载入及处理 (不再自动 split，而是分别读取)
    runner.load_data()

    # 2) 训练教师并保存 soft labels
    runner.train_teacher_and_save_logits(epochs=80, lr=1e-3, batch_size=64)

    # 3) 训练学生（Tiny Transformer 蒸馏）
    # 如果通过 resume_checkpoint 传入路径，则进行增训
    runner.train_student_with_logits(
        epochs=50, lr=2e-4, batch_size=64, kd_weight=1.0, ce_weight=1.0, T=2.0,
        d_model=128, num_layers=2, num_heads=4, mlp_ratio=4,
        # resume_checkpoint="test_model/result/tiny_student_state_dict_fp32.pt" # 取消注释以进行增训 测试
    )

    # 4) 评估与导出（分类报告）
    metrics = runner.evaluate(model_save_dir=model_save_path, report_save_dir=report_save_path, save_precision=save_precision)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
