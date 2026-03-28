import os
import pickle
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from pt2onnx import TinyTransformerForClassification


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
        tokens: List[str] = []
        buff: List[str] = []

        def flush_buff() -> None:
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
    def pad_id(self) -> int:
        return self.token2id["[PAD]"]

    def load_vocab(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)


def _extract_tensor_from_state_value(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value.dequantize() if value.is_quantized else value

    if isinstance(value, dict):
        if "int8" in value and "scale" in value:
            int8_tensor = value["int8"]
            scale_tensor = value["scale"]
            if torch.is_tensor(int8_tensor) and torch.is_tensor(scale_tensor):
                return int8_tensor.float() * scale_tensor.float()

        preferred_keys = ("tensor", "value", "data", "param", "weight", "bias", "qweight")
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
        raise ValueError("checkpoint is not a dict")

    candidates = ("state_dict", "model_state_dict", "student_state_dict", "model", "net")
    state_obj: Any = raw_checkpoint
    for key in candidates:
        if key in raw_checkpoint and isinstance(raw_checkpoint[key], dict):
            state_obj = raw_checkpoint[key]
            break

    if not isinstance(state_obj, dict):
        raise ValueError("failed to locate state_dict in checkpoint")

    normalized: Dict[str, torch.Tensor] = {}
    for name, value in state_obj.items():
        tensor = _extract_tensor_from_state_value(value)
        if tensor is None:
            raise ValueError(f"parameter `{name}` does not contain tensor")
        normalized[name] = tensor

    return normalized


def _infer_model_dims(state_dict: Dict[str, torch.Tensor], fallback_vocab_size: int, fallback_num_labels: int) -> Dict[str, int]:
    emb = state_dict.get("encoder.token_emb.weight")
    cls_w = state_dict.get("classifier.weight")

    vocab_size = int(emb.shape[0]) if emb is not None and emb.ndim == 2 else fallback_vocab_size
    num_labels = int(cls_w.shape[0]) if cls_w is not None and cls_w.ndim == 2 else fallback_num_labels
    return {"vocab_size": vocab_size, "num_labels": num_labels}


def evaluate_student_pt(
    model_pt_path: str,
    vocab_json_path: str,
    le_path: str,
    test_csv_path: str,
    text_col: str,
    label_col: str,
    max_len: int,
    batch_size: int,
    use_gpu: bool,
    report_json_path: str,
    pred_csv_path: str,
) -> None:
    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"model checkpoint not found: {model_pt_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"test csv not found: {test_csv_path}")

    if not os.path.exists(vocab_json_path):
        raise FileNotFoundError(f"vocab not found: {vocab_json_path}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"label encoder not found: {le_path}")

    tokenizer = SimpleCNENTokenizer()
    tokenizer.load_vocab(vocab_json_path)
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    df = pd.read_csv(test_csv_path)
    if text_col not in df.columns:
        raise ValueError(f"text_col `{text_col}` not found in csv columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"label_col `{label_col}` not found in csv columns: {list(df.columns)}")

    text_series = pd.Series(df[text_col], index=df.index)
    label_series = pd.Series(df[label_col], index=df.index)
    eval_df = pd.DataFrame({text_col: text_series, label_col: label_series})
    metric_mask = pd.Series(pd.notna(text_series) & pd.notna(label_series), index=df.index)
    metric_df = eval_df.loc[metric_mask].copy()
    if metric_df.empty:
        raise ValueError("no valid rows after dropping NA")

    encoded_rows = []
    texts = df[text_col].fillna("").astype(str).tolist()
    for text in tqdm(texts, desc="Encoding test samples", unit="sample"):
        encoded_rows.append(tokenizer.encode(text, max_len=max_len))
    input_ids = np.asarray(encoded_rows, dtype=np.int64)
    attention_mask = (input_ids != tokenizer.pad_id).astype(np.int64)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw_checkpoint = torch.load(model_pt_path, map_location="cpu")
    state_dict = _normalize_state_dict(raw_checkpoint)

    fallback_vocab_size = int(len(tokenizer.token2id))
    fallback_num_labels = int(len(label_encoder.classes_))
    inferred = _infer_model_dims(state_dict, fallback_vocab_size, fallback_num_labels)

    if inferred["vocab_size"] != fallback_vocab_size:
        raise ValueError(
            f"tokenizer vocab ({fallback_vocab_size}) != checkpoint vocab ({inferred['vocab_size']}). "
            "请使用与该模型同源的 tokenizer。"
        )
    if inferred["num_labels"] != fallback_num_labels:
        print(
            f"Warning: label encoder classes ({fallback_num_labels}) != checkpoint classes ({inferred['num_labels']})."
        )

    model = TinyTransformerForClassification(
        num_labels=inferred["num_labels"],
        vocab_size=inferred["vocab_size"],
        d_model=128,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        max_len=max_len,
    ).to(device)
    model = model.float()
    model.load_state_dict(state_dict)
    model.eval()

    pred_ids: List[int] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(input_ids), batch_size), desc="PT inference", unit="batch"):
            end = start + batch_size
            batch_input_ids = torch.from_numpy(input_ids[start:end]).to(device)
            batch_attention_mask = torch.from_numpy(attention_mask[start:end]).to(device)
            logits, _ = model(batch_input_ids, batch_attention_mask)
            batch_pred = torch.argmax(logits, dim=1)
            pred_ids.extend(batch_pred.detach().cpu().tolist())

    classes = [str(x) for x in label_encoder.classes_]
    pred_labels: List[str] = []
    valid_pred_ids: List[int] = []
    for idx in pred_ids:
        if 0 <= idx < len(classes):
            pred_labels.append(classes[idx])
            valid_pred_ids.append(int(idx))
        else:
            pred_labels.append(f"__out_of_range_{idx}")
            valid_pred_ids.append(-1)

    pred_id_series = pd.Series(valid_pred_ids, index=df.index)
    pred_label_series = pd.Series(pred_labels, index=df.index)

    known_set = set(classes)
    metric_label_series = pd.Series(metric_df[label_col], index=metric_df.index)
    known_mask = metric_label_series.astype(str).isin(list(known_set))
    metric_df = metric_df.loc[known_mask].copy()
    if metric_df.empty:
        raise ValueError("测试集标签与当前 LabelEncoder 无交集，无法评估。")

    filtered_true_ids = label_encoder.transform(metric_label_series.loc[known_mask].astype(str).tolist()).tolist()
    filtered_pred_ids = pred_id_series.loc[metric_df.index].tolist()

    acc = accuracy_score(filtered_true_ids, filtered_pred_ids)
    f1_macro = f1_score(filtered_true_ids, filtered_pred_ids, average="macro")
    f1_weighted = f1_score(filtered_true_ids, filtered_pred_ids, average="weighted")
    report_dict = classification_report(
        filtered_true_ids,
        filtered_pred_ids,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
    )

    eval_result = {
        "timestamp": datetime.now().isoformat(),
        "model_pt_path": model_pt_path,
        "test_csv_path": test_csv_path,
        "vocab_json_path": vocab_json_path,
        "le_path": le_path,
        "text_col": text_col,
        "label_col": label_col,
        "max_len": max_len,
        "batch_size": batch_size,
        "use_gpu": use_gpu,
        "device": str(device),
        "samples": int(len(filtered_true_ids)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classification_report": report_dict,
    }

    report_dir = os.path.dirname(report_json_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    pred_dir = os.path.dirname(pred_csv_path)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
    out_df = df.copy()  # 使用原始的df，保留所有列
    out_df["predicted_label"] = pred_label_series.values
    out_df["is_correct"] = (out_df[label_col].astype(str) == out_df["predicted_label"]).map({True: "true", False: "false"})
    out_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

    print(f"Samples: {len(metric_df)}")
    print(f"Accuracy: {acc:.6f}")
    print(f"F1(macro): {f1_macro:.6f}, F1(weighted): {f1_weighted:.6f}")
    print(f"Report JSON: {report_json_path}")
    print(f"Predictions CSV: {pred_csv_path}")


if __name__ == "__main__":
    model_pt_path = "tiny_student_fp32_weights.pt"
    vocab_json_path = "soft_labels/vocab.json"
    le_path = "soft_labels/le.pkl"
    test_csv_path = "test_data.csv"  # 改这个
    text_col = "MERGED_TEXT"
    label_col = "scene_label"
    max_len = 1500
    batch_size = 64
    use_gpu = True
    report_json_path = "pt_eval_report.json"
    pred_csv_path = "pt_eval_predictions.csv"

    evaluate_student_pt(
        model_pt_path=model_pt_path,
        vocab_json_path=vocab_json_path,
        le_path=le_path,
        test_csv_path=test_csv_path,
        text_col=text_col,
        label_col=label_col,
        max_len=max_len,
        batch_size=batch_size,
        use_gpu=use_gpu,
        report_json_path=report_json_path,
        pred_csv_path=pred_csv_path,
    )
