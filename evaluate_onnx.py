import json
import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


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
    def pad_id(self):
        return self.token2id["[PAD]"]

    def load_vocab(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)


def evaluate_onnx(
    onnx_path: str,
    test_csv_path: str,
    vocab_json_path: str,
    le_path: str,
    report_json_path: str,
    pred_csv_path: str,
    text_col: str,
    label_col: str,
    max_len: int,
    batch_size: int,
    use_gpu: bool,
) -> None:
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("ONNX evaluation requires onnxruntime.") from e

    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    if not os.path.exists(vocab_json_path):
        raise FileNotFoundError(f"Tokenizer vocab not found: {vocab_json_path}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder not found: {le_path}")

    tokenizer = SimpleCNENTokenizer()
    tokenizer.load_vocab(vocab_json_path)
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    df = pd.read_csv(test_csv_path)
    if text_col not in df.columns:
        raise ValueError(f"text_col `{text_col}` not found in CSV columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"label_col `{label_col}` not found in CSV columns: {list(df.columns)}")

    eval_df = df[[text_col, label_col]].dropna().copy()
    if eval_df.empty:
        raise ValueError("No valid rows after dropping NA in text/label columns.")

    texts = eval_df[text_col].astype(str).tolist()
    true_labels = eval_df[label_col].astype(str).tolist()

    encoded_rows = []
    for text in tqdm(texts, desc="Encoding test samples", unit="sample"):
        encoded_rows.append(tokenizer.encode(text, max_len=max_len))
    input_ids = np.asarray(encoded_rows, dtype=np.int64)
    attention_mask = (input_ids != tokenizer.pad_id).astype(np.int64)

    available_providers = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    active_providers = session.get_providers()
    print(f"ONNX providers: {active_providers}")
    y_pred_ids: List[int] = []

    for start in tqdm(range(0, len(input_ids), batch_size), desc="ONNX inference", unit="batch"):
        end = start + batch_size
        ort_inputs = {
            "input_ids": input_ids[start:end],
            "attention_mask": attention_mask[start:end],
        }
        outputs = session.run(["logits"], ort_inputs)
        batch_logits = outputs[0]
        batch_pred_ids = np.argmax(batch_logits, axis=1)
        y_pred_ids.extend(batch_pred_ids.tolist())

    y_pred_labels = label_encoder.inverse_transform(np.asarray(y_pred_ids, dtype=np.int64)).astype(str).tolist()

    labels_for_report = label_encoder.classes_.astype(str).tolist()
    acc = accuracy_score(true_labels, y_pred_labels)
    f1_macro = f1_score(true_labels, y_pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, y_pred_labels, average="weighted")
    report_dict = classification_report(
        true_labels,
        y_pred_labels,
        labels=labels_for_report,
        target_names=labels_for_report,
        output_dict=True,
    )

    eval_result = {
        "timestamp": datetime.now().isoformat(),
        "onnx_path": onnx_path,
        "test_csv_path": test_csv_path,
        "vocab_json_path": vocab_json_path,
        "le_path": le_path,
        "text_col": text_col,
        "label_col": label_col,
        "max_len": max_len,
        "batch_size": batch_size,
        "use_gpu": use_gpu,
        "active_providers": active_providers,
        "samples": int(len(true_labels)),
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
    out_df = eval_df.copy()
    out_df["predicted_label"] = y_pred_labels
    out_df["is_correct"] = (out_df[label_col].astype(str) == out_df["predicted_label"]).astype(int)
    out_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

    print(f"Evaluation finished. samples={len(true_labels)}")
    print(f"accuracy={acc:.6f}, f1_macro={f1_macro:.6f}, f1_weighted={f1_weighted:.6f}")
    print(f"Report JSON: {report_json_path}")
    print(f"Predictions CSV: {pred_csv_path}")


if __name__ == "__main__":
    # onnx_path = "tiny_student_state_dict_fp16.onnx"
    onnx_path = "tiny_student_state_dict_fp32_反量化_no_clip.onnx"
    # 评估直接使用 raw test CSV，不需要 npz。
    test_csv_path = "test_split_desample50.csv" # 改这个
    vocab_json_path = "soft_labels/vocab.json"
    le_path = "soft_labels/le.pkl"
    text_col = "MERGED_TEXT"
    label_col = "scene_label"
    max_len = 1500
    batch_size = 64
    use_gpu = True
    report_json_path = "onnx_eval_report.json"
    pred_csv_path = "onnx_eval_predictions.csv"

    evaluate_onnx(
        onnx_path=onnx_path,
        test_csv_path=test_csv_path,
        vocab_json_path=vocab_json_path,
        le_path=le_path,
        report_json_path=report_json_path,
        pred_csv_path=pred_csv_path,
        text_col=text_col,
        label_col=label_col,
        max_len=max_len,
        batch_size=batch_size,
        use_gpu=use_gpu,
    )
