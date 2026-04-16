# trainDistillation.py 使用说明

## 1. 脚本作用

`trainDistillation.py` 是一个文本分类蒸馏训练脚本，用来完成以下流程：

1. 使用本地 `BGE-M3` 提取文本向量；
2. 训练教师模型并生成 soft labels；
3. 训练轻量级学生模型；
4. 输出模型权重和分类报告。

适用数据为：**已经完成预处理和 merge 流程后的 CSV 文件**。

### 当前流程位置

- 上游输入：标注并完成 merge 的文本数据
- 当前输出：教师模型结果、学生模型权重、`soft labels`、`vocab.json`、`le.pkl`、分类报告
- 下游步骤：使用 `pt2onnx.py` 将学生模型导出为 ONNX

---

## 2. 技术与数据要求

### 使用的主要技术

- 教师模型：`BGE-M3` 向量 + `MLP` 分类头
- 学生模型：自定义轻量级 `Transformer`
- 蒸馏损失：`KL Divergence + CrossEntropyLoss`
- 分词器：`SimpleCNENTokenizer`
- 注意力机制：`ALiBi`
- 运行方式：支持 `CPU / 单卡 GPU / 多卡 GPU`

### 输入数据要求

输入文件必须是 CSV，并至少包含两列：

- `scene_label`：分类标签
- `MERGED_TEXT`：训练文本

建议使用 `UTF-8` 或 `UTF-8-SIG` 编码。脚本会自动丢弃这两列为空的记录。

### 本地模型要求

脚本默认从本地加载：

```python
model_name_or_path = "model/bge-m3"
```

`BGE-M3` 需要先从 Hugging Face 下载到本地目录，再运行脚本。

---

## 3. 目录与输出说明

典型目录示例：

```text
model/
└─ bge-m3/

data/
├─ 0205_训练集_desample10000.csv
├─ 0205_验证集.csv
└─ 0205_测试集.csv

test_zl/
└─ 260303_1500fp32/
   ├─ dataset/
   ├─ model/
   ├─ report/
   └─ soft_labels/

trainDistillation.py
trainDistillation.md
```

运行后通常会生成：

- `soft_labels/train_logits_split.npy`
- `soft_labels/val_logits_split.npy`（有验证集时）
- `soft_labels/test_logits_split.npy`
- `soft_labels/labels_split.npy`
- `soft_labels/val_labels_split.npy`（有验证集时）
- `soft_labels/le.pkl`
- `soft_labels/vocab.json`
- `model/teacher_model_state_dict.pt`
- `model/tiny_student_state_dict.pt`
- `report/teacher_cls_report.csv`
- `report/tiny_student_cls_report.csv`
- `dataset/train_split.csv`、`val_split.csv`、`test_split.csv`（仅 `full` 模式）

其中与后续部署关系最密切的文件是：

- `model/tiny_student_state_dict.pt`：后续给 `pt2onnx.py` 用于导出 ONNX
- `soft_labels/vocab.json`：端侧将输入文本转换成 `token id` 时使用
- `soft_labels/le.pkl`：端侧将模型输出的类别索引还原成标签名时使用

---

## 4. 两种数据模式

### `separate`

适合已经准备好训练集、验证集、测试集三个文件的情况：

```python
data_mode = "separate"
train_file_path = "data/0205_训练集_desample10000.csv"
val_file_path = "data/0205_验证集.csv"
test_file_path = "data/0205_测试集.csv"
```

### `full`

适合只有一张总表、需要脚本自动切分数据的情况：

```python
data_mode = "full"
full_file_path = "data/0205_训练集_desample10000.csv"
full_train_ratio = 0.8
full_val_ratio = 0.1
full_test_ratio = 0.1
full_shuffle = True
full_per_label_base_count = 10000
```

---

## 5. 使用步骤

### 步骤 1：准备模型和数据

- 将 Hugging Face 下载的 `BGE-M3` 放到本地 `model/bge-m3`
- 准备好 merge 后的 CSV 数据

### 步骤 2：修改脚本底部参数

重点确认：

- `data_mode`
- `train_file_path / val_file_path / test_file_path` 或 `full_file_path`
- `max_len`
- `base_output_dir`

其中：

```python
max_len = 1500
```

`max_len` **必须显式设置**，否则脚本会报错。

### 步骤 3：运行脚本

```bash
python trainDistillation.py
```

### 步骤 4：查看结果

训练完成后，在 `base_output_dir` 下查看：

- `soft_labels/`
- `model/`
- `report/`
- `dataset/`（仅 `full` 模式）

---

## 6. 注意事项

- 该脚本是**蒸馏训练脚本**，不是通用推理脚本；
- 学生模型是自定义轻量 `Transformer`，不是直接导出 `BGE-M3`；
- 输入数据必须包含 `scene_label` 和 `MERGED_TEXT`；
- `BGE-M3` 需要提前下载到本地；
- 如果已存在 soft labels，脚本会跳过教师模型训练，直接复用历史结果；
- 建议每次实验使用新的 `base_output_dir`，避免覆盖旧结果。
- 如果后续要部署到端侧，建议妥善保留 `tiny_student_state_dict.pt`、`vocab.json` 和 `le.pkl`。
