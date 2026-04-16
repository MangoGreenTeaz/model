# pt2onnx.py 使用说明

## 1. 脚本作用

`pt2onnx.py` 用于将训练完成的学生模型权重（`.pt`）转换为 `ONNX` 格式，便于后续部署或推理使用。

它通常接在 `trainDistillation.py` 之后使用。

### 当前流程位置

- 上游输入：`trainDistillation.py` 生成的学生模型权重和配套 `vocab.json`
- 当前输出：ONNX 模型文件
- 下游步骤：使用 `deleteClip.py` 删除端侧不支持的 Clip 节点

---

## 2. 输入与导出说明

脚本需要两个输入文件：

- 学生模型权重，例如：`test_zl/260327_1500fp32/model/tiny_student_state_dict.pt`
- 对应词表文件，例如：`test_zl/260327_1500fp32/soft_labels/vocab.json`

这两个文件必须配套使用，建议来自同一次训练结果。

脚本导出后会生成一个 ONNX 文件，例如：

- `tiny_student_state_dict_fp32.onnx`

在整个部署链路中：

- `vocab.json`：后续端侧分词时，把输入文本转换成 `token id`
- `le.pkl`：虽然不是本脚本输入，但端侧部署时通常需要一并携带，用于把模型输出类别索引还原成标签名

---

## 3. 技术说明

脚本会重新构建与训练阶段一致的学生模型结构，再加载权重进行导出。

主要包含：

- 模型结构：`TinyTransformerForClassification`
- 分词器：`SimpleCNENTokenizer`
- 注意力机制：`ALiBi`
- 导出方式：`torch.onnx.export`

导出结果特点：

- 默认 `FP32`
- 支持动态 `batch_size`
- 支持动态 `seq_len`

ONNX 输入：

- `input_ids`
- `attention_mask`

ONNX 输出：

- `logits`
- `pooled_output`

---

## 4. 目录示例

```text
test_zl/
└─ 260327_1500fp32/
   ├─ model/
   │  └─ tiny_student_state_dict.pt
   └─ soft_labels/
      └─ vocab.json

pt2onnx.py
pt2onnx.md
tiny_student_state_dict_fp32.onnx
```

---

## 5. 使用步骤

### 步骤 1：确认训练产物已生成

先准备好：

- `.pt` 学生模型权重
- 配套的 `vocab.json`

### 步骤 2：修改脚本底部参数

默认示例：

```python
max_len = 1500
model_pt_path = "test_zl/260327_1500fp32/model/tiny_student_state_dict.pt"
vocab_json_path = "test_zl/260327_1500fp32/soft_labels/vocab.json"
onnx_path = "tiny_student_state_dict_fp32.onnx"
```

重点确认：

- `max_len`
- `model_pt_path`
- `vocab_json_path`
- `onnx_path`

建议 `max_len` 与训练时保持一致。

### 步骤 3：运行脚本

```bash
python pt2onnx.py
```

### 步骤 4：查看导出结果

成功后会得到 `.onnx` 文件，并在控制台看到输入输出名称与保存路径。

---

## 6. 注意事项

- 该脚本用于**导出学生模型**，不是训练脚本；
- 导出的不是 `BGE-M3` 本体，而是蒸馏后的学生模型；
- 模型结构定义必须与训练时一致，否则可能无法正确加载权重；
- `vocab.json` 必须与 `.pt` 权重配套使用；
- 如果词表和 checkpoint 不匹配，脚本可能报错；
- `max_len` 必须大于 0；
- 当前脚本默认在 `CPU` 上执行导出。
- 如果后续要部署到端侧，通常需要同时交付：`.onnx`、`vocab.json`、`le.pkl`。
