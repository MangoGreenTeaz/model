# deleteClip.py 使用说明

## 1. 脚本作用

`deleteClip.py` 用于对 ONNX 模型做后处理：**删除模型中的指定 Clip 节点**。

这样做是因为目标端侧环境**不支持 Clip 算子**，所以需要在部署前把它从 ONNX 图中移除。

这个脚本通常接在 `pt2onnx.py` 之后使用。

### 当前流程位置

- 上游输入：`pt2onnx.py` 导出的 ONNX 模型
- 当前输出：删除 Clip 节点后的 ONNX 模型
- 下游步骤：将 ONNX 文件部署到端侧，并同时提供 `vocab.json` 与 `le.pkl`

---

## 2. 输入与输出

输入文件：

- 原始 ONNX 模型，例如：`tiny_student_state_dict_fp32.onnx`

输出文件：

- 删除 Clip 节点后的 ONNX 模型，例如：`tiny_student_state_dict_fp32_no_clip.onnx`

部署到端侧时，通常还需要同时准备：

- `vocab.json`：把端侧输入文本转换成 `token id`
- `le.pkl`：把模型输出的类别索引还原成标签名

当前脚本默认处理的目标节点输出名为：

```text
/encoder/Clip_output_0
```

---

## 3. 脚本处理逻辑

脚本会：

1. 加载 ONNX 模型；
2. 查找目标 Clip 节点；
3. 记录 Clip 的输入；
4. 将后继节点改为直接连接 Clip 的输入；
5. 删除 Clip 节点；
6. 检查并保存修改后的 ONNX 模型。

简单理解：就是把 Clip 节点从图里摘掉，并重新接好前后节点。

---

## 4. 目录示例

```text
pt2onnx.py
pt2onnx.md
deleteClip.py
deleteClip.md

tiny_student_state_dict_fp32.onnx
tiny_student_state_dict_fp32_no_clip.onnx
```

---

## 5. 使用步骤

### 步骤 1：先准备 ONNX 模型

通常先通过 `pt2onnx.py` 导出：

```text
tiny_student_state_dict_fp32.onnx
```

### 步骤 2：确认目标 Clip 节点

脚本当前固定删除：

```python
target_node_output = "/encoder/Clip_output_0"
```

如果你的模型节点名不同，需要先改这行。

### 步骤 3：确认输入输出路径

脚本底部默认调用：

```python
remove_clip_node('tiny_student_state_dict_fp32.onnx', 'tiny_student_state_dict_fp32_no_clip.onnx')
```

如有需要，改成你自己的路径。

### 步骤 4：运行脚本

```bash
python deleteClip.py
```

### 步骤 5：查看结果

成功后会得到新的 ONNX 文件：

```text
tiny_student_state_dict_fp32_no_clip.onnx
```

---

## 6. 注意事项

- 当前脚本只处理**一个固定的 Clip 节点**，不是通用删除所有 Clip；
- 节点通过输出名 `/encoder/Clip_output_0` 查找；
- 默认取 Clip 的**第一个输入**作为保留数据流；
- 脚本会做 `onnx.checker.check_model()` 检查，但这不等于功能一定完全等价；
- 如果模型中不存在该节点，脚本会提示未找到；
- 该脚本是 **ONNX 图后处理脚本**，不是训练脚本，也不是推理脚本。
- 端侧部署时，除了处理后的 `.onnx`，通常还需要一并提供 `vocab.json` 和 `le.pkl`。
