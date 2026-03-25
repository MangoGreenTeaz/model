import onnx

def remove_clip_node(model_path, save_path):
    model = onnx.load(model_path)
    graph = model.graph
    
    # 1. 找到 Clip 节点及其前驱/后继信息
    # 根据你的日志，输出名为 /encoder/Clip_output_0
    target_node_output = "/encoder/Clip_output_0"
    
    # 找到 Clip 节点的第一个输入（通常是真正的数据输入，跳过 constant 限制值）
    # 根据结构，223 的输入应该是 221 的输出
    new_input_source = None
    clip_node = None
    
    for node in graph.node:
        if node.output[0] == target_node_output:
            clip_node = node
            # 假设第一个输入是需要保留的数据流
            new_input_source = node.input[0] 
            break
            
    if not clip_node:
        print("未找到指定的 Clip 节点")
        return

    # 2. 将所有以 Clip 输出作为输入的节点，改为指向 Clip 的输入
    for node in graph.node:
        for i, node_input in enumerate(node.input):
            if node_input == target_node_output:
                node.input[i] = new_input_source
                print(f"已将节点 {node.name} ({node.op_type}) 的输入修改为 {new_input_source}")

    # 3. 从计算图中移除 Clip 节点
    graph.node.remove(clip_node)
    
    # 4. 保存并检查
    onnx.checker.check_model(model)
    onnx.save(model, save_path)
    print(f"修改完成，模型已保存至: {save_path}")

if __name__ == "__main__":
    remove_clip_node('tiny_student_state_dict_fp32.onnx', 'tiny_student_state_dict_fp32_no_clip.onnx')
