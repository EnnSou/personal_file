import onnx
from onnx import helper

'''
整理记录常用 onnx 模型修改方法，整个脚本不一定能直接运行，需要根据具体模型修改。
'''

## 加载模型
model_path = "/home/zheng/Downloads/yepan/c3_od_gen2.onnx"
model_onnx = onnx.load(model_path)

### 获取模型graph和node
graph = model_onnx.graph
node = graph.node

### 修改模型input信息
input = graph.input[0]
graph.input[0].type.tensor_type.shape.dim[2].dim_value = 540
new_input = graph.input[0]



### 在模型头部增加slice算子
starts_tensor = helper.make_tensor(
    'starts', onnx.TensorProto.INT64, dims = [2], vals=[28,0])
ends_tensor = helper.make_tensor('ends', onnx.TensorProto.INT64, dims=[2], vals=[540, 960])
axes_tensor = helper.make_tensor(
    'axes',
    onnx.TensorProto.INT64,
    [2],
    vals=[2,3]
)
graph.initializer.extend(
        [starts_tensor, ends_tensor, axes_tensor])
slice_node = onnx.helper.make_node(
    op_type='Slice',
    inputs=['c','starts','ends','axes'],
    outputs=['c_crop']
)

graph.node.insert(0, slice_node)
graph.node[1].input[0] = slice_node.output[0]

### 删除指定算子
remove_node_name = 'Concat_220'
remove_node_index = next(i for i , node in enumerate(graph.node) if remove_node_name == node.name)
print("remove_node_index:", remove_node_index)
graph.node.remove(graph.node[remove_node_index])

### 在模型output前增加resize算子
rois = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
scales = [1.0, 1.0, 0.6875, 0.699999988079071]
roi_tensor = helper.make_tensor(
    'roi', onnx.TensorProto.FLOAT, [8], rois)
scales_tensor = helper.make_tensor(
    'scales', onnx.TensorProto.FLOAT, [4], scales)

graph.initializer.extend(
    [roi_tensor, scales_tensor])

resize_node = onnx.helper.make_node(
    op_type='Resize',
    inputs=['830','roi', 'scales'],
    outputs=['head_stixel'],
    mode='linear',
)
# print(resize_node)

insert_node_name = 'Conv_215'
insert_node_index = next(i for i , node in enumerate(graph.node) if insert_node_name == node.name)
print("insert_node_index:", insert_node_index)
graph.node.insert(insert_node_index + 1, resize_node)


### 修改模型output 信息
graph.output[3].type.tensor_type.shape.dim[2].dim_value = 88
graph.output[3].type.tensor_type.shape.dim[3].dim_value = 168

### 添加指定节点为模型 output
graph.output.extend([onnx.ValueInfoProto(name='1')])

### 整个模型node添加到输出
for node in graph.node:
    for output in node.output:
        graph.output.extend([onnx.ValueInfoProto(name=output)])

## 生成新模型，完成shape检查
new_graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
new_model = onnx.helper.make_model(new_graph)
new_onnx_model = onnx.shape_inference.infer_shapes(new_model)

## 模型检查并保存
onnx.checker.check_model(new_onnx_model)
onnx.save(model_onnx, "/home/zheng/Downloads/yepan/c3_od_gen2_resize.onnx")