from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto

# ----- 从 onnx 文件加载模型 -----
input_path = "$path/model.onnx"
model = onnx.load(input_path)

# ---- 将模型精度转换为 fp16 -----
model = float16.convert_float_to_float16(model, op_block_list=['IsInf'], keep_io_types=True)

# ---- 自定义算子需要手动修改模型输出类型为 fp16 -----
for id, node in enumerate(model.graph.node):
    if "MergeEmbeddingLookupCombineOp" in node.name:
        for attr in node.attribute:
            if attr.name == "output_types":
                ints = list(map(lambda x: 10, attr.ints))
                attr.ints[:] = []
                attr.ints.extend(ints)
              
# ----- 执行模型检查并保存 -----
onnx.checker.check_model(model)
onnx.save(model, output_path)
