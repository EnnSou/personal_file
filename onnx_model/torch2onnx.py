# ----- 定义 onnx 算子关系 -----
# 单输入单输出算子
def fbgemm_asynchronous_complete_cumsum_custom(g, input):
    return g.op("cus_ops::fbgemm_asynchronous_complete_cumsum_op", input)
# 多输入多输出算子
def fbgemm_dense_to_jagged_custom(g, dense, x_offsets, total_L=None):
    return g.op("cus_ops::fbgemm_dense_to_jagged_custom_op", dense, x_offsets, total_L), \
           g.op("cus_ops::fbgemm_dense_to_jagged_custom_op", dense, x_offsets, total_L)
# 多输入单输出算子
def fbgemm_jagged_to_padded_dense_custom(g, value, offsets, max_lengths, padding_value=0):
    return g.op("cus_ops::fbgemm_jagged_to_padded_dense_custom_op", value, offsets, max_lengths, padding_value)
  
# ----- 注册 torch 算子到 onnx 算子映射关系 -----
torch.onnx.register_custom_op_symbolic("fbgemm::asynchronous_complete_cumsum", fbgemm_asynchronous_complete_cumsum_custom, 9)
torch.onnx.register_custom_op_symbolic("fbgemm::jagged_to_padded_dense", fbgemm_jagged_to_padded_dense_custom, 9)
torch.onnx.register_custom_op_symbolic("fbgemm::dense_to_jagged", fbgemm_dense_to_jagged_custom, 9)

# ----- 构建模型 -----
model = ModelNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# ----- 加载 pre-train 模型参数 -----
model_pth = "$path/torch_model.pth"
model_weight = torch.load(model_pth)
model.load_state_dict(model_weight)

# ----- fp16 精度 -----
# model.half()

# ----- torch2onnx 转换 ----
torch.onnx.export(model=model,args=({'inputs':data}), f='model.onnx', opset_version=13, verbose=True, input_names=None, output_names=None, dynamic_axes=None,custom_opsets={"fbgemm":9})
