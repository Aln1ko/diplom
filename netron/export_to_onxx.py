import torch
from dip4 import Network

model = Network()
model.eval()

dummy_input1 = torch.randn(1, 1, 370, 1226)
dummy_input2 = torch.randn(1, 1, 370, 1226)

# Экспортируй в ONNX
torch.onnx.export(model, (dummy_input1,dummy_input2), "model1.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=17)
