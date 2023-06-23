import torch
import torch.nn as nn
import torch.onnx as onnx

Matrix_B_dim1 = 4096
Matrix_B_dim2 = 32000

class MatMulModel(nn.Module):
    def __init__(self):
        super(MatMulModel, self).__init__()
        self.matmul = nn.Linear(Matrix_B_dim1, Matrix_B_dim2, bias=False)

    def forward(self, x):
        return self.matmul(x)

# Create the PyTorch model
model = MatMulModel()

# Set the model in evaluation mode
model.eval()

# Create an example input tensor
input_shape = (1, 1026, 4096)
input_tensor = torch.randn(input_shape)

# Export the model to ONNX
onnx_path = 'model.onnx'
torch.onnx.export(model, input_tensor, onnx_path, opset_version=12)

print('Model exported to ONNX format as', onnx_path)