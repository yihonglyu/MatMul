import torch
import torch.nn as nn

Matrix_B_dim1 = 4096
Matrix_B_dim2 = 32000

class MatrixMultiplicationModel(nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(Matrix_B_dim1, Matrix_B_dim2))

    def forward(self, x):
        return torch.mm(x, self.weight)

# Create the model
model = MatrixMultiplicationModel()

# Create an example input tensor
input_shape = (1026, 4096)
input_tensor = torch.randn(input_shape[0], input_shape[1])

# Perform a forward pass to initialize the model parameters
_ = model(input_tensor)

# Save the model as a .pth file
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)
print('PyTorch model saved as', model_path)