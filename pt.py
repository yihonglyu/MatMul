import gc
import time
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

Matrix_B_dim1 = 4096
Matrix_B_dim2 = 32000

class MatrixMultiplicationModel(nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(Matrix_B_dim1, Matrix_B_dim2))

    def forward(self, x):
        return torch.mm(x, self.weight)

# Create the model instance
model = MatrixMultiplicationModel()

# Load the model state from the .pth file
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# Create an example input tensor
input_shape = (1026, 4096)
input_tensor = torch.randn(input_shape[0], input_shape[1])

# Warm up
model(input_tensor)

count = 5
# Perform inference
gc.collect()
gc.disable()
start = time.time()
#with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#    with record_function("model_inference"):
for _ in range(count):
    _ = model(input_tensor)
end = time.time()
gc.enable()
#print(f"PT for count {count}: {(end - start):.3f} s")
print(f"{(end - start):.3f}")
#print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=-1))

#print("Output shape:", output_tensor.shape)
#print("Output tensor:")
#print(output_tensor)
