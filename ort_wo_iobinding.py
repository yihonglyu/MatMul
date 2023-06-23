import gc
import numpy as np
import onnxruntime as ort
import time

# Load the ONNX model
model_path = 'model.onnx'
session = ort.InferenceSession(model_path)

# Get the input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Create an example input tensor
input_shape = (1, 1026, 4096)
input_tensor = np.random.randn(*input_shape).astype(np.float32)

# Warm up
session.run([output_name], {input_name: input_tensor})

# Run the inference
gc.collect()
gc.disable()
start = time.time()
for _ in range(320):
    _ = session.run([output_name], {input_name: input_tensor})
end = time.time()
gc.enable()
print(f"ORT without io binding: {(end - start):.3f} s")

# Get the output tensor
#output_tensor = outputs[0]

#print("Output shape:", output_tensor.shape)
#print("Output tensor:")
#print(output_tensor)