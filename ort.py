import gc
import numpy as np
import onnxruntime as ort
import time

# Load the ONNX model
model_path = 'model.onnx'
options = ort.SessionOptions()
#options.enable_profiling=True
#options.intra_op_num_threads = 16
#options.inter_op_num_threads = 16
session = ort.InferenceSession(
    model_path,
    sess_options=options
    )

# Get the input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Create an example input tensor
input_shape = (1, 1026, 4096)
input_tensor = np.random.randn(*input_shape).astype(np.float32)
input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_tensor)

# Create IO binding
io_binding = session.io_binding()
io_binding.bind_input(
    input_name,
    device_type=input_ortvalue.device_name(),
    device_id=0,
    element_type=np.float32,
    shape=input_ortvalue.shape(),
    buffer_ptr=input_ortvalue.data_ptr(),
    )
io_binding.bind_output(output_name)

# Warm up
session.run_with_iobinding(io_binding)

count = 5
# Run the inference
gc.collect()
gc.disable()
start = time.time()
for _ in range(count):
    session.run_with_iobinding(io_binding)
end = time.time()
gc.enable()
#print(f"ORT with io binding for count {count}: {(end - start):.3f} s")
print(f"{(end - start):.3f}")

# Get the output tensor
#output_tensor = io_binding.copy_outputs_to_cpu()[0]

#print("Output shape:", output_tensor.shape)
#print("Output tensor:")
#print(output_tensor)
