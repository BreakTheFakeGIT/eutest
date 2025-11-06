print("cudnn is not included in the uv installation, so this is a test file to check if it works.")

import torch
import cudnn

# Prepare sample input data. nvmath-python accepts input tensors from pytorch, cupy, and
# numpy.
b, m, n, k = 1, 1024, 1024, 512
A = torch.randn(b, m, k, dtype=torch.float32, device="cuda")
B = torch.randn(b, k, n, dtype=torch.float32, device="cuda")
bias = torch.randn(b, m, 1, dtype=torch.float32, device="cuda")

result = torch.empty(b, m, n, dtype=torch.float32, device="cuda")

# Use the stateful Graph object in order to perform multiple matrix multiplications
# without replanning. The cudnn API allows us to fine-tune our operations by, for
# example, selecting a mixed-precision compute type.
graph = cudnn.pygraph(
   intermediate_data_type=cudnn.data_type.FLOAT,
   compute_data_type=cudnn.data_type.FLOAT,
)

a_cudnn_tensor    = graph.tensor_like(A)
b_cudnn_tensor    = graph.tensor_like(B)
bias_cudnn_tensor = graph.tensor_like(bias)

c_cudnn_tensor = graph.matmul(name="matmul", A=a_cudnn_tensor, B=b_cudnn_tensor)
d_cudnn_tensor = graph.bias(name="bias", input=c_cudnn_tensor, bias=bias_cudnn_tensor)

# Build the matrix multiplication. Building returns a sequence of algorithms that can be
# configured. Each algorithm is a JIT generated function that can be executed on the GPU.

graph.build([cudnn.heur_mode.A])
workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

# Execute the matrix multiplication.
graph.execute(
   {
       a_cudnn_tensor: A,
       b_cudnn_tensor: B,
       bias_cudnn_tensor: bias,
       d_cudnn_tensor: result,
   },
   workspace
)