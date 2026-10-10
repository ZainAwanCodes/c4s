import mindspore as ms
from mindspore import ops

# Setup context
ms.set_device("CPU")
ms.set_context(mode=ms.PYNATIVE_MODE)

# Create two 2x2 tensors
x = ms.Tensor([[1, 2], [3, 4]], ms.float32)
y = ms.Tensor([[5, 6], [7, 8]], ms.float32)

print("Tensor x:\n", x)
print("Tensor y:\n", y)

# 1. Add two tensors.
print("\n--- 1. Addition (x + y) ---")
print(x + y)

# 2. Multiply them element-wise.
print("\n--- 2. Element-wise Multiplication (x * y) ---")
print(x * y)

# 3. Perform matrix multiplication.
print("\n--- 3. Matrix Multiplication (matmul) ---")
print(ops.matmul(x, y))

# 4. Calculate mean.
print("\n--- 4. Mean of x ---")
print(ops.mean(x))

# 5. Reshape a 2x2 tensor into 4x1.
print("\n--- 5. Reshape x into 4x1 ---")
print(x.reshape((4, 1)))

# 6. Select one row using indexing.
print("\n--- 6. Select first row of x ---")
print(x[0])
