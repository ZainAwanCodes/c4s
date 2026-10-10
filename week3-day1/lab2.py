import mindspore as ms
import numpy as np

# Suppress the warning by using set_device instead of set_context(device_target=...)
ms.set_device("CPU")
ms.set_context(mode=ms.PYNATIVE_MODE)

# 1. Create a 1-D tensor with five values.
tensor_1d = ms.Tensor([10, 20, 30, 40, 50], ms.float32)

# 2. Create a 2x3 tensor from NumPy.
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_2x3 = ms.Tensor(np_array)

# 3. Print shape and dtype.
print("--- 1-D Tensor ---")
print(tensor_1d)
print("Shape:", tensor_1d.shape)
print("Dtype:", tensor_1d.dtype)

print("\n--- 2x3 Tensor ---")
print(tensor_2x3)
print("Shape:", tensor_2x3.shape)
print("Dtype:", tensor_2x3.dtype)

# 4. Convert the tensor back to NumPy using asnumpy()
numpy_converted = tensor_2x3.asnumpy()
print("\n--- Converted back to NumPy ---")
print(numpy_converted)
print("Type after conversion:", type(numpy_converted))
