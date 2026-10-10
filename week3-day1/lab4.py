import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

ms.set_device("CPU")
ms.set_context(mode=ms.PYNATIVE_MODE)

class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        # 1st Layer: Input size = 2, Hidden size = 4
        self.fc1 = nn.Dense(2, 4)
        self.relu1 = nn.ReLU()
        
        # 2nd Layer: Hidden size = 4, Hidden size = 4
        self.fc2 = nn.Dense(4, 4)
        self.relu2 = nn.ReLU()
        
        # 3rd Layer: Hidden size = 4, Hidden size = 4
        self.fc3 = nn.Dense(4, 4)
        self.relu3 = nn.ReLU()
        
        # 4th Layer: Hidden size = 4, Output size = 2
        self.fc4 = nn.Dense(4, 2)

    def construct(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)

net = SimpleNet()
print(net)

sample = Tensor(np.array([[0.25, 0.80]], dtype=np.float32))
logits = net(sample)

# 4. Compare output shape
print("\n--- Model Evaluation ---")
print("Input shape:", sample.shape)
print("Output shape:", logits.shape)
print("Raw output (logits):", logits)

# 5. Print trainable parameter names and shapes
print("\n--- Trainable Parameters ---")
for param in net.trainable_params():
    print(f"Name: {param.name}, Shape: {param.shape}")