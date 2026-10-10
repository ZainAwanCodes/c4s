import numpy as np 
import mindspore as ms 
from mindspore import nn, Tensor 
 
ms.set_seed(42) 
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU") 
 
# y = 2x + 1 
x_np = np.arange(1, 11, dtype=np.float32).reshape(-1, 1) 
y_np = (2 * x_np + 1).astype(np.float32) 
 
x = Tensor(x_np) 
y = Tensor(y_np) 
 
net = nn.Dense(1, 1) 
loss_fn = nn.MSELoss() 
optimizer = nn.SGD(net.trainable_params(), learning_rate=0.001) 
 
def forward_fn(data, label): 
    pred = net(data) 
    loss = loss_fn(pred, label) 
    return loss, pred 
 
grad_fn = ms.value_and_grad( 
    forward_fn, 
    grad_position=None, 
    weights=optimizer.parameters, 
    has_aux=True 
) 
 
def train_step(data, label): 
    (loss, _), grads = grad_fn(data, label) 
    optimizer(grads) 
    return loss 
for epoch in range(300): 
    loss = train_step(x, y) 
    if (epoch + 1) % 50 == 0: 
        print(f"Epoch {epoch+1:3d} | Loss: {float(loss.asnumpy()):.6f}") 
 
new_x = Tensor(np.array([[12]], dtype=np.float32)) 
print("Prediction for x=12:", net(new_x).asnumpy()) 
print("Expected value is close to 25.")