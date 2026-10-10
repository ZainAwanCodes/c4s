"""
============================================================
 Student Pass/Fail Predictor â€” MindSpore Neural Network
 Dataset : UCI Student Performance (student-mat.csv)
 Author  : Beginner MindSpore Tutorial
============================================================

WHAT THIS SCRIPT DOES (plain English):
1. Load the student CSV â†’ DataFrame
2. Turn grades into Pass (1) / Fail (0)
3. Pick 8 numerical features, normalize them
4. Split 80 % train / 20 % test (no sklearn)
5. Build a tiny 3-layer neural net with MindSpore
6. Train it with CrossEntropyLoss + Adam optimizer
7. Evaluate accuracy on unseen test students
8. Predict Pass/Fail for a brand-new student
"""

# â”€â”€â”€ 0. IMPORTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "student-mat.csv")
import numpy  as np
import pandas as pd
import mindspore as ms
from mindspore import nn, Tensor

# Reproducibility â€” same random numbers every run
ms.set_seed(42)
np.random.seed(42)

# Run eagerly on CPU (PYNATIVE_MODE = easy debugging)
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device("CPU")

print("=" * 55)
print("  Student Pass/Fail Predictor â€” MindSpore")
print("=" * 55)


# â”€â”€â”€ 1. LOAD THE DATASET â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The CSV uses semicolons (;) as delimiters
df = pd.read_csv(CSV_PATH, sep=";")

print("\nâ”€â”€ Dataset Overview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst 3 rows:")
print(df.head(3).to_string())

print("\nColumn info:")
df.info()

print("\nBasic statistics:")
print(df.describe())

# G1 and G2 are sometimes stored as quoted strings â€” convert them
df["G1"] = pd.to_numeric(df["G1"], errors="coerce")
df["G2"] = pd.to_numeric(df["G2"], errors="coerce")
df["G3"] = pd.to_numeric(df["G3"], errors="coerce")


# â”€â”€â”€ 2. CHECK FOR MISSING VALUES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
missing = df.isnull().sum()
print("\nMissing values per column:")
if missing.any():
    print(missing[missing > 0])
else:
    print("  None â€” dataset is clean!")


# â”€â”€â”€ 3. CREATE THE TARGET COLUMN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# G3 is the final grade (0â€“20).
# We call it a PASS if G3 >= 10, FAIL otherwise.
df["Result"] = (df["G3"] >= 10).astype(int)   # 1 = Pass, 0 = Fail

pass_count = df["Result"].sum()
fail_count = len(df) - pass_count
print(f"\nTarget distribution: Pass={pass_count}  Fail={fail_count}")

# Drop G3 â€” it would perfectly predict the answer (data leakage!)
df.drop(columns=["G3"], inplace=True)


# â”€â”€â”€ 4. SELECT FEATURES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# These 8 columns are all numeric â€” no encoding needed.
feature_cols = [
    "studytime",   # weekly study time (1â€“4)
    "failures",    # number of past class failures (0â€“4)
    "absences",    # number of school absences
    "G1",          # first period grade  (0â€“20)
    "G2",          # second period grade (0â€“20)
    "health",      # current health status (1â€“5)
    "freetime",    # free time after school (1â€“5)
    "goout",       # going out with friends (1â€“5)
]

X = df[feature_cols].values.astype(np.float32)   # shape: (395, 8)
y = df["Result"].values.astype(np.int32)          # shape: (395,)

print(f"\nFeature matrix X : {X.shape}")
print(f"Label vector   y : {y.shape}")


# â”€â”€â”€ 5. NORMALIZE (STANDARDIZE) FEATURES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Formula: X_norm = (X - mean) / std
# This keeps all features on the same scale so no single
# feature dominates just because it has large values.
X_mean = X.mean(axis=0)    # mean of each column
X_std  = X.std(axis=0)     # std  of each column
X_std[X_std == 0] = 1      # avoid division by zero (safety)

X = (X - X_mean) / X_std

print("\nNormalized feature means  (should be close to 0):", np.round(X.mean(axis=0), 3))
print("Normalized feature stds   (should be close to 1):", np.round(X.std(axis=0),  3))


# â”€â”€â”€ 6. TRAIN / TEST SPLIT (80 / 20) â€” NO SKLEARN â”€â”€â”€â”€â”€â”€â”€
# Shuffle indices randomly, then slice 80% / 20%.
n_samples  = len(X)
indices    = np.random.permutation(n_samples)   # e.g. [302, 7, 194, ...]
split      = int(0.8 * n_samples)               # e.g. 316

train_idx  = indices[:split]
test_idx   = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"\nTrain samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")


# â”€â”€â”€ 7. CONVERT NUMPY ARRAYS TO MINDSPORE TENSORS â”€â”€â”€â”€â”€â”€â”€â”€
# MindSpore works with Tensor objects, just like PyTorch.
# float32 for features, int32 for class labels.
X_train_t = Tensor(X_train)          # shape (316, 8)  float32
y_train_t = Tensor(y_train)          # shape (316,)    int32
X_test_t  = Tensor(X_test)           # shape (79, 8)   float32
y_test_t  = Tensor(y_test)           # shape (79,)     int32

print(f"\nX_train_t dtype: {X_train_t.dtype}, shape: {X_train_t.shape}")
print(f"y_train_t dtype: {y_train_t.dtype}, shape: {y_train_t.shape}")


# â”€â”€â”€ 8. DEFINE THE NEURAL NETWORK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# nn.Cell is MindSpore's base class for all neural networks.
# You must define:
#   __init__    : create the layers
#   construct   : define how data flows through them (forward pass)

class StudentNet(nn.Cell):
    """
    A simple 3-layer feed-forward network.

    Architecture:
        Input(8) --> Dense(8->16) --> ReLU
                 --> Dense(16->8) --> ReLU
                 --> Dense(8->2)  --> [logits]

    The output is 2 raw scores (logits), one per class:
        logits[0] = score for Fail (class 0)
        logits[1] = score for Pass (class 1)
    """
    def __init__(self):
        super(StudentNet, self).__init__()

        # Dense(in, out) is a fully-connected (linear) layer.
        # It learns a weight matrix W and bias b such that:
        #   output = input @ W.T + b
        self.layer1 = nn.Dense(8,  16)   # 8  input features --> 16 hidden neurons
        self.layer2 = nn.Dense(16, 8)    # 16 hidden neurons  --> 8  hidden neurons
        self.layer3 = nn.Dense(8,  2)    # 8  hidden neurons  --> 2  output classes

        # ReLU activation: turns negatives to 0, keeps positives.
        # It introduces non-linearity â€” without it every layer
        # would just be a linear function of the input.
        self.relu   = nn.ReLU()

    def construct(self, x):
        """
        Forward pass â€” data flows from top to bottom.
        'x' is a batch of student feature rows, shape (N, 8).
        """
        x = self.relu(self.layer1(x))   # (N, 8)  --> (N, 16) --> ReLU --> (N, 16)
        x = self.relu(self.layer2(x))   # (N, 16) --> (N, 8)  --> ReLU --> (N, 8)
        x = self.layer3(x)              # (N, 8)  --> (N, 2)   raw logits
        return x                        # NO softmax here â€” loss handles it


# Create one instance of the network
net = StudentNet()
print("\nâ”€â”€ Network Architecture â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(net)


# â”€â”€â”€ 9. LOSS FUNCTION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CrossEntropyLoss internally:
#   1. Applies softmax to logits -> probabilities
#   2. Computes -log(probability of the correct class)
#   3. Averages across the batch
# A lower loss means the model is more confident and correct.
loss_fn = nn.CrossEntropyLoss()


# â”€â”€â”€ 10. OPTIMIZER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Adam adjusts each weight's learning rate individually.
# It is usually faster and more stable than plain SGD.
# trainable_params() returns all W and b tensors in the net.
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)


# â”€â”€â”€ 11. FORWARD FUNCTION + GRADIENT SETUP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ms.value_and_grad() wraps a function so that calling it
# returns BOTH the function's output AND the gradients of
# the loss with respect to the model parameters.

def forward_fn(data, label):
    """
    Runs one forward pass and returns (loss, logits).
    'has_aux=True' tells value_and_grad that the second
    return value (logits) is auxiliary â€” not differentiated.
    """
    logits = net(data)              # forward pass -> raw class scores
    loss   = loss_fn(logits, label) # compare predictions to true labels
    return loss, logits             # return both for grad calculation

# grad_fn(data, label) -> ((loss, logits), grads)
# grads is a tuple of gradient tensors, one per trainable parameter.
grad_fn = ms.value_and_grad(
    forward_fn,
    grad_position=None,            # don't differentiate w.r.t. inputs
    weights=optimizer.parameters,  # differentiate w.r.t. model weights
    has_aux=True                   # forward_fn returns (loss, aux); aux = logits
)


# â”€â”€â”€ 12. ONE TRAINING STEP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_step(data, label):
    """
    Complete one gradient-descent update:
      1. Forward pass  -> loss
      2. Backward pass -> gradients
      3. Optimizer     -> update weights
    Returns the scalar loss for logging.
    """
    # grad_fn runs forward_fn AND computes gradients in one call
    (loss, _), grads = grad_fn(data, label)
    # optimizer(grads) applies the Adam update rule to every
    # trainable parameter: param = param - lr * grad (roughly)
    optimizer(grads)
    return loss


# â”€â”€â”€ 13. TRAINING LOOP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EPOCHS = 50

print("\nâ”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
for epoch in range(1, EPOCHS + 1):
    loss = train_step(X_train_t, y_train_t)
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:2d}/{EPOCHS} â€” Loss: {float(loss.asnumpy()):.4f}")


# â”€â”€â”€ 14. EVALUATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pass the test set through the network to get logits.
test_logits = net(X_test_t)                          # shape (N_test, 2)

# .asnumpy() converts Tensor -> NumPy array.
logits_np   = test_logits.asnumpy()                  # (N_test, 2) float32

# argmax(axis=1) picks the column with the higher score:
#   0 -> Fail,  1 -> Pass
predicted   = np.argmax(logits_np, axis=1)           # (N_test,)

# Compare predictions to true labels
correct     = np.sum(predicted == y_test)
total       = len(y_test)
accuracy    = correct / total * 100

print("\nâ”€â”€ Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  Correct predictions   : {correct} / {total}")
print(f"  Incorrect predictions : {total - correct} / {total}")
print(f"  Test Accuracy         : {accuracy:.2f}%")


# â”€â”€â”€ 15. PREDICT FOR A SINGLE NEW STUDENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Use the SAME normalization parameters as training!
#   studytime=2, failures=0, absences=3, G1=14, G2=13,
#   health=4, freetime=3, goout=2
sample_raw  = np.array([[2, 0, 3, 14, 13, 4, 3, 2]], dtype=np.float32)
sample_norm = (sample_raw - X_mean) / X_std          # normalize with training stats
sample_t    = Tensor(sample_norm)

sample_logits = net(sample_t)                        # shape (1, 2)
sample_np     = sample_logits.asnumpy()[0]           # [fail_score, pass_score]

# Softmax converts raw logits -> probabilities that sum to 1
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

probs      = softmax(sample_np)
pred_class = int(np.argmax(probs))                   # 0 or 1
confidence = probs[pred_class] * 100

label_map = {0: "FAIL", 1: "PASS"}
print("\nâ”€â”€ Single-Student Prediction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
print(f"  Input  : studytime=2, failures=0, absences=3,")
print(f"           G1=14, G2=13, health=4, freetime=3, goout=2")
print(f"  Fail probability : {probs[0]*100:.1f}%")
print(f"  Pass probability : {probs[1]*100:.1f}%")
print(f"  Prediction       : *** {label_map[pred_class]} *** "
      f"(confidence: {confidence:.1f}%)")


# â”€â”€â”€ 16. KEY LINES EXPLAINED â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("""
========================================================
 KEY LINES EXPLAINED
========================================================

(1) logits = net(data)
    Calls StudentNet.construct(data).
    'data' is a batch of feature rows (shape: N x 8).
    The output 'logits' has shape N x 2 â€” two raw scores
    per student, one for Fail, one for Pass.
    They are NOT probabilities yet (can be negative).

(2) loss = loss_fn(logits, label)
    CrossEntropyLoss:
      a) Applies softmax to logits -> probabilities
      b) Looks up the probability assigned to the TRUE class
      c) Returns the average of -log(p_correct) over the batch
    A perfect model -> loss ~ 0; a random model -> loss ~ 0.69

(3) (loss, _), grads = grad_fn(data, label)
    grad_fn wraps forward_fn with ms.value_and_grad().
    It performs:
      - The forward pass  (same as steps 1 + 2)
      - Automatic differentiation (backpropagation)
    The result is:
      - (loss, logits) â€” the forward outputs
      - grads          â€” d(loss)/d(param) for every W and b
    We only need 'loss' here; _ discards the logits.

(4) optimizer(grads)
    Adam uses the gradients to update every trainable parameter:
      param <- param - lr * adam_adjusted_grad
    After this call the model is (slightly) better at predicting
    the training data.

========================================================
 COMPLETE TRAINING PIPELINE (simple terms)
========================================================

  CSV file
    |  Pandas reads rows, selects 8 columns
    v
  NumPy array
    |  Normalize so all features are on the same scale
    v
  MindSpore Tensor
    |
    v
  StudentNet.construct()        <-- FORWARD PASS
    Dense(8->16) + ReLU
    Dense(16->8) + ReLU
    Dense(8->2)
    logits: 2 raw class scores per student
    |
    v
  CrossEntropyLoss
    Compares logits to true labels -> single loss number
    |
    v
  ms.value_and_grad()           <-- BACKWARD PASS
    Computes d(loss)/d(every_weight)  (backpropagation)
    |
    v
  Adam optimizer
    Updates weights to reduce the loss
    |
    v
  Repeat for 50 epochs
    |
    v
  Evaluate on unseen test students
    argmax(logits) -> predicted class -> accuracy
========================================================
""")

