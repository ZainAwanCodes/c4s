import numpy as np
import mindspore.dataset as ds
features = np.array([
 [2, 60, 45],
 [4, 75, 70],
 [6, 88, 82],
 [1, 50, 40]
], dtype=np.float32)
labels = np.array([0, 1, 1, 0], dtype=np.int32)
dataset = ds.NumpySlicesDataset(
 (features, labels),
 column_names=["features", "label"],
 shuffle=False
).batch(2)
for feature_batch, label_batch in dataset.create_tuple_iterator():
 print("Features:\n", feature_batch)
 print("Labels:", label_batch)
