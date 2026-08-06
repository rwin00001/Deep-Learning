import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 1. Load CIFAR-10 Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Class labels in CIFAR-10
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 2. Normalize Data
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Model Architecture (Multi-Layer Perceptron)
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# 4. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()

# 5. Train Model
history = model.fit(
    X_train, y_train, validation_split=0.1, epochs=15, batch_size=256
)

# 6. Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# 7. Predictions & Visualization
y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=-1)

indices = np.random.choice(len(X_test), size=4, replace=False)
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

for ax, idx in zip(axes, indices):
    ax.axis("off")
    ax.imshow(X_test[idx])
    true_label = class_names[y_test[idx][0]]
    pred_label = class_names[y_pred[idx]]
    color = "green" if true_label == pred_label else "red"
    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)

plt.tight_layout()
plt.show()
