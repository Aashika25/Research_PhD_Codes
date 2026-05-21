import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras
from keras import layers

# ── Choose backend before importing keras ─────────────────────────────────────
def _pick_backend():
    try:
        import jax; return "jax"
    except ImportError: pass
    try:
        import torch; return "torch"
    except ImportError: pass
    return "numpy"

backend = _pick_backend()
os.environ.setdefault("KERAS_BACKEND", backend)
print(f"Keras backend: {backend}")

keras.utils.set_random_seed(42)
np.random.seed(42)

EPOCHS     = 10
BATCH_SIZE = 256
LR         = 1e-3
VAL_SPLIT  = 0.15

# ─── Data ────────────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test .reshape(-1, 784).astype("float32") / 255.0
print(f"Train: {len(x_train):,}  |  Test: {len(x_test):,}")

# ─── Models ──────────────────────────────────────────────────────────────────
def build_plain_model():
    return keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10,  activation="softmax"),
    ], name="PlainNet")

def build_reg_model(drop_p=0.2):
    return keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512), layers.BatchNormalization(), layers.Activation("relu"), layers.Dropout(drop_p),
        layers.Dense(256), layers.BatchNormalization(), layers.Activation("relu"), layers.Dropout(drop_p),
        layers.Dense(128), layers.BatchNormalization(), layers.Activation("relu"), layers.Dropout(drop_p/2),
        layers.Dense(10,  activation="softmax"),
    ], name="DropBN_Net")

# ─── Train ───────────────────────────────────────────────────────────────────
def run_experiment(model, label):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    print(f"\n{'─'*55}\n  Training : {label}\n{'─'*55}")

    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=[keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5, verbose=0)],
        verbose=1,
    )
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  ✓ Test accuracy ({label}): {test_acc*100:.2f}%")
    return history, test_acc

plain_model = build_plain_model()
reg_model   = build_reg_model()
hist_plain, test_plain = run_experiment(plain_model, "Plain Network")
hist_reg,   test_reg   = run_experiment(reg_model,   "Dropout + BatchNorm")

# ─── Plot ────────────────────────────────────────────────────────────────────
epochs_range  = range(1, EPOCHS + 1)
plain_loss    = hist_plain.history["loss"]
plain_val_acc = [v * 100 for v in hist_plain.history["val_accuracy"]]
reg_loss      = hist_reg.history["loss"]
reg_val_acc   = [v * 100 for v in hist_reg.history["val_accuracy"]]

C_PLAIN, C_REG = "#E05C5C", "#4A90D9"
fig = plt.figure(figsize=(14, 6), facecolor="#0F0F14")
fig.suptitle("MNIST — Plain Network vs Dropout + BatchNorm  (Keras)",
             fontsize=15, fontweight="bold", color="white", y=0.97)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

def style_ax(ax, title, ylabel):
    ax.set_facecolor("#161620")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.set_xlabel("Epoch", color="#AAAAAA", fontsize=11)
    ax.set_ylabel(ylabel, color="#AAAAAA", fontsize=11)
    ax.tick_params(colors="#888888")
    ax.set_xticks(list(epochs_range))
    for sp in ax.spines.values(): sp.set_edgecolor("#333340")
    ax.grid(color="#222230", linestyle="--", linewidth=0.6)

ax1 = fig.add_subplot(gs[0])
style_ax(ax1, "Training Loss", "Cross-Entropy Loss")
ax1.plot(epochs_range, plain_loss, color=C_PLAIN, lw=2.2, marker="o", markersize=5, label="Plain")
ax1.plot(epochs_range, reg_loss,   color=C_REG,   lw=2.2, marker="s", markersize=5, label="Dropout + BN")
ax1.fill_between(epochs_range, plain_loss, alpha=0.08, color=C_PLAIN)
ax1.fill_between(epochs_range, reg_loss,   alpha=0.08, color=C_REG)
ax1.legend(framealpha=0.2, facecolor="#111118", edgecolor="#333340", labelcolor="white", fontsize=10)

ax2 = fig.add_subplot(gs[1])
style_ax(ax2, "Validation Accuracy", "Accuracy (%)")
ax2.plot(epochs_range, plain_val_acc, color=C_PLAIN, lw=2.2, marker="o", markersize=5,
         label=f"Plain  (test {test_plain*100:.1f}%)")
ax2.plot(epochs_range, reg_val_acc,   color=C_REG,   lw=2.2, marker="s", markersize=5,
         label=f"Dropout + BN  (test {test_reg*100:.1f}%)")
ax2.fill_between(epochs_range, plain_val_acc, alpha=0.08, color=C_PLAIN)
ax2.fill_between(epochs_range, reg_val_acc,   alpha=0.08, color=C_REG)
ax2.legend(framealpha=0.2, facecolor="#111118", edgecolor="#333340", labelcolor="white", fontsize=10)

fig.text(0.5, 0.01,
    f"PlainNet params: {plain_model.count_params():,}   |   "
    f"RegNet params: {reg_model.count_params():,}   |   "
    f"Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Backend: {backend}",
    ha="center", va="bottom", fontsize=8.5, color="#555566")

plt.savefig(r"Image/mnist_keras_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nPlot saved → mnist_keras_comparison.png")
plt.show()