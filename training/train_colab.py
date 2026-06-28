"""
Entrenamiento CIL self-contained (Colab o local). NO depende del notebook.

USO EN COLAB — una sola celda nueva:

    !git clone https://github.com/cesarcamilov1/navegacion-autonoma-proyecto-final-equipo-21.git
    !python navegacion-autonoma-proyecto-final-equipo-21/training/train_colab.py

Encuentra el dataset relativo a ESTE archivo (../dataset), entrena el modelo CIL
de 3 comandos (FOLLOW/LEFT/RIGHT) con balanceo + data augmentation, y guarda
cil_model.keras y model_config.json en /content (Colab) o en el directorio
actual. STRAIGHT se enruta a FOLLOW en inferencia (no es rama del modelo).
"""

import os
import json
import math
import random

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# --- Rutas: el dataset está en ../dataset respecto a este archivo (training/) ---
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "dataset"))
CSV_PATH = os.path.join(DATASET_DIR, "driving_log_relabeled.csv")
assert os.path.exists(CSV_PATH), (
    f"No existe {CSV_PATH}. ¿Clonaste el repo? ¿Es PÚBLICO?")

# Salida donde sea fácil de descargar (Colab => /content).
OUT_DIR = "/content" if os.path.isdir("/content") else os.getcwd()
OUT_MODEL = os.path.join(OUT_DIR, "cil_model.keras")
OUT_CONFIG = os.path.join(OUT_DIR, "model_config.json")

# --- Hiperparámetros ---
CROP_TOP, CROP_BOTTOM, IMG_W, IMG_H = 55, 5, 200, 88
COMMANDS = [2, 3, 4]                       # FOLLOW, LEFT, RIGHT
CMD_INDEX = {c: i for i, c in enumerate(COMMANDS)}
N_CMD = len(COMMANDS)
BATCH, EPOCHS, LR, VAL_SPLIT, MAX_STEER = 64, 60, 1e-4, 0.15, 0.5
BALANCE_CAP = 8000

print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))
print("Dataset:", DATASET_DIR)

# --- Carga del CSV ---
df = pd.read_csv(CSV_PATH)
df["path"] = df["image"].apply(lambda p: os.path.join(DATASET_DIR, p))
df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
print("Filas válidas:", len(df), "|", df["command_name"].value_counts().to_dict())


def preprocess(img_bgr):
    h = img_bgr.shape[0]
    img = img_bgr[CROP_TOP:h - CROP_BOTTOM, :, :]
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


FLIP_SWAP = {2: 2, 3: 4, 4: 3}
SHIFT_MAX_PX, SHIFT_STEER_PER_PX = 24, 0.004


def augment(img, st, cmd):
    if random.random() < 0.5:
        img = cv2.flip(img, 1); st = -st; cmd = FLIP_SWAP[cmd]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.5, 1.3), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.5:
        dx = random.randint(-SHIFT_MAX_PX, SHIFT_MAX_PX)
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
        st = st - dx * SHIFT_STEER_PER_PX
    return img, float(np.clip(st, -MAX_STEER, MAX_STEER)), cmd


class CILSequence(keras.utils.Sequence):
    def __init__(self, frame, training=True, **kw):
        super().__init__(**kw)
        self.df = frame.reset_index(drop=True)
        self.training = training
        c = self.df["command"].values
        self.groups = {int(k): np.where(c == k)[0] for k in np.unique(c)}
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.training:
            t = min(max(len(g) for g in self.groups.values()), BALANCE_CAP)
            self.idx = np.concatenate(
                [np.random.choice(g, t, replace=len(g) < t)
                 for g in self.groups.values()])
            np.random.shuffle(self.idx)
        else:
            self.idx = np.arange(len(self.df))

    def __len__(self):
        return math.ceil(len(self.idx) / BATCH)

    def __getitem__(self, i):
        rows = self.idx[i * BATCH:(i + 1) * BATCH]
        X, C, Y = [], [], []
        for r in rows:
            row = self.df.iloc[r]
            img = cv2.imread(row["path"])
            if img is None:
                continue
            st = float(row["steering"]); cmd = int(row["command"])
            if self.training:
                img, st, cmd = augment(img, st, cmd)
            X.append(preprocess(img))
            oh = np.zeros(N_CMD, np.float32); oh[CMD_INDEX[cmd]] = 1.0
            C.append(oh); Y.append(st)
        return ({"image": np.array(X, np.float32),
                 "command": np.array(C, np.float32)}, np.array(Y, np.float32))


tr, va = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED,
                          stratify=df["command"])
print("train", len(tr), "| val", len(va))
train_seq = CILSequence(tr, training=True)
val_seq = CILSequence(va, training=False)


def build_cil_model():
    ii = keras.Input((IMG_H, IMG_W, 3), name="image")
    ci = keras.Input((N_CMD,), name="command")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(ii)
    for f, k, s in [(24, 5, 2), (36, 5, 2), (48, 5, 2), (64, 3, 1), (64, 3, 1)]:
        x = layers.Conv2D(f, k, strides=s, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(50, activation="relu")(x)
    bo = [layers.Dense(1, name=f"steer_{i}")(
            layers.Dense(25, activation="relu", name=f"head_{i}")(x))
          for i in range(N_CMD)]
    out = layers.Dot(axes=1, name="steering")(
        [layers.Concatenate(name="branches")(bo), ci])
    m = keras.Model([ii, ci], out, name="CIL_branched")
    m.compile(keras.optimizers.Adam(LR), loss="mse", metrics=["mae"])
    return m


model = build_cil_model()
model.summary()

cbs = [
    keras.callbacks.ModelCheckpoint(OUT_MODEL, monitor="val_loss",
                                    save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                  restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=4, min_lr=1e-6, verbose=1),
]
model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS, callbacks=cbs)

model.save(OUT_MODEL)
json.dump({"crop_top": CROP_TOP, "crop_bottom": CROP_BOTTOM,
           "img_w": IMG_W, "img_h": IMG_H,
           "commands": COMMANDS, "max_steer": MAX_STEER},
          open(OUT_CONFIG, "w"), indent=2)
print("\nGUARDADO:\n ", OUT_MODEL, "\n ", OUT_CONFIG)
