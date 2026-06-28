# Proyecto final — Conditional Imitation Learning (CIL) en Webots

Solución de Behavioral Cloning **condicionada por comandos de navegación**
(Codevilla et al.). Se entrena en el **mundo #1** conduciendo manualmente y se
evalúa el modelo en el **mundo #2** (peatones, vehículos estacionados y tráfico
de SUMO).

## Pipeline

```
Fase A  (mundo #1)        Fase B  (Notebook/Keras)        Fase C  (mundo #2)
conducción manual   -->   entrenamiento del modelo   -->   conducción autónoma
+ comandos + CSV          CIL branched (PilotNet)          por comandos + sensores
controllers/cil_recorder  training/train_cil.ipynb         controllers/cil_drive
```

## Estructura

```
proyecto-final/
├── worlds/
│   ├── city_traffic_2025_01.wbt        # mundo #1 (entrenamiento)
│   ├── city_traffic_2025_02.wbt        # mundo #2 (evaluación, sensores añadidos)
│   └── city_traffic_2025_02.wbt.orig   # backup del mundo #2 original
├── controllers/
│   ├── cil_recorder/                   # Fase A — recolector de dataset
│   └── cil_drive/                      # Fase C — inferencia + seguridad
├── training/
│   ├── train_cil.ipynb                 # Fase B — entrenamiento (Keras)
│   ├── merge_dataset.py                # une sesiones del equipo (idempotente)
│   └── relabel_from_gps.py             # corrige comandos desde la trayectoria GPS
└── dataset/                            # generado por la Fase A
    ├── img/
    ├── driving_log.csv                 # crudo (comandos por tecla, pueden venir mal)
    └── driving_log_relabeled.csv       # comandos corregidos por GPS (lo que entrena)
```

## Modelo

CIL **branched** estilo Codevilla con backbone convolucional **PilotNet /
Bojarski** compartido y una rama por comando. El comando activo (one-hot)
selecciona la rama; la salida es el **ángulo de dirección**. La velocidad NO se
entrena (es constante ≤ 30 km/h). Comandos: `2=FOLLOW`, `3=LEFT`, `4=RIGHT`
(**3 ramas**). `5=STRAIGHT` no tiene rama propia: en una ciudad con líneas de
carril "cruzar derecho" se comporta igual que seguir el carril, así que el
controlador de inferencia **enruta STRAIGHT a FOLLOW**.

## Entornos de Python

- **Fase A y C** (controladores Webots): Python con `opencv-python` y `numpy`.
  La Fase C además necesita `tensorflow`/`keras` para cargar el modelo.
- **Fase B** (entrenamiento): **Google Colab (GPU)** o un venv con **Python
  3.11/3.12 + TensorFlow**. El Python 3.14 del sistema todavía no tiene wheels
  de TensorFlow.

## Flujo de trabajo

1. **Recolectar** (cada integrante del equipo): correr `cil_recorder` en el
   mundo #1, manejar en ambos sentidos cubriendo todas las rutas, incluir
   maniobras de recuperación. Ver `controllers/cil_recorder/README.md`.
2. **Unir** los datasets del equipo: `python training/merge_dataset.py
   <carpeta_dataset_de_otro>`. Es idempotente (solo agrega sesiones nuevas,
   identificadas por la columna `session`) y copia las imágenes + agrega las
   filas al `driving_log.csv`. Sirve para sesiones grabadas en Windows o Linux.
3. **Re-etiquetar**: `python training/relabel_from_gps.py`. Deriva el comando
   de cada frame de la geometría de la trayectoria GPS (las anotaciones por
   tecla suelen quedar mal) y normaliza las rutas de imagen a `/`. Genera
   `dataset/driving_log_relabeled.csv` sin tocar el original ni el steering.
4. **Subir** la carpeta `dataset/` a un repo de GitHub.
5. **Entrenar** con `training/train_cil.ipynb` (celda 0: `!git clone` del repo)
   → exporta `cil_model.keras` y `model_config.json`.
6. **Evaluar**: copiar esos dos archivos a `controllers/cil_drive/` y correr en
   el mundo #2 las 3 rutas. Ver `controllers/cil_drive/README.md`.

> Objetivo de dataset: > 10 000 imágenes (logrado: **35 750 frames** de 2
> sesiones — `camilo` 15 750 + `Suzu` 20 000). La data augmentation del notebook
> (flip, brillo, shift) se aplica online en el entrenamiento.
