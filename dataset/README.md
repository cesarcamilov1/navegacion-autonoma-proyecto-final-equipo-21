# CIL dataset — Webots (mundo #1)

Dataset de Behavioral Cloning **condicionado por comandos de navegación**
(Codevilla et al.) para el proyecto final. Recolectado conduciendo manualmente
el mundo #1 con `cil_recorder`.

- **35 750 frames**, 2 sesiones (`camilo` 15 750 + `Suzu` 20 000), dos conductores.
- Cámara a bordo 320×160, JPG, en `img/`.

## Archivos

- **`driving_log_relabeled.csv`** — el que entrena. Columnas:
  `session, frame, time, image, command, command_name, steering, speed, gps_x, gps_y`.
  Comandos: `2=FOLLOW`, `3=LEFT`, `4=RIGHT`. Las etiquetas de comando se
  derivaron de la **geometría de la trayectoria GPS** (las anotadas con el
  teclado quedaron mal); el steering es el del humano, sin tocar.
- `driving_log.csv` — crudo (comandos por tecla, sin corregir; se conserva para
  reproducibilidad).
- `img/` — imágenes referenciadas por la columna `image`.

## Uso (Colab)

Este dataset vive dentro del repo del proyecto, en la carpeta `dataset/`:

```python
!git clone https://github.com/cesarcamilov1/navegacion-autonoma-proyecto-final-equipo-21.git
# DATASET_DIR = "navegacion-autonoma-proyecto-final-equipo-21/dataset"
```
