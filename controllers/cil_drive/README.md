# Fase C — Conducción autónoma con el modelo CIL (mundo #2)

El controlador `cil_drive.py` carga el modelo entrenado y conduce por comandos
de navegación, con tres capas de seguridad por encima del steering del modelo.

## Requisitos

1. Copiar el modelo y su config junto a este controlador:
   ```
   controllers/cil_drive/cil_model.keras
   controllers/cil_drive/model_config.json
   ```
2. Ejecutar con un Python que tenga **TensorFlow/Keras + OpenCV** (el mismo
   entorno del entrenamiento — NO el Python 3.14 del sistema, que aún no tiene
   wheels de TensorFlow).

## Cómo ejecutarlo

1. Abrí `worlds/city_traffic_2025_02.wbt` en Webots y dale **play**.
2. Lanzá el controlador extern:
   ```bash
   webots-controller "controllers/cil_drive/cil_drive.py"
   ```

## Sensores agregados al mundo #2

Se editó `city_traffic_2025_02.wbt` (backup en `*.wbt.orig`) para montar en el
`BmwX5`, reusando las declaraciones de las actividades 3.1/4.2:

| Sensor | Slot | Uso |
|--------|------|-----|
| `Lidar` (64 rayos, 30°, 80 m) | front | Distancia frontal, disparo de evasión, ACC |
| `Radar` (80 m) | front | Velocidad relativa del vehículo más próximo |
| `Camera` + `Recognition` | top | Detección de peatón (`pedestrian`) y obstáculo |
| `ds_right_front/mid/rear` | center | Seguimiento de costado en la evasión |
| `Gyro`, `GPS`, `Display` | center | Orientación, posición, HUD |

También se fijó `SumoInterface.maxVehicles 50` (≥ 30, como pide el enunciado).

## Capas de seguridad (prioridad de arriba hacia abajo)

1. **Peatón → freno de emergencia.** `Recognition` detecta `pedestrian`,
   confirmado por LiDAR a < `PED_BRAKE_DIST = 12 m` → velocidad 0.
2. **Evasión de obstáculo.** Vehículo estacionado detectado por `Recognition` +
   LiDAR a < `EVADE_TRIGGER_DIST = 18 m` → cambio de carril por la izquierda y
   seguimiento de costado (máquina de estados de la act. 4.2). Tecla `E` la
   habilita/deshabilita por ruta.
3. **Distancia de umbral (ACC).** Distancia al vehículo más próximo (LiDAR +
   radar). **DISTANCIA DE UMBRAL = `VEHICLE_STOP_DIST = 10 m`**: si el vehículo
   detectado está a menos de 10 m, el auto **se detiene**; entre 10 y 20 m
   reduce la velocidad proporcionalmente. *(Indicá este valor en el video.)*

## Las 3 rutas de evaluación

Cada ruta debe usar **al menos un comando distinto** y favorecer el carril
derecho (sin vueltas en U). Sugerencia de cobertura:

| Ruta | Comando principal | Característica obligatoria |
|------|-------------------|---------------------------|
| 1 | `STRAIGHT` (derecho en intersecciones) | ACC: mantener distancia de umbral |
| 2 | `RIGHT` (al menos un giro a la derecha) | Evasión de vehículo estacionado (`E` ON) |
| 3 | `LEFT` (al menos un giro a la izquierda) | Peatón + freno de emergencia |

Reposicioná el vehículo en el origen de cada ruta (edificios de interés:
gasolinería, silos, iglesia, parque, etc.), origen y destino sobre el lado
derecho de conducción, y grabá desde ese punto.

## Optional Rendering (Webots → View → Optional Rendering)

Activá para el video: **Show Camera Frustums**, **Show LiDAR Point Cloud**,
**Show LiDAR Ray Paths**, **Show Radar Frustums**, **Show DistanceSensor Rays**
y **Show Recognition Bounding Boxes**.
