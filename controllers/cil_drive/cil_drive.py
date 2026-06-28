"""
CIL Drive — Fase C del proyecto final (Conditional Imitation Learning).

Controlador de INFERENCIA para el mundo #2. La dirección (steering) la produce
el modelo CIL entrenado en la Fase B, condicionado por el comando de navegación
que el operador introduce con el teclado. Sobre esa dirección se montan tres
capas de seguridad reusadas de actividades anteriores:

  1. PEATÓN  -> detección por el nodo Recognition de la cámara, confirmada por
               LiDAR -> FRENO DE EMERGENCIA  (actividad 3.1).
  2. EVASIÓN -> vehículo estacionado / obstáculo detectado por Recognition +
               LiDAR -> maniobra de cambio de carril por la izquierda y
               seguimiento de costado  (actividad 4.2).
  3. ACC     -> distancia de umbral al vehículo más próximo (LiDAR, con radar
               como complemento de velocidad): si está más cerca que
               VEHICLE_STOP_DIST el vehículo se detiene; entre STOP y SLOW
               reduce la velocidad proporcionalmente.

La velocidad NO la produce el modelo: es una crucero constante modulada solo
por las capas de seguridad.

Comandos de navegación (códigos de Codevilla):
    F = FOLLOW (2)   L = LEFT (3)   R = RIGHT (4)   T = STRAIGHT (5)

Teclado:
    F / L / R / T   comando de navegación
    ↑ / ↓           subir / bajar la velocidad de crucero
    E               habilitar / deshabilitar la evasión de obstáculos
    P               pausa (mantener detenido) / reanudar

NOTA: ejecutar con un Python que tenga TensorFlow/Keras + OpenCV (el mismo
entorno del entrenamiento). El modelo `cil_model.keras` y `model_config.json`
deben estar junto a este archivo (o ajustar las rutas abajo).
"""

from __future__ import print_function

import os
import json
import math

import numpy as np
import cv2

from controller import Display, Keyboard
from vehicle import Driver


# =====================================================================
#  RUTAS DEL MODELO
# =====================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "cil_model.keras")
CONFIG_PATH = os.path.join(HERE, "model_config.json")


# =====================================================================
#  PARÁMETROS DE CONDUCCIÓN
# =====================================================================
TARGET_SPEED = 30.0           # velocidad de crucero constante (km/h)
MAX_SPEED = 30.0

# Comandos (códigos Codevilla) y orden de las ramas del modelo.
CMD_FOLLOW, CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT = 2, 3, 4, 5
CMD_NAMES = {2: "FOLLOW", 3: "LEFT", 4: "RIGHT", 5: "STRAIGHT"}


# =====================================================================
#  CAPA 1 — PEATÓN (Recognition + LiDAR)  [act. 3.1]
# =====================================================================
PED_BRAKE_DIST = 12.0         # m: peatón reconocido a < esta distancia -> freno


# =====================================================================
#  CAPA 2 — EVASIÓN DE OBSTÁCULO  [act. 4.2]
# =====================================================================
EVADE_TRIGGER_DIST = 18.0     # m: distancia LiDAR para iniciar la evasión
AVOID_SPEED = 20.0            # km/h durante la maniobra
SHIFT_YAW = 0.35              # rad: apertura del cambio de carril a la izq.
WALL_TARGET = 2.5             # m: distancia lateral objetivo al obstáculo
DS_FREE_FRONT = 7.5
DS_FREE_SIDE = 5.5
KD_WALL = 0.06
KA_WALL = 0.10
K_YAW = 0.8
YAW_DONE = 0.04

ST_DRIVE = "MODEL_DRIVE"
ST_SHIFT = "SHIFT_LEFT"
ST_WALL = "WALL_FOLLOW"
ST_RECOVER = "RECOVER_HEADING"


# =====================================================================
#  CAPA 3 — DISTANCIA DE UMBRAL AL VEHÍCULO (ACC)  [radar + LiDAR]
# =====================================================================
# DISTANCIA DE UMBRAL a reportar en el video: si el vehículo más próximo está
# a menos de VEHICLE_STOP_DIST, el auto se DETIENE. Entre STOP y SLOW reduce
# proporcionalmente la velocidad.
VEHICLE_STOP_DIST = 10.0      # m  <-- distancia de umbral
VEHICLE_SLOW_DIST = 20.0      # m


# =====================================================================
#  PREPROCESAMIENTO (idéntico al de la Fase B)
# =====================================================================

def make_preprocess(cfg):
    ct, cb = cfg["crop_top"], cfg["crop_bottom"]
    iw, ih = cfg["img_w"], cfg["img_h"]

    def preprocess(img_bgr):
        h = img_bgr.shape[0]
        img = img_bgr[ct:h - cb, :, :]
        img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    return preprocess


# =====================================================================
#  SENSORES
# =====================================================================

def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))


def lidar_front_distance(lidar):
    """Distancia mínima en el sector central (tercio central) del LiDAR."""
    ranges = lidar.getRangeImage()
    n = len(ranges)
    window = ranges[n // 3: 2 * n // 3]
    finite = [r for r in window if math.isfinite(r)]
    return min(finite) if finite else float('inf')


def find_recognition(camera, models):
    """Devuelve el primer objeto reconocido cuyo modelo esté en `models`."""
    for obj in camera.getRecognitionObjects():
        if str(obj.getModel()).strip().lower() in models:
            return obj
    return None


def nearest_radar_target(radar, az_limit=0.4):
    """Objetivo de radar más próximo dentro de un cono frontal.
    Devuelve (distancia, velocidad_relativa) o (inf, 0)."""
    best_d, best_v = float('inf'), 0.0
    for tgt in radar.getTargets():
        if abs(tgt.azimuth) <= az_limit and tgt.distance < best_d:
            best_d, best_v = tgt.distance, tgt.speed
    return best_d, best_v


def heading_steer(yaw, target_yaw, limit=0.25):
    """Control P de orientación con el yaw integrado del giroscopio."""
    return float(np.clip(K_YAW * (yaw - target_yaw), -limit, limit))


# =====================================================================
#  DISPLAY
# =====================================================================

def show(display, img_bgra, hud_lines):
    dw, dh = display.getWidth(), display.getHeight()
    bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
    vis = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_NEAREST)
    y = 14
    for txt, color in hud_lines:
        cv2.putText(vis, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
        y += 16
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    ref = display.imageNew(vis_rgb.tobytes(), Display.RGB, width=dw, height=dh)
    display.imagePaste(ref, 0, 0, False)


def drain_keyboard(kb):
    keys, k = [], kb.getKey()
    while k != -1:
        keys.append(k & 0x0000FFFF)
        k = kb.getKey()
    return keys


# =====================================================================
#  MAIN
# =====================================================================

def main():
    # --- Modelo CIL ---
    from tensorflow import keras
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    model = keras.models.load_model(MODEL_PATH)
    preprocess = make_preprocess(cfg)
    commands = cfg["commands"]
    cmd_index = {c: i for i, c in enumerate(commands)}
    n_cmd = len(commands)
    max_steer = cfg.get("max_steer", 0.5)
    print(f"[INIT] Modelo cargado: {MODEL_PATH}")

    driver = Driver()
    timestep = int(driver.getBasicTimeStep())
    dt = timestep / 1000.0

    # --- Dispositivos ---
    camera = driver.getDevice("camera")
    camera.enable(timestep)
    if camera.hasRecognition():
        camera.recognitionEnable(timestep)
        print("[INIT] Recognition habilitado")
    else:
        print("[INIT] ADVERTENCIA: la cámara no tiene Recognition")

    lidar = driver.getDevice("lidar")
    lidar.enable(timestep)

    radar = None
    try:
        radar = driver.getDevice("radar")
        radar.enable(timestep)
        print("[INIT] Radar habilitado")
    except Exception:
        print("[INIT] Radar no disponible (se usa solo LiDAR)")

    gyro = driver.getDevice("gyro")
    gyro.enable(timestep)

    ds_front = driver.getDevice("ds_right_front")
    ds_mid = driver.getDevice("ds_right_mid")
    ds_rear = driver.getDevice("ds_right_rear")
    for ds in (ds_front, ds_mid, ds_rear):
        ds.enable(timestep)

    disp = driver.getDevice("display")
    kb = Keyboard()
    kb.enable(timestep)

    # --- Estado ---
    command = CMD_FOLLOW
    speed = TARGET_SPEED
    paused = False
    evade_enabled = True

    state = ST_DRIVE
    yaw = 0.0
    yaw_ref = 0.0
    rear_engaged = False
    prev_keys = set()             # para detectar flancos en teclas de toggle

    print("[INIT] F/L/R/T comando | ↑/↓ velocidad | E evasión | P pausa")
    print(f"[INIT] Distancia de umbral (stop): {VEHICLE_STOP_DIST} m")

    # ======================== BUCLE ========================
    while driver.step() != -1:
        t = driver.getTime()
        yaw += gyro.getValues()[2] * dt
        img = get_image(camera)

        # --- Sensores de percepción ---
        front_dist = lidar_front_distance(lidar)
        pedestrian = find_recognition(camera, {"pedestrian"})
        obstacle = find_recognition(camera, {"bus", "truck", "car"})
        radar_d, radar_v = (nearest_radar_target(radar)
                            if radar is not None else (float('inf'), 0.0))
        d_f, d_m, d_r = ds_front.getValue(), ds_mid.getValue(), ds_rear.getValue()

        # --- Teclado ---
        keys = drain_keyboard(kb)
        pressed = set(keys) - prev_keys      # teclas con flanco (toggles)
        prev_keys = set(keys)
        if ord('F') in keys:
            command = CMD_FOLLOW
        elif ord('L') in keys:
            command = CMD_LEFT
        elif ord('R') in keys:
            command = CMD_RIGHT
        elif ord('T') in keys:
            command = CMD_STRAIGHT
        if kb.UP in keys:
            speed = min(speed + 1.0, MAX_SPEED)
        if kb.DOWN in keys:
            speed = max(speed - 1.0, 0.0)
        if ord('E') in pressed:
            evade_enabled = not evade_enabled
            print(f"[EVADE] {'ON' if evade_enabled else 'OFF'}")
        if ord('P') in pressed:
            paused = not paused
            print(f"[PAUSE] {'detenido' if paused else 'en marcha'}")

        # --- Dirección del MODELO CIL ---
        x_img = np.expand_dims(preprocess(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)), 0)
        x_cmd = np.zeros((1, n_cmd), np.float32)
        # El modelo se entrenó con 3 comandos (FOLLOW/LEFT/RIGHT). STRAIGHT no
        # tiene rama propia: "seguir derecho en el cruce" se comporta igual que
        # seguir el carril, así que el comando STRAIGHT se enruta a FOLLOW.
        model_cmd = command if command in cmd_index else CMD_FOLLOW
        x_cmd[0, cmd_index[model_cmd]] = 1.0
        # Llamada directa al modelo (más rápida que predict() para 1 muestra).
        steer_model = float(model([x_img, x_cmd], training=False)[0, 0])
        steer_model = float(np.clip(steer_model, -max_steer, max_steer))

        # ============ ARBITRACIÓN DE COMPORTAMIENTO ============
        steering = steer_model
        target_speed = speed
        reason = "MODEL"

        # ---- CAPA 2: máquina de estados de evasión (tiene prioridad de giro)
        if state != ST_DRIVE:
            if state == ST_SHIFT:
                steering = heading_steer(yaw, yaw_ref + SHIFT_YAW)
                target_speed = AVOID_SPEED
                reason = "EVADE/SHIFT"
                if yaw - yaw_ref >= SHIFT_YAW - 0.03:
                    state = ST_WALL
            elif state == ST_WALL:
                target_speed = AVOID_SPEED
                reason = "EVADE/WALL"
                wall_visible = d_m < DS_FREE_SIDE or d_f < DS_FREE_FRONT
                if wall_visible:
                    d_side = min(d_f, d_m)
                    steering = KD_WALL * (d_side - WALL_TARGET)
                    if d_f < DS_FREE_FRONT and d_r < DS_FREE_SIDE:
                        steering += KA_WALL * (d_f - d_r)
                    steering = float(np.clip(steering, -0.15, 0.15))
                else:
                    hold = yaw_ref if rear_engaged else yaw_ref + SHIFT_YAW
                    steering = heading_steer(yaw, hold)
                if d_r < DS_FREE_SIDE:
                    rear_engaged = True
                if (rear_engaged and d_r >= DS_FREE_SIDE and
                        d_m >= DS_FREE_SIDE and d_f >= DS_FREE_FRONT):
                    state = ST_RECOVER
            elif state == ST_RECOVER:
                steering = heading_steer(yaw, yaw_ref)
                target_speed = AVOID_SPEED
                reason = "EVADE/RECOVER"
                if abs(yaw - yaw_ref) < YAW_DONE:
                    state = ST_DRIVE
        else:
            # Disparo de la evasión: obstáculo estacionado al frente.
            if (evade_enabled and obstacle is not None and
                    front_dist < EVADE_TRIGGER_DIST and pedestrian is None):
                yaw_ref = yaw
                rear_engaged = False
                state = ST_SHIFT
                print(f"[EVADE] obstáculo '{obstacle.getModel()}' a "
                      f"{front_dist:.1f} m -> inicio evasión")

        # ---- CAPA 3: ACC por distancia de umbral (modula la velocidad) ----
        vehicle_dist = min(front_dist, radar_d)
        if state == ST_DRIVE:
            if vehicle_dist < VEHICLE_STOP_DIST:
                target_speed = 0.0
                reason = "ACC/STOP"
            elif vehicle_dist < VEHICLE_SLOW_DIST:
                frac = ((vehicle_dist - VEHICLE_STOP_DIST) /
                        (VEHICLE_SLOW_DIST - VEHICLE_STOP_DIST))
                target_speed = min(target_speed, speed * frac)
                reason = "ACC/SLOW"

        # ---- CAPA 1: peatón -> freno de emergencia (máxima prioridad) ----
        if pedestrian is not None and front_dist < PED_BRAKE_DIST:
            target_speed = 0.0
            reason = "PEDESTRIAN/BRAKE"

        if paused:
            target_speed = 0.0
            reason = "PAUSED"

        # --- Enviar al vehículo ---
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(target_speed)

        # --- HUD ---
        ped_txt = "PED!" if pedestrian is not None else "-"
        show(disp, img, [
            (f"CMD {CMD_NAMES[command]}  {reason}", (255, 255, 0)),
            (f"steer {steering:+.3f}  v {target_speed:.0f}", (255, 255, 255)),
            (f"lidar {front_dist:5.1f}  veh {vehicle_dist:5.1f}", (0, 255, 255)),
            (f"radar d{radar_d:5.1f} v{radar_v:+.1f}  ped {ped_txt}",
             (0, 200, 255)),
        ])

        # --- Telemetría (~cada 0.5 s) ---
        if abs(t % 0.5) < dt:
            print(f"[t={t:7.2f}] cmd={CMD_NAMES[command]:<8s} st={state:<12s} "
                  f"{reason:<16s} steer={steering:+.3f} v={target_speed:.0f} "
                  f"lidar={front_dist:5.1f} veh={vehicle_dist:5.1f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("=" * 50)
        print("ERROR EN EL CONTROLADOR:")
        print(traceback.format_exc())
        print("=" * 50)
