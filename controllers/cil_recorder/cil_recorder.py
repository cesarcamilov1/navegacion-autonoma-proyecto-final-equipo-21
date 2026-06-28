"""
CIL Recorder — Fase A del proyecto final (Conditional Imitation Learning).

Controlador de RECOLECCIÓN de dataset para el mundo #1.

Basado en el controlador de la actividad 2.1, pero SIN PID: el vehículo se
conduce MANUALMENTE con el teclado mientras se inyectan comandos de
navegación (estilo Codevilla). En cada periodo de captura se guarda:

    - la imagen de la cámara a bordo (JPG),
    - el ángulo de dirección (steering) en ese instante,
    - el comando de navegación activo,

y se registra una fila en un CSV. La velocidad se mantiene CONSTANTE y baja
(<= 30 km/h) y NO forma parte del entrenamiento.

Códigos de comando (mismos enteros que usa Codevilla en CARLA):
    2 = FOLLOW    (seguir el carril, sin decisión de cruce)
    3 = LEFT      (girar a la izquierda en la siguiente intersección)
    4 = RIGHT     (girar a la derecha en la siguiente intersección)
    5 = STRAIGHT  (seguir derecho a través de la intersección)

Teclado:
    ←  / →            dirección (girar izquierda / derecha)
    ↑  / ↓            acelerar / frenar (la velocidad se acota a <= MAX_SPEED)
    F                 comando FOLLOW   (2)
    L                 comando LEFT     (3)
    R                 comando RIGHT    (4)
    T                 comando STRAIGHT (5)   ("T" = through)
    SPACE             pausar / reanudar la GRABACIÓN
    B                 mantener detenido (reposicionar) / soltar

El dataset se guarda en  proyecto-final/dataset/  con un prefijo de sesión
para que varios integrantes del equipo puedan grabar sin pisarse y luego
unir los CSV.
"""

from __future__ import print_function

import os
import csv
import socket
import datetime

import numpy as np
import cv2

from controller import Display, Keyboard
from vehicle import Driver


# =====================================================================
#  PARÁMETROS DE CONDUCCIÓN MANUAL
# =====================================================================

# Velocidad de crucero constante durante la recolección (km/h). El enunciado
# pide mantenerla baja (<= 30) y NO la usa el modelo.
TARGET_SPEED = 30.0
MAX_SPEED = 30.0

# Límite de dirección. El controlador 2.1 usaba 0.36, pero los giros del
# centro de ambos mundos son abruptos; subimos el límite para poder
# capturarlos. El BmwX5 admite holgadamente este rango.
MAX_STEER = 0.5

# Incremento de dirección por ciclo mientras se mantiene una flecha (16 ms).
# En ~14 ciclos (~0.22 s) se alcanza el giro máximo.
STEER_STEP = 0.035

# Cuando NO se presiona dirección, el volante retorna suavemente al centro.
# Así los tramos rectos quedan etiquetados con steering ~ 0.
STEER_CENTER_DECAY = 0.88


# =====================================================================
#  PARÁMETROS DE CAPTURA
# =====================================================================

# Periodo de captura de imágenes (s). A 16 ms de timestep, 0.1 s equivale a
# ~10 imágenes/segundo, suficiente para clonación de comportamiento y sin
# saturar el disco ni la PC. Si hace falta, súbelo.
CAPTURE_PERIOD_S = 0.1

# Calidad JPG (80-95 es buen compromiso tamaño/calidad para BC).
JPG_QUALITY = 92

# Comandos de navegación (códigos compatibles con Codevilla / CARLA).
CMD_FOLLOW = 2
CMD_LEFT = 3
CMD_RIGHT = 4
CMD_STRAIGHT = 5

CMD_NAMES = {
    CMD_FOLLOW: "FOLLOW",
    CMD_LEFT: "LEFT",
    CMD_RIGHT: "RIGHT",
    CMD_STRAIGHT: "STRAIGHT",
}


# Carpeta del dataset, relativa a la raíz del proyecto (dos niveles arriba de
# este archivo: controllers/cil_recorder/ -> proyecto-final/).
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
IMG_DIR = os.path.join(DATASET_DIR, "img")
CSV_PATH = os.path.join(DATASET_DIR, "driving_log.csv")

CSV_HEADER = [
    "session", "frame", "time",
    "image", "command", "command_name",
    "steering", "speed", "gps_x", "gps_y",
]


# =====================================================================
#  CÁMARA / DISPLAY
# =====================================================================

def get_image(camera):
    """Captura la imagen BGRA de la cámara como arreglo (H, W, 4)."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))


def show(display, img_bgra, command_name, steering, speed, recording, frames):
    """Dibuja la imagen de la cámara y el HUD de estado en el Display."""
    dw, dh = display.getWidth(), display.getHeight()
    bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
    vis = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_NEAREST)

    rec_txt = "REC" if recording else "PAUSE"
    rec_color = (0, 0, 255) if recording else (160, 160, 160)
    cv2.putText(vis, rec_txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, rec_color, 1, cv2.LINE_AA)
    cv2.putText(vis, f"CMD: {command_name}", (4, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"steer {steering:+.3f}", (4, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"v {speed:.0f} km/h", (4, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"n={frames}", (4, dh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    ref = display.imageNew(vis_rgb.tobytes(), Display.RGB, width=dw, height=dh)
    display.imagePaste(ref, 0, 0, False)


# =====================================================================
#  TECLADO
# =====================================================================

def drain_keyboard(kb):
    """Devuelve TODAS las teclas presionadas en este ciclo (Webots entrega
    una por llamada hasta -1). Se descartan los bits de modificadores."""
    keys = []
    k = kb.getKey()
    while k != -1:
        keys.append(k & 0x0000FFFF)
        k = kb.getKey()
    return keys


# =====================================================================
#  MAIN
# =====================================================================

def main():
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())   # 16 ms en el mundo #1
    dt = timestep / 1000.0

    # --- Dispositivos ---
    camera = driver.getDevice("camera")
    camera.enable(timestep)
    W, H = camera.getWidth(), camera.getHeight()
    print(f"[INIT] Cámara: {W}x{H}")

    disp = driver.getDevice("display")
    print(f"[INIT] Display: {disp.getWidth()}x{disp.getHeight()}")

    gps = driver.getDevice("gps")
    gps.enable(timestep)

    kb = Keyboard()
    kb.enable(timestep)

    # --- Dataset en disco ---
    os.makedirs(IMG_DIR, exist_ok=True)
    session = "{host}_{ts}".format(
        host=socket.gethostname().split(".")[0],
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    csv_is_new = not os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if csv_is_new:
        writer.writerow(CSV_HEADER)
    print(f"[INIT] Dataset: {DATASET_DIR}")
    print(f"[INIT] Sesión: {session}")

    # --- Estado de conducción ---
    steering = 0.0
    speed = TARGET_SPEED
    command = CMD_FOLLOW
    recording = True
    holding = False              # mantener detenido para reposicionar
    frames = 0
    last_capture_t = -1.0
    prev_keys = set()            # para detectar flancos en teclas de toggle

    print("[INIT] ←/→ dirección | ↑/↓ velocidad | F/L/R/T comando | "
          "SPACE rec | B hold")

    # ======================== BUCLE ========================
    while driver.step() != -1:
        t = driver.getTime()
        img = get_image(camera)

        keys = drain_keyboard(kb)
        pressed = set(keys) - prev_keys      # teclas con flanco de bajada->subida
        prev_keys = set(keys)

        # --- Dirección ---
        steer_input = False
        if kb.LEFT in keys:
            steering -= STEER_STEP
            steer_input = True
        if kb.RIGHT in keys:
            steering += STEER_STEP
            steer_input = True
        if not steer_input:
            # Retorno suave al centro: tramos rectos quedan en steering ~ 0.
            steering *= STEER_CENTER_DECAY
        steering = float(np.clip(steering, -MAX_STEER, MAX_STEER))

        # --- Velocidad ---
        if kb.UP in keys:
            speed = min(speed + 1.0, MAX_SPEED)
        if kb.DOWN in keys:
            speed = max(speed - 1.0, 0.0)

        # --- Comandos de navegación ---
        if ord('F') in keys:
            command = CMD_FOLLOW
        elif ord('L') in keys:
            command = CMD_LEFT
        elif ord('R') in keys:
            command = CMD_RIGHT
        elif ord('T') in keys:
            command = CMD_STRAIGHT

        # --- Grabación / hold (flanco: una sola acción por pulsación) ---
        if ord(' ') in pressed:
            recording = not recording
            print(f"[REC] {'ON' if recording else 'OFF'}")
        if ord('B') in pressed:
            holding = not holding
            print(f"[HOLD] {'detenido' if holding else 'liberado'}")

        # --- Enviar comandos al vehículo ---
        applied_speed = 0.0 if holding else speed
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(applied_speed)

        # --- Captura periódica al dataset ---
        capture = (recording and not holding and
                   (last_capture_t < 0 or t - last_capture_t >= CAPTURE_PERIOD_S))
        if capture:
            last_capture_t = t
            fname = f"{session}_{frames:06d}.jpg"
            fpath = os.path.join(IMG_DIR, fname)
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(fpath, bgr, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])

            gx, gy, _ = gps.getValues()
            writer.writerow([
                session, frames, f"{t:.3f}",
                os.path.join("img", fname), command, CMD_NAMES[command],
                f"{steering:.5f}", f"{applied_speed:.2f}",
                f"{gx:.3f}", f"{gy:.3f}",
            ])
            frames += 1
            if frames % 50 == 0:
                csv_file.flush()

        # --- HUD ---
        show(disp, img, CMD_NAMES[command], steering, applied_speed,
             recording and not holding, frames)

        # --- Telemetría (~cada 1 s) ---
        if abs(t % 1.0) < dt:
            print(f"[t={t:7.2f}] cmd={CMD_NAMES[command]:<8s} "
                  f"steer={steering:+.3f} v={applied_speed:.0f} "
                  f"rec={'ON' if recording else 'off':<3s} n={frames}")

    csv_file.flush()
    csv_file.close()
    print(f"[FIN] {frames} imágenes guardadas en {IMG_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("=" * 50)
        print("ERROR EN EL CONTROLADOR:")
        print(traceback.format_exc())
        print("=" * 50)
