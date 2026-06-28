"""
Re-etiquetado de comandos de navegación a partir de la trayectoria GPS.

PROBLEMA QUE RESUELVE
---------------------
Durante la recolección manual (Fase A) el comando de navegación se anota
presionando una tecla (F/L/R/T). Si el conductor no respeta la disciplina de
comando, la columna `command` queda mal (p.ej. todo RIGHT, o todo FOLLOW),
pese a que el auto físicamente SÍ giró a ambos lados.

SOLUCIÓN
--------
La maniobra REALMENTE ejecutada se recupera de la trayectoria: el GPS está
guardado por frame. Calculamos el cambio de rumbo (heading) en una ventana
alrededor de cada frame y clasificamos:

    cambio de rumbo fuerte (+)  -> LEFT
    cambio de rumbo fuerte (-)  -> RIGHT
    cambio de rumbo pequeño     -> FOLLOW

La ventana mira un poco hacia ATRÁS y bastante hacia ADELANTE, de modo que los
frames de ACERCAMIENTO al cruce quedan etiquetados con la maniobra que viene
(igual que la "intermitente antes del giro" de Codevilla) y los de EJECUCIÓN
también.

POR SESIÓN (importante al unir varios datasets del equipo)
----------------------------------------------------------
Cada sesión se procesa POR SEPARADO. Si se concatenaran las trayectorias de
dos conductores, la frontera entre ellas sería un salto de posición que el
cálculo de rumbo leería como un giro falso. Agrupando por la columna `session`
se evita ese artefacto.

SIGNO AUTO-CALIBRADO
--------------------
La convención izquierda/derecha (qué signo de cambio de rumbo es un giro a la
izquierda) se calibra automáticamente correlacionando el cambio de rumbo con el
steering humano en los frames de giro fuerte. No hay que tocar constantes.

NO DESTRUCTIVO: lee driving_log.csv y escribe driving_log_relabeled.csv. El
steering (la etiqueta que entrena el modelo) NO se toca. Se corrige `command` /
`command_name` y se normalizan las rutas de imagen a '/' (las sesiones grabadas
en Windows guardan 'img\\..', que cv2.imread no encuentra en Linux/Colab).

NOTA: no genera STRAIGHT. Desde el GPS no se distingue de forma fiable "cruzar
derecho una intersección" de "seguir el carril". El modelo usa 3 comandos
(FOLLOW/LEFT/RIGHT) y el controlador de inferencia enruta STRAIGHT a FOLLOW.
"""

import csv
import math
import os
import collections
import statistics

# --- Rutas ---
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, os.pardir, "dataset")
SRC_CSV = os.path.join(DATASET, "driving_log.csv")
OUT_CSV = os.path.join(DATASET, "driving_log_relabeled.csv")

# --- Códigos de comando (Codevilla / CARLA) ---
CMD = {"FOLLOW": 2, "LEFT": 3, "RIGHT": 4, "STRAIGHT": 5}

# --- Parámetros de detección (ajustables) ---
HEADING_K = 4        # frames a cada lado para estimar el rumbo (suaviza ruido)
YAW_W = 8            # ventana centrada para el yaw de calibración de signo
WIN_BACK = 6         # frames hacia atrás de la ventana de etiquetado
WIN_FWD = 22         # frames hacia adelante (~2.2 s @10 Hz: anticipa la maniobra)
TURN_TH = 0.50       # rad: cambio de rumbo neto para considerar GIRO (~29°)
MIN_RUN = 5          # frames mínimos de un bloque de giro (quita parpadeos)
GAP_FILL = 8         # frames: une bloques de giro separados por huecos chicos
CALIB_STEER = 0.12   # |steer| mínimo para usar un frame en la calibración


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def heading(xs, ys, i):
    n = len(xs)
    j = min(i + HEADING_K, n - 1)
    i0 = max(i - HEADING_K, 0)
    dx, dy = xs[j] - xs[i0], ys[j] - ys[i0]
    return math.atan2(dy, dx) if (dx or dy) else None


def centered_yaw(xs, ys):
    """Tasa de giro centrada en cada frame (capta la EJECUCIÓN del giro)."""
    n = len(xs)
    yr = [0.0] * n
    for i in range(n):
        h0 = heading(xs, ys, max(i - YAW_W, 0))
        h1 = heading(xs, ys, min(i + YAW_W, n - 1))
        yr[i] = wrap(h1 - h0) if (h0 is not None and h1 is not None) else 0.0
    return yr


def lookahead_turn(xs, ys):
    """Cambio de rumbo neto en [i-WIN_BACK, i+WIN_FWD] (anticipa la maniobra)."""
    n = len(xs)
    out = [0.0] * n
    for i in range(n):
        h0 = heading(xs, ys, max(i - WIN_BACK, 0))
        h1 = heading(xs, ys, min(i + WIN_FWD, n - 1))
        out[i] = wrap(h1 - h0) if (h0 is not None and h1 is not None) else 0.0
    return out


def clean_labels(labels):
    """Une huecos cortos dentro de un giro y borra bloques de giro muy cortos."""
    n = len(labels)
    for turn in ("LEFT", "RIGHT"):
        i = 0
        while i < n:
            if labels[i] == turn:
                j = i
                while j < n and labels[j] == turn:
                    j += 1
                k = j
                while k < n and k - j < GAP_FILL and labels[k] != turn:
                    k += 1
                if k < n and labels[k] == turn and (k - j) <= GAP_FILL:
                    for m in range(j, k):
                        labels[m] = turn
                    i = k
                else:
                    i = j
            else:
                i += 1
    i = 0
    while i < n:
        if labels[i] in ("LEFT", "RIGHT"):
            j = i
            while j < n and labels[j] == labels[i]:
                j += 1
            if j - i < MIN_RUN:
                for m in range(i, j):
                    labels[m] = "FOLLOW"
            i = j
        else:
            i += 1
    return labels


def label_session(xs, ys, pos_is_left):
    turn = lookahead_turn(xs, ys)
    labels = []
    for d in turn:
        if d > TURN_TH:
            labels.append("LEFT" if pos_is_left else "RIGHT")
        elif d < -TURN_TH:
            labels.append("RIGHT" if pos_is_left else "LEFT")
        else:
            labels.append("FOLLOW")
    return clean_labels(labels)


def events(labs, turn):
    c, inb = 0, False
    for l in labs:
        if l == turn and not inb:
            c += 1
            inb = True
        elif l != turn:
            inb = False
    return c


def main():
    with open(SRC_CSV) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    n = len(rows)

    # Agrupar índices por sesión, preservando el orden del archivo.
    sessions = collections.OrderedDict()
    for i, r in enumerate(rows):
        sessions.setdefault(r["session"], []).append(i)

    # Pre-cálculo del yaw centrado por sesión (sin saltos de frontera) y
    # recolección de pares (yaw, steer) para calibrar el signo izq/der.
    sess_xy = {}
    left_yr, right_yr = [], []
    for s, idxs in sessions.items():
        xs = [float(rows[i]["gps_x"]) for i in idxs]
        ys = [float(rows[i]["gps_y"]) for i in idxs]
        yr = centered_yaw(xs, ys)
        sess_xy[s] = (xs, ys)
        for k, i in enumerate(idxs):
            st = float(rows[i]["steering"])
            if st < -CALIB_STEER:
                left_yr.append(yr[k])
            elif st > CALIB_STEER:
                right_yr.append(yr[k])
    pos_is_left = (statistics.mean(left_yr) > 0) if left_yr else True

    # Etiquetar cada sesión por separado.
    labels = [""] * n
    for s, idxs in sessions.items():
        xs, ys = sess_xy[s]
        labs = label_session(xs, ys, pos_is_left)
        for k, i in enumerate(idxs):
            labels[i] = labs[k]

    # Escribir CSV nuevo (no destructivo): corrige comando + normaliza rutas.
    before = collections.Counter(r["command_name"] for r in rows)
    after = collections.Counter(labels)
    for r, lab in zip(rows, labels):
        r["command"] = CMD[lab]
        r["command_name"] = lab
        r["image"] = r["image"].replace("\\", "/")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Reporte.
    print(f"Frames: {n}   Sesiones: {len(sessions)}")
    print(f"Convención calibrada: yaw+ = "
          f"{'IZQUIERDA' if pos_is_left else 'DERECHA'}")
    print("\n            ANTES (tecla)     ->   DESPUÉS (GPS)")
    for k in ("FOLLOW", "STRAIGHT", "LEFT", "RIGHT"):
        print(f"  {k:9s} {before.get(k,0):6d} ({100*before.get(k,0)/n:5.1f}%)"
              f"   ->   {after.get(k,0):6d} ({100*after.get(k,0)/n:5.1f}%)")
    print("\nEventos de giro por sesión:")
    for s, idxs in sessions.items():
        labs = [labels[i] for i in idxs]
        print(f"  {s:28s} LEFT {events(labs,'LEFT'):3d}  "
              f"RIGHT {events(labs,'RIGHT'):3d}   ({len(idxs)} frames)")
    print(f"\nEscrito: {OUT_CSV}")


if __name__ == "__main__":
    main()
