"""
Une un dataset externo (otra sesión del equipo) en proyecto-final/dataset/.

Uso:
    python training/merge_dataset.py <carpeta_dataset_externo>

Ejemplo:
    python training/merge_dataset.py ~/Downloads/dataset

IDEMPOTENTE: solo agrega las sesiones que NO estén ya en el dataset destino
(identificadas por la columna 'session'), así que correrlo dos veces no duplica
nada. Copia las imágenes nuevas a dataset/img/ y agrega las filas al
driving_log.csv. NO toca el steering ni re-etiqueta: después de unir hay que
correr training/relabel_from_gps.py sobre el combinado.

Funciona aunque las sesiones se hayan grabado en Windows (rutas con '\\') o en
Linux (rutas con '/'): la copia de imágenes usa solo el nombre de archivo, y el
re-etiquetado posterior normaliza las rutas.
"""

import csv
import os
import sys
import shutil
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, os.pardir, "dataset")


def main():
    if len(sys.argv) < 2:
        print("Uso: python merge_dataset.py <carpeta_dataset_externo>")
        sys.exit(1)

    src = os.path.abspath(os.path.expanduser(sys.argv[1]))
    src_csv = os.path.join(src, "driving_log.csv")
    src_img = os.path.join(src, "img")
    dest_csv = os.path.join(DEST, "driving_log.csv")
    dest_img = os.path.join(DEST, "img")
    os.makedirs(dest_img, exist_ok=True)

    if not os.path.exists(src_csv):
        print(f"No existe {src_csv}")
        sys.exit(1)

    src_rows = list(csv.DictReader(open(src_csv)))

    if os.path.exists(dest_csv):
        dest_rows = list(csv.DictReader(open(dest_csv)))
        cols = list(dest_rows[0].keys()) if dest_rows else list(src_rows[0].keys())
        have = set(r["session"] for r in dest_rows)
        new_csv = False
    else:
        cols = list(src_rows[0].keys())
        have = set()
        new_csv = True

    src_sessions = collections.OrderedDict()
    for r in src_rows:
        src_sessions.setdefault(r["session"], 0)
        src_sessions[r["session"]] += 1

    new_sessions = [s for s in src_sessions if s not in have]
    if not new_sessions:
        print("Nada que unir: esas sesiones ya están en el dataset destino.")
        print(f"  destino ya tiene sesiones: {sorted(have)}")
        return

    to_add = [r for r in src_rows if r["session"] in new_sessions]

    copied = missing = 0
    for r in to_add:
        name = os.path.basename(r["image"].replace("\\", "/"))
        s = os.path.join(src_img, name)
        d = os.path.join(dest_img, name)
        if not os.path.exists(s):
            missing += 1
            continue
        if not os.path.exists(d):
            shutil.copy2(s, d)
            copied += 1

    with open(dest_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_csv:
            w.writeheader()
        for r in to_add:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"Sesiones unidas: {new_sessions}")
    print(f"  filas agregadas: {len(to_add)}  "
          f"imágenes copiadas: {copied}  (faltantes en origen: {missing})")
    print(f"  destino: {dest_csv}")
    print("\nSiguiente paso:  python training/relabel_from_gps.py")


if __name__ == "__main__":
    main()
