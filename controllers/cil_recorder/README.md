# Fase A — Recolector de dataset (CIL)

Controlador de conducción **manual** para el mundo #1. Captura imágenes de la
cámara a bordo de forma automática y registra, por cada imagen, el **ángulo de
dirección** y el **comando de navegación** activo en un CSV. La velocidad se
mantiene constante (≤ 30 km/h) y **no** se entrena.

## Cómo ejecutarlo (controlador extern)

El mundo trae `controller "<extern>"`, así que el controlador se lanza por
fuera de Webots y se conecta al vehículo que está en pausa/play.

1. Abrí `worlds/city_traffic_2025_01.wbt` en Webots y dale **play**.
2. En una terminal, lanzá el controlador con el launcher de Webots (resuelve
   solo las variables de entorno y el `PYTHONPATH`):

   ```bash
   webots-controller "controllers/cil_recorder/cil_recorder.py"
   ```

   Si `webots-controller` no está en el PATH, suele estar en
   `"$WEBOTS_HOME/webots-controller"` (Linux) o dentro de la carpeta de
   instalación de Webots.

> Requiere `python-opencv` y `numpy` en el Python que use el launcher.

## Teclas

| Tecla        | Acción                                             |
|--------------|----------------------------------------------------|
| ← / →        | Girar a la izquierda / derecha                     |
| ↑ / ↓        | Subir / bajar velocidad (tope 30 km/h)             |
| **F**        | Comando `FOLLOW` (2) — seguir carril               |
| **L**        | Comando `LEFT` (3) — girar izq. en intersección    |
| **R**        | Comando `RIGHT` (4) — girar der. en intersección   |
| **T**        | Comando `STRAIGHT` (5) — derecho en intersección   |
| **SPACE**    | Pausar / reanudar la grabación                     |
| **B**        | Mantener detenido (reposicionar) / soltar          |

El volante **retorna solo al centro** cuando no presionás flechas, de modo que
los tramos rectos quedan etiquetados con `steering ≈ 0`.

## Flujo de comandos (igual que Codevilla)

Por defecto el comando es `FOLLOW`. Antes de llegar a una intersección, fijá el
comando del giro que vas a hacer (`L`, `R` o `T`), ejecutá el giro con las
flechas, y al salir volvé a `FOLLOW` con `F`. La etiqueta guardada en cada frame
es el comando activo en ese instante.

## Salida

```
proyecto-final/dataset/
├── img/                      # <session>_000000.jpg, ...
└── driving_log.csv          # session, frame, time, image, command,
                             # command_name, steering, speed, gps_x, gps_y
```

Cada sesión usa un prefijo `host_fecha_hora`, así varios integrantes pueden
grabar y luego **unir** sus carpetas `img/` y concatenar los CSV sin colisiones.

## Recomendaciones de recolección (del enunciado)

- Recorré **todo** el mundo en **ambos sentidos**, cubriendo todas las rutas.
- Incluí maniobras de **recuperación** (salir del carril y volver) para que el
  modelo aprenda a reencauzarse.
- Buscá un dataset **balanceado** entre giros izquierda / derecha / recto.
- Objetivo: dataset grande; tras *data augmentation* debe superar las 10 000
  imágenes.
