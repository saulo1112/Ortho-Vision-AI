# OrthoVision AI

**English** | [Español](#español)

> Instance segmentation of orthopedic implants in radiographs — from a research
> model to a certified export, a torch-free API, and a native-feeling mobile app.

**⚠️ Educational / portfolio demo — not a medical device. Not for clinical use.**

## The problem

Surgeons who don't already know which implant a patient is carrying lose real
time finding out. A 2012 survey of hip and knee surgeons found an average of
20 minutes and at least three identification methods before getting it right,
with support staff needing 30. Close to 9 in 10 surgeons reported failing to
identify an implant before surgery at least once, a miss that means extra
devices brought into the operating room, longer procedures, more blood loss,
and slower recovery. OrthoVision AI segments three implant types directly from
a radiograph, automatically:

| Class | Label |
|---|---|
| `clavo_intramedular` | Intramedullary nail |
| `placa_atornillada` | Screwed plate |
| `protesis_articular` | Joint prosthesis |

## The dataset

No public dataset for segmenting orthopedic implants exists, so building one
came first. Over 4,000 candidate images were pulled from three sources: UCI's
Shoulder Implant X-ray Manufacturer dataset, Kaggle's Bone Fracture Detection
set filtered down to non-redundant, diagnostic-quality images, and targeted
MedlinePlus searches to cover intramedullary nails specifically, since the
first two barely showed them. Four team members hand-labeled 60 images each
on Roboflow under one agreed standard, yielding 240 curated images, split
70/15/15 and expanded to 408 through controlled rotation augmentation so the
model would see the same implant tilted the way a real radiograph tilts it.

The limiting factor was never volume, it was variety: the pooled images
skewed unevenly across anatomical regions and rarely captured one implant
from more than a single angle, which is documented as an explicit limitation
rather than smoothed over. See [docs/architecture.md](docs/architecture.md)
for the full pipeline and [model/notebooks/](model/notebooks/) for the
training notebook, including the ablation that tested (and did not confirm)
whether classical CLAHE preprocessing would improve segmentation.

---

## Highlights

- **Validated ONNX export.** The `.pt → ONNX` conversion is not taken on faith:
  a script runs both pipelines over the held-out test set and certifies they
  agree. Mask-to-mask agreement ≥ 0.996, per-class IoU/Dice delta ≤ 0.001.
  See [docs/onnx-parity.md](docs/onnx-parity.md).
- **Torch-free serving.** YOLOv8-seg post-processing (NMS, prototype masks,
  polygonization) is re-implemented in numpy + OpenCV, so the Docker image is
  ~300 MB and fits a 512 MB free-tier instance.
- **Premium mobile UI.** iOS-first design with native Liquid Glass, a clinical
  dark palette, and SVG mask overlays that scale to any screen.
- **Honest engineering.** Normalized coordinates, no-detection handling, EXIF
  correction, image-size limits, model versioning, and a medical disclaimer are
  all first-class, not afterthoughts.

## Results

Test set: 36 held-out radiographs (Roboflow v3), confidence threshold 0.5.

| Class | IoU (ONNX) | Dice (ONNX) | Agreement vs `.pt` |
|---|---|---|---|
| Intramedullary nail | 0.9596 | 0.9771 | 0.9990 |
| Screwed plate | 0.9098 | 0.9368 | 0.9968 |
| Joint prosthesis | 0.9377 | 0.9535 | 0.9974 |

Production engine latency (CPU): ~450 ms mean per image. Full report:
[docs/onnx-parity.md](docs/onnx-parity.md).

## Architecture

```
Expo app  ──POST /v1/predict (multipart)──►  FastAPI + ONNX Runtime  ──►  SQLite/Postgres
    ▲                                              │
    └────────── polygons + confidence ◄────────────┘
```

Details and diagram: [docs/architecture.md](docs/architecture.md). API contract:
[docs/api-contract.md](docs/api-contract.md).

```
ortho-vision-ai/
├── model/      # ONNX export + parity validation (torch, offline)
├── backend/    # FastAPI + ONNX Runtime, Dockerized (torch-free)
├── mobile/     # Expo / React Native app
└── docs/       # architecture, API contract, parity report
```

## Quickstart

### 1. Model → ONNX (once)

```bash
cd model
py -3.11 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python export_onnx.py            # writes weights/best.onnx

# Optional: reproduce the parity report (requires a Roboflow API key)
cp .env.example .env && $EDITOR .env           # set ROBOFLOW_API_KEY (gitignored)
.venv/Scripts/python download_dataset.py
.venv/Scripts/python validate_parity.py        # writes docs/onnx-parity.md
```

### 2. Backend

```bash
cd backend
py -3.11 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# --host 0.0.0.0 exposes the API on your LAN so a physical phone can reach it
.venv/Scripts/uvicorn app.main:app --reload --host 0.0.0.0   # http://localhost:8000/docs
.venv/Scripts/python -m pytest                               # 7 end-to-end tests
```

Or with Docker (build from the repo root so the weights are included):

```bash
docker build -f backend/Dockerfile -t orthovision-api .
docker run -p 8000:8000 orthovision-api
```

### 3. Mobile

```bash
cd mobile
npm install
npm run android       # dev build on a USB-connected Android device
```

> Note: Expo Go only supports a single SDK version, so an outdated Expo Go
> from the store may refuse this project (SDK 57). The dev build
> (`npm run android`) is the standard workflow here and enables all native
> modules. Requires Android Studio / SDK; always run it from `mobile/`, not
> the repo root.

#### Test on a physical device

The app probes candidate backend URLs at startup (`/health`) and uses the
first one that answers, in this order:

1. `expo.extra.apiUrl` from `mobile/app.json` — set this for a deployed
   backend (e.g. your Render URL).
2. `http://localhost:8000` — works over **USB**: `npm run android` runs
   `adb reverse tcp:8000 tcp:8000`, which tunnels the phone's port 8000 to
   your dev machine. Immune to firewalls and multi-adapter setups.
3. `http://<metro-host-ip>:8000` — same **Wi-Fi** network. Requires the
   backend started with `--host 0.0.0.0` and Windows Firewall allowing
   inbound TCP 8000 (`netsh advfirewall firewall add rule
   name="OrthoVision API" dir=in action=allow protocol=TCP localport=8000`,
   from an elevated shell).

Either way, start the backend first (`npm run backend` from the repo root).

## Deployment

The backend ships to [Render](https://render.com) via `render.yaml` (Docker web
service + managed Postgres). Point Render at this repo and it provisions both
from the blueprint; `DATABASE_URL` is injected automatically.

## Tech stack

**Model:** YOLOv8s-seg (ultralytics 8.0.196), ONNX Runtime.
**Backend:** FastAPI, ONNX Runtime, OpenCV, SQLAlchemy, Pydantic, Docker.
**Mobile:** Expo (SDK 57), Expo Router, react-native-svg, expo-glass-effect, TypeScript.

## Credits

Trained model and original research notebook by *Grupo 3 — Proyecto Final Visión
Computacional con Deep Learning*. Dataset via Roboflow. Productization,
backend, mobile app, and ONNX validation in this repository.

---

# Español

[English](#orthovision-ai) | **Español**

> Segmentación por instancias de implantes ortopédicos en radiografías, desde
> un modelo de investigación hasta una exportación certificada, una API sin
> PyTorch y una app móvil con sensación nativa.

**⚠️ Demo educativa / de portafolio, no es un dispositivo médico. No apto para uso clínico.**

## El problema

Los cirujanos que no saben de antemano qué implante lleva un paciente pierden
tiempo real averiguándolo. Una encuesta de 2012 a cirujanos de cadera y
rodilla encontró un promedio de 20 minutos y al menos tres métodos de
identificación antes de acertar, con el personal de apoyo necesitando 30.
Cerca de 9 de cada 10 cirujanos reportaron haber fallado en identificar un
implante antes de una cirugía al menos una vez, un error que implica traer
dispositivos extra al quirófano, procedimientos más largos, más pérdida de
sangre y recuperación más lenta. OrthoVision AI segmenta tres tipos de
implante directamente desde una radiografía, de forma automática:

| Clase | Etiqueta |
|---|---|
| `clavo_intramedular` | Clavo intramedular |
| `placa_atornillada` | Placa atornillada |
| `protesis_articular` | Prótesis articular |

## El dataset

No existe un dataset público para segmentar implantes ortopédicos, así que
construir uno fue el primer paso. Se reunieron más de 4,000 imágenes
candidatas de tres fuentes: el dataset Shoulder Implant X-ray Manufacturer de
UCI, el dataset Bone Fracture Detection de Kaggle filtrado a imágenes no
redundantes y de calidad diagnóstica, y búsquedas dirigidas en MedlinePlus
para cubrir clavos intramedulares específicamente, ya que las dos primeras
fuentes apenas los mostraban. Cuatro integrantes del equipo etiquetaron a
mano 60 imágenes cada uno en Roboflow bajo un estándar acordado, produciendo
240 imágenes curadas, divididas 70/15/15 y ampliadas a 408 mediante
aumentación por rotación controlada, para que el modelo viera el mismo
implante inclinado como realmente se inclina en una radiografía.

El factor limitante nunca fue el volumen, fue la variedad: las imágenes
reunidas estaban distribuidas de forma desigual entre regiones anatómicas y
rara vez capturaban un mismo implante desde más de un ángulo, algo que se
documenta como limitación explícita en vez de disimularse. Ver
[docs/architecture.md](docs/architecture.md) para el pipeline completo y
[model/notebooks/](model/notebooks/) para el notebook de entrenamiento,
incluida la prueba de ablación que evaluó (y no confirmó) si el
preprocesamiento clásico con CLAHE mejoraba la segmentación.

---

## Puntos destacados

- **Exportación a ONNX validada.** La conversión `.pt → ONNX` no se da por
  sentada: un script corre ambos pipelines sobre el conjunto de prueba y
  certifica que coinciden. Concordancia máscara a máscara ≥ 0.996, delta de
  IoU/Dice por clase ≤ 0.001. Ver [docs/onnx-parity.md](docs/onnx-parity.md).
- **Servido sin PyTorch.** El post-procesamiento de YOLOv8-seg (NMS, máscaras
  prototipo, poligonización) está reimplementado en numpy + OpenCV, así que
  la imagen Docker pesa ~300 MB y cabe en una instancia gratuita de 512 MB.
- **UI móvil premium.** Diseño iOS-first con Liquid Glass nativo, paleta
  oscura clínica, y máscaras SVG que escalan a cualquier pantalla.
- **Ingeniería honesta.** Coordenadas normalizadas, manejo de detección nula,
  corrección EXIF, límites de tamaño de imagen, versionado del modelo y un
  descargo médico, todo de primera clase, no como ocurrencia tardía.

## Resultados

Conjunto de prueba: 36 radiografías separadas (Roboflow v3), umbral de
confianza 0.5.

| Clase | IoU (ONNX) | Dice (ONNX) | Concordancia vs `.pt` |
|---|---|---|---|
| Clavo intramedular | 0.9596 | 0.9771 | 0.9990 |
| Placa atornillada | 0.9098 | 0.9368 | 0.9968 |
| Prótesis articular | 0.9377 | 0.9535 | 0.9974 |

Latencia del motor de producción (CPU): ~450 ms en promedio por imagen.
Reporte completo: [docs/onnx-parity.md](docs/onnx-parity.md).

## Arquitectura

```
App Expo  ──POST /v1/predict (multipart)──►  FastAPI + ONNX Runtime  ──►  SQLite/Postgres
    ▲                                              │
    └────────── polígonos + confianza ◄────────────┘
```

Detalles y diagrama: [docs/architecture.md](docs/architecture.md). Contrato
de la API: [docs/api-contract.md](docs/api-contract.md).

```
ortho-vision-ai/
├── model/      # Exportación a ONNX + validación de paridad (torch, offline)
├── backend/    # FastAPI + ONNX Runtime, en Docker (sin torch)
├── mobile/     # App Expo / React Native
└── docs/       # arquitectura, contrato de API, reporte de paridad
```

## Inicio rápido

### 1. Modelo → ONNX (una vez)

```bash
cd model
py -3.11 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python export_onnx.py            # escribe weights/best.onnx

# Opcional: reproducir el reporte de paridad (requiere una API key de Roboflow)
cp .env.example .env && $EDITOR .env           # define ROBOFLOW_API_KEY (en .gitignore)
.venv/Scripts/python download_dataset.py
.venv/Scripts/python validate_parity.py        # escribe docs/onnx-parity.md
```

### 2. Backend

```bash
cd backend
py -3.11 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# --host 0.0.0.0 expone la API en tu red local para que un teléfono físico llegue a ella
.venv/Scripts/uvicorn app.main:app --reload --host 0.0.0.0   # http://localhost:8000/docs
.venv/Scripts/python -m pytest                               # 7 pruebas end-to-end
```

O con Docker (compilar desde la raíz del repo para que se incluyan los pesos):

```bash
docker build -f backend/Dockerfile -t orthovision-api .
docker run -p 8000:8000 orthovision-api
```

### 3. Móvil

```bash
cd mobile
npm install
npm run android       # dev build en un dispositivo Android conectado por USB
```

> Nota: Expo Go solo soporta una versión de SDK a la vez, así que un Expo Go
> desactualizado de la tienda puede rechazar este proyecto (SDK 57). El dev
> build (`npm run android`) es el flujo estándar aquí y habilita todos los
> módulos nativos. Requiere Android Studio / SDK; siempre ejecutar desde
> `mobile/`, no desde la raíz del repo.

#### Probar en un dispositivo físico

La app sondea backends candidatos al arrancar (`/health`) y usa el primero
que responde, en este orden:

1. `expo.extra.apiUrl` de `mobile/app.json` — configúralo para un backend
   desplegado (por ejemplo, tu URL de Render).
2. `http://localhost:8000` — funciona por **USB**: `npm run android` corre
   `adb reverse tcp:8000 tcp:8000`, que tunela el puerto 8000 del teléfono
   hacia tu máquina de desarrollo. Inmune a firewalls y configuraciones de
   múltiples adaptadores.
3. `http://<ip-del-host-metro>:8000` — misma red **Wi-Fi**. Requiere que el
   backend arranque con `--host 0.0.0.0` y que el Firewall de Windows permita
   el puerto TCP 8000 entrante (`netsh advfirewall firewall add rule
   name="OrthoVision API" dir=in action=allow protocol=TCP localport=8000`,
   desde una consola elevada).

En cualquier caso, arranca primero el backend (`npm run backend` desde la
raíz del repo).

## Despliegue

El backend se despliega en [Render](https://render.com) vía `render.yaml`
(servicio web Docker + Postgres administrado). Apunta Render a este repo y
aprovisiona ambos desde el blueprint; `DATABASE_URL` se inyecta
automáticamente.

## Stack tecnológico

**Modelo:** YOLOv8s-seg (ultralytics 8.0.196), ONNX Runtime.
**Backend:** FastAPI, ONNX Runtime, OpenCV, SQLAlchemy, Pydantic, Docker.
**Móvil:** Expo (SDK 57), Expo Router, react-native-svg, expo-glass-effect, TypeScript.

## Créditos

Modelo entrenado y notebook de investigación original por *Grupo 3 —
Proyecto Final Visión Computacional con Deep Learning*. Dataset vía
Roboflow. Productización, backend, app móvil y validación ONNX en este
repositorio.
