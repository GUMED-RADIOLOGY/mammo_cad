# mammo_cad
pre opus paper

# GPU Dev Container (PyTorch + DICOM) — README

* Runs as **your user** (no root-owned files on mounts)
* **CUDA / PyTorch** ready
* **DICOM** stack (pydicom, pylibjpeg, GDCM, dcmtk)
* **Git + GitHub CLI (`gh`)** inside
* No Jupyter—pure CLI. One-liners for shell, one-off runs, and detached jobs.

---

## Prereqs

* **NVIDIA driver** new enough for CUDA 12.x (e.g. 570+ for 12.8)
* **Docker** + **Docker Compose v2**
* **NVIDIA Container Toolkit** configured for Docker

Quick sanity check (host):

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

---

## Quick start

Build once:

```bash
docker compose build --pull
```

Open a dev shell (auto-remove on exit):

```bash
./dev
```

Run a one-off command:

```bash
./dev run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run your training script:

```bash
./dev run python src/ensambling_model.py
```

Detached/background job:

```bash
job=$(./dev bg python src/ensambling_model.py --epochs 10)
./dev logs "$job"   # follow logs
./dev stop "$job"   # stop early (auto-removes)
```

---

## Git + GitHub

Authenticate once (inside the container):

```bash
gh auth login
```

Clone your repo into `/app` (the project root):

```bash
cd /app
gh repo clone <your-user>/<your-repo> .
```

Typical git flow:

```bash
git checkout -b feature/my-change
git add .
git commit -m "feat: something useful"
git push -u origin main
gh pr create --fill --web
```
---

## What’s mounted

* `./` → `/app` (your code)
* `./.devhome` → `/home/dev` (ssh/gh/pip caches; keeps auth & settings)
* Add datasets in `docker-compose.yml` under `volumes:` (e.g. `/data:/data`)

## Cheatsheet

```bash
./dev              # shell
./dev run CMD...   # one-off command
./dev bg CMD...    # detached; prints name
./dev logs NAME    # follow logs
./dev stop NAME    # stop detached job
```
