#!/usr/bin/env bash
# Setup du projet traffic-sign-recognition sur Google Colab.
# Idempotent : on peut le relancer sans risque.
#
# Usage (dans une cellule Colab) :
#   import os
#   from google.colab import userdata
#   os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")  # repo privé uniquement
#   !bash <(curl -fsSL -H "Authorization: token $GITHUB_TOKEN" \
#       https://raw.githubusercontent.com/<USER>/<REPO>/main/scripts/setup_colab.sh)
# ou, si le repo est déjà cloné :
#   !bash scripts/setup_colab.sh

set -euo pipefail

# ---- Configuration (à adapter) ----
REPO_URL="${REPO_URL:-https://github.com/<USER>/traffic-sign-recognition.git}"
BRANCH="${BRANCH:-main}"
TARGET_DIR="${TARGET_DIR:-/content/traffic-sign-recognition}"

log() { echo -e "\n\033[1;34m[setup]\033[0m $*"; }

# ---- 1. Clone ou mise à jour du repo ----
if [ -f "pyproject.toml" ] && [ -d "src/trafficsigns" ]; then
    TARGET_DIR="$(pwd)"
    log "Repo détecté dans $TARGET_DIR, pas de clone."
elif [ -d "$TARGET_DIR/.git" ]; then
    log "Repo déjà présent, mise à jour (branche $BRANCH)..."
    git -C "$TARGET_DIR" fetch origin
    git -C "$TARGET_DIR" checkout "$BRANCH"
    git -C "$TARGET_DIR" pull --ff-only origin "$BRANCH"
else
    log "Clone du repo (branche $BRANCH)..."
    CLONE_URL="$REPO_URL"
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        CLONE_URL="${REPO_URL/https:\/\//https://${GITHUB_TOKEN}@}"
    fi
    git clone --branch "$BRANCH" "$CLONE_URL" "$TARGET_DIR"
fi
cd "$TARGET_DIR"

# ---- 2. Installation de uv ----
if ! command -v uv >/dev/null 2>&1; then
    log "Installation de uv..."
    pip install -q uv
fi

# ---- 3. Installation du projet ----
# --system : on installe dans l'environnement Colab (celui du kernel du notebook).
# On réutilise ainsi TensorFlow / PyTorch / OpenCV déjà présents sur Colab.
log "Installation des dépendances (mode éditable)..."
uv pip install --system -q -e .

# ---- 4. Dossiers de travail (non versionnés) ----
mkdir -p data/raw data/processed models/baseline models/cnn_final models/yolo reports/figures

# ---- 5. Vérification ----
log "Vérification..."
python - <<'EOF'
import sys
import trafficsigns
print("Python        :", sys.version.split()[0])
print("trafficsigns  :", trafficsigns.__file__)
try:
    import torch
    print("PyTorch GPU   :", torch.cuda.is_available())
except ImportError:
    pass
try:
    import tensorflow as tf
    print("TF GPU        :", bool(tf.config.list_physical_devices("GPU")))
except ImportError:
    pass
EOF

log "Terminé. Dans le notebook : import os; os.chdir('$TARGET_DIR')"
