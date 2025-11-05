#!/bin/bash
set -e

# === Config ===
PROJECT_DIR="/workspace/projects/nutrition-table"
VENV_DIR="$PROJECT_DIR/.venv"
REQ_FILE="$PROJECT_DIR/requirements.txt"

# Persistent cache & tmp inside /workspace
export PIP_CACHE_DIR="/workspace/.pipcache"
export TMPDIR="/workspace/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# Detect current Python version (major.minor, e.g. 3.12)
CURRENT_PY=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Function to (re)create venv
create_venv () {
    echo "🔧 Creating venv with Python $CURRENT_PY..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
}

# If venv exists, check if Python version matches
if [ -d "$VENV_DIR" ]; then
    VENV_PY=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "MISMATCH")

    if [ "$VENV_PY" != "$CURRENT_PY" ]; then
        echo "⚠️ Venv Python version mismatch (venv=$VENV_PY, system=$CURRENT_PY)"
        create_venv
    else
        echo "✅ Venv already matches Python $CURRENT_PY"
    fi
else
    create_venv
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "$REQ_FILE" ]; then
    echo "📦 Installing from $REQ_FILE..."
    pip install --no-cache-dir -r "$REQ_FILE"
else
    echo "⚠️ No requirements.txt found at $REQ_FILE"
fi

echo "🎉 Environment ready!"
echo "👉 To activate later, run: source $VENV_DIR/bin/activate"

# === Add project src to PYTHONPATH permanently ===
if ! grep -q "export PYTHONPATH=.*nutrition-table/src" "$VENN_DIR/bin/activate" 2>/dev/null; then
  echo "export PYTHONPATH=$PROJECT_DIR/src:\$PYTHONPATH" >> "$VENN_DIR/bin/activate"
  echo "✅ Added PYTHONPATH to venv activate script."
fi

