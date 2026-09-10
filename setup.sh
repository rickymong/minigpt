#!/usr/bin/env bash
# setup.sh — one-time setup. Creates a local Python virtual environment and
# installs PyTorch into it. Run this once from the project folder:
#
#     bash setup.sh
#
# Then activate the environment whenever you work on the project:
#
#     source .venv/bin/activate
#
set -e

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv

echo "==> Upgrading pip"
./.venv/bin/python -m pip install --upgrade pip

echo "==> Installing PyTorch + NumPy"
# On Intel Macs the last supported PyTorch is 2.2.2; pip picks the right wheel
# for your platform automatically. On Apple Silicon / Linux you get the latest.
./.venv/bin/pip install "torch" "numpy" || ./.venv/bin/pip install "torch==2.2.2" "numpy"

echo ""
echo "==> Done. Next steps:"
echo "    source .venv/bin/activate      # turn the environment on"
echo "    python train.py                # train the model (a few minutes on CPU)"
echo "    python sample.py --prompt 'ROMEO:'   # generate text"
