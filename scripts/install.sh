#!/usr/bin/env bash
set -euo pipefail
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
python -m pip install --upgrade 'pip<26'
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 --only-binary=:all: -f https://data.pyg.org/whl/torch-2.1.0+cu121.html -c constraints.txt
python -m pip install -e . -c constraints.txt
python -m pip check
