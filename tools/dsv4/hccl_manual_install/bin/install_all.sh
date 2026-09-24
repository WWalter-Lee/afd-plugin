#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for step in \
  00_print_config.sh \
  01_preflight.sh \
  02_prepare_sources.sh \
  03_create_venv.sh \
  04_install_python_deps.sh \
  05_install_stack.sh \
  06_verify_install.sh; do
  printf '[hccl-install] Running %s\n' "${step}"
  bash "${SCRIPT_DIR}/${step}"
done

printf '[hccl-install] Installation completed. Start with bin/07_start.sh\n'
