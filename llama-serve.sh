#!/usr/bin/env bash
# Serveur llama.cpp local pour Seldon : http://127.0.0.1:9931/v1 (API compatible OpenAI).
# Écoute seulement la machine elle-même ; --cors-origins localhost empêche une page web
# ouverte dans le navigateur d'utiliser le modèle. Options en plus : ./llama-serve.sh --port 9932
set -euo pipefail
exec llama serve --host 127.0.0.1 --cors-origins localhost \
    --temp 0.1 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.0 --repeat-penalty 1.0 \
    -c 65000 "$@"
