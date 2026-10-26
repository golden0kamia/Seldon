#!/usr/bin/env bash
# Serveur MCP Nextcloud pour Seldon (nextcloud-mcp-server) : http://127.0.0.1:8000/mcp.
# Lit NEXTCLOUD_HOST, NEXTCLOUD_USERNAME et NEXTCLOUD_PASSWORD (mot de passe d'application)
# dans le .env à côté du script. Écoute seulement la machine elle-même ; seules les
# applications utiles sont activées. Options en plus : ./nextcloud-mcp.sh -e webdav
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then source .env; fi   # seules les 3 variables ci-dessous sont transmises
: "${NEXTCLOUD_HOST:?NEXTCLOUD_HOST manquant (fichier .env)}"
: "${NEXTCLOUD_USERNAME:?NEXTCLOUD_USERNAME manquant (fichier .env)}"
: "${NEXTCLOUD_PASSWORD:?NEXTCLOUD_PASSWORD manquant (mot de passe d application, fichier .env)}"
export NEXTCLOUD_HOST NEXTCLOUD_USERNAME NEXTCLOUD_PASSWORD
exec pipx run nextcloud-mcp-server run --host 127.0.0.1 --port 8000 \
    -e notes -e calendar -e contacts "$@"
