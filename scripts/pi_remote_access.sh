#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---status}"
if [[ "$MODE" == "--help" ]]; then
  echo 'Usage: bash scripts/pi_remote_access.sh [--status|--install|--connect|--serve|--help]'
  echo 'Default is read-only. Other modes explicitly change system configuration.'
  exit 0
fi
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="$ROOT_DIR/venv/bin/python3"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
if [[ "$(id -u)" -eq 0 ]]; then SUDO=(); else SUDO=(sudo); fi

usage() {
  echo 'Usage: bash scripts/pi_remote_access.sh [--status|--install|--connect|--serve|--help]'
  echo 'Default is read-only. Each other mode changes system configuration explicitly.'
}

require_tailscale() {
  if ! command -v tailscale >/dev/null 2>&1; then
    echo 'Tailscale is not installed. Run --install after operator approval.' >&2
    exit 1
  fi
}

case "$MODE" in
  --help) usage ;;
  --install)
    if ! command -v tailscale >/dev/null 2>&1; then
      echo 'Installing Tailscale from its official Linux installer.'
      TMP_DIR="$(mktemp -d)"
      trap 'rm -rf "$TMP_DIR"' EXIT
      curl --fail --silent --show-error --location https://tailscale.com/install.sh -o "$TMP_DIR/install.sh"
      "${SUDO[@]}" sh "$TMP_DIR/install.sh"
    fi
    "${SUDO[@]}" systemctl enable --now tailscaled
    echo 'Installed. Run --connect to link this Pi to your own account.'
    ;;
  --connect)
    require_tailscale
    echo 'Link this Pi using the printed login URL; use the same account as Windows/iPhone.'
    # Timeout bounds the command, but the printed login URL can still be used.
    "${SUDO[@]}" tailscale up --hostname=redbot --timeout=30s
    ;;
  --serve)
    require_tailscale
    tailscale status --json | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("BackendState")=="Running" else "Connect Tailscale first")'
    curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8080/api/health >/dev/null
    echo 'Configuring private tailnet HTTPS port 443 to proxy the existing app on localhost:8080.'
    echo 'HTTPS enablement may require confirmation in your Tailscale account.'
    echo 'This updates the default port-443 Serve route; review existing routes first.'
    "${SUDO[@]}" timeout 30 tailscale serve --bg http://127.0.0.1:8080
    tailscale serve status
    ;;
  --status)
    require_tailscale
    tailscale status --json | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); s=d.get("Self") or {}; print("Tailscale:",d.get("BackendState")); print("Pi name:",s.get("DNSName")); print("Pi IP:",", ".join(s.get("TailscaleIPs",[]))); print("Key expiry:",s.get("KeyExpiry") or "not reported; verify in admin console"); print("Health:",d.get("Health") or [])'
    tailscale serve status
    ;;
  *) usage >&2; exit 2 ;;
esac
