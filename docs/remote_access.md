# Private Pi access and iPhone home-screen app

## Current setup

- Private HTTPS: https://redbot.tail644ec2.ts.net
- Pi Tailscale IP: 100.69.240.76
- LAN fallback: 192.168.68.106 or redbot.local
- HTTPS proxies localhost:8080 with Tailscale Serve in background mode.
- No Funnel, public router port-forwarding, exit node or Tailscale SSH is enabled
  by these scripts. Normal OpenSSH still uses the existing dedicated key.
- Tailscale runs as a boot-enabled system service; Serve --bg persists its config.

These identifiers describe the current installation, not credentials. A rebuilt
Pi or different account may get different addresses. Discover them with --status.
Do not put private keys, auth keys, passwords or account state in Git.

## Rebuild (operator-approved system changes)

First restore the repo, Python environment and normal Red Bot systemd services.
Then run each step deliberately, not as part of every pi_update.sh:

```bash
bash scripts/pi_remote_access.sh --install
bash scripts/pi_remote_access.sh --connect
```

Use the printed device-login link with the same account used on Windows and
iPhone. No password or auth key needs to be shared with the coding assistant.
Installation downloads the current official Tailscale installer, not a pinned
package version. It enables tailscaled, not the trading services.

After login, inspect existing Serve routes before changing the port-443 route:

```bash
bash scripts/pi_remote_access.sh --status
bash scripts/pi_remote_access.sh --serve
```

Approve HTTPS enablement in the Tailscale console if prompted. HTTPS certificate
issuance publishes the machine's certificate hostname in transparency logs;
the app itself remains private to authorized tailnet devices. The installer
does not change tailnet access policies. Review which devices/users have access.
Timeout after a login/HTTPS prompt is expected until its approval is completed;
rerun that same step after approval. Never use Funnel for this app.

## Verify before calling it ready

Check the HTTPS app, /api/health, /api/mobile and /redbot-icon.png using the real
hostname returned by --status. Validate HTTPS normally, never disable certificate
verification. HTTP health alone does not prove trading or data freshness.
Test normal SSH over the Tailscale IP with the pinned host key and existing
dedicated key. On rebuild verify the Pi host fingerprint directly before changing
known_hosts; never automatically trust a changed host identity.

The assistant's local pi_connect.ps1 is workspace tooling, not a private key
stored in this repo. It prefers the current Tailscale IP, then bounded LAN
fallback, and enforces strict host-key checking. Update its target after rebuild.

## iPhone

Keep Tailscale connected. Open the private HTTPS URL in Safari, remove the old
LAN shortcut, then Share -> Add to Home Screen. All page assets use relative
URLs so the icon and API stay on the same origin. Test with Wi-Fi disabled and
mobile data enabled. The icon is a shortcut, not an independent native service.
Pi power/internet, running app, valid Tailscale login and phone connection are
still required. Old LAN shortcuts cannot be rewritten remotely.

## Expiry and recovery

At setup on October 10, 2026, the Pi device key was due to expire April 8, 2027.
Check --status/admin console for the actual current date. Renew beforehand, or
separately approve disabling key expiry for this unattended server in the device
admin console. Disabling expiry trades reduced login interruptions for a longer
credential lifetime; revoke the device promptly if lost or compromised.
No expiry setting is changed by this script. Windows/iPhone login can expire too.

Git pulls do not overwrite installed system packages or Tailscale state. A clean
OS/SD-card installation does: rerun the above procedure and update the shortcut
if the hostname changes. Keep secret-containing Tailscale state outside Git.

To disable only this private HTTPS proxy (explicit operator approval):

```bash
sudo tailscale serve --https=443 off
```

Do not stop/restart trading merely to fix remote access. Diagnose local app,
tailnet connectivity and local Codex runtime separately.
