#!/usr/bin/env bash
set -e

APP_ID="dev.mitchellrenouf.mrml"

echo "=== Checking updates for $APP_ID ==="

# 1. Update AppStream metadata
if command -v flatpak &>/dev/null; then
    echo "Updating Flatpak AppStream catalog..."
    flatpak update --appstream -y || true

    # 2. Check if the app is installed and update it
    if flatpak list | grep -q "$APP_ID"; then
        echo "Updating $APP_ID..."
        flatpak update -y "$APP_ID"
        echo "✔ $APP_ID is up to date!"
    else
        echo "ℹ $APP_ID is not currently installed via Flatpak."
        echo "Run ./scripts/build-flatpak.sh to build and install it."
    fi
else
    echo "✖ Flatpak is not installed on this system."
    exit 1
fi
