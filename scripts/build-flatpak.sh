#!/usr/bin/env bash
set -e

echo "=== Building QT_llama.cpp Flatpak (Freedesktop SDK 24.08) ==="

# 1. Ensure runtime & sdk are installed
echo "Checking Freedesktop SDK 24.08 and Rust extension..."
flatpak install -y --noninteractive flathub org.freedesktop.Platform//24.08 org.freedesktop.Sdk//24.08 org.freedesktop.Sdk.Extension.rust-stable//24.08 2>/dev/null || true

# 2. Build and export bundle
BUILD_DIR=".flatpak-build"
REPO_DIR=".flatpak-repo"
mkdir -p "$BUILD_DIR" "$REPO_DIR"

echo "Building package with flatpak-builder..."
flatpak-builder --force-clean --user --install-deps-from=flathub --repo="$REPO_DIR" "$BUILD_DIR" flatpak/org.llamacpp.QT_llama.yml

# 3. Create standalone .flatpak single-file bundle
echo "Creating standalone bundle org.llamacpp.QT_llama.flatpak..."
flatpak build-bundle "$REPO_DIR" org.llamacpp.QT_llama.flatpak org.llamacpp.QT_llama 24.08

echo "=== Flatpak Build Complete! ==="
echo "To install locally: flatpak install --user -y org.llamacpp.QT_llama.flatpak"
echo "To run: flatpak run org.llamacpp.QT_llama"
