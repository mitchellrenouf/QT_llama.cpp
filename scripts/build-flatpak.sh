#!/usr/bin/env bash
set -e

echo "=== Building Gemma Agent Flatpak (Freedesktop SDK 26.08) ==="

# 1. Ensure runtime & sdk are installed
echo "Checking Freedesktop SDK 26.08 and Rust extension..."
flatpak install -y --noninteractive flathub-beta org.freedesktop.Platform//26.08 org.freedesktop.Sdk//26.08 org.freedesktop.Sdk.Extension.rust-stable//26.08 2>/dev/null || \
flatpak install -y --noninteractive flathub org.freedesktop.Platform//26.08 org.freedesktop.Sdk//26.08 org.freedesktop.Sdk.Extension.rust-stable//26.08 2>/dev/null || true

# 2. Build and export bundle
BUILD_DIR=".flatpak-build"
REPO_DIR=".flatpak-repo"
mkdir -p "$BUILD_DIR" "$REPO_DIR"

echo "Building package with flatpak-builder..."
flatpak-builder --force-clean --user --install-deps-from=flathub --repo="$REPO_DIR" "$BUILD_DIR" flatpak/org.gemma.GemmaAgent.yml

# 3. Create standalone .flatpak single-file bundle
echo "Creating standalone bundle org.gemma.GemmaAgent.flatpak..."
flatpak build-bundle "$REPO_DIR" org.gemma.GemmaAgent.flatpak org.gemma.GemmaAgent 26.08

echo "=== Flatpak Build Complete! ==="
echo "To install locally: flatpak install --user -y org.gemma.GemmaAgent.flatpak"
echo "To run: flatpak run org.gemma.GemmaAgent"
