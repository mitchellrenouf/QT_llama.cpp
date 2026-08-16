#!/usr/bin/env bash
set -e

echo "=== Building QT_llama.cpp Flatpak (Freedesktop SDK 25.08) ==="

# 1. Ensure runtime & sdk are installed
echo "Checking Freedesktop SDK 25.08 and Rust extension..."
flatpak install -y --noninteractive flathub org.freedesktop.Platform//25.08 org.freedesktop.Sdk//25.08 org.freedesktop.Sdk.Extension.rust-stable//25.08 2>/dev/null || true

# 2. Vendor cargo dependencies for offline Flatpak build
echo "Vendoring cargo dependencies..."
cargo vendor

# 3. Build and export bundle
BUILD_DIR=".flatpak-build"
REPO_DIR=".flatpak-repo"
mkdir -p "$BUILD_DIR" "$REPO_DIR"

echo "Building package with flatpak-builder..."
flatpak-builder --force-clean --user --install-deps-from=flathub --repo="$REPO_DIR" "$BUILD_DIR" flatpak/dev.mitchellrenouf.QT_llama.yml

# 4. Create standalone .flatpak single-file bundle
echo "Creating standalone bundle dev.mitchellrenouf.QT_llama.flatpak..."
flatpak build-bundle "$REPO_DIR" dev.mitchellrenouf.QT_llama.flatpak dev.mitchellrenouf.QT_llama 25.08

echo "=== Flatpak Build Complete! ==="
echo "To install locally: flatpak install --user -y dev.mitchellrenouf.QT_llama.flatpak"
echo "To run: flatpak run dev.mitchellrenouf.QT_llama"
