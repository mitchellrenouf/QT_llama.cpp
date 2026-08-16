#!/usr/bin/env bash
set -e

echo "=== Building QT_llama.cpp Flatpak (KDE 6.11 / Qt6) ==="

# 1. Ensure runtime & sdk are installed
echo "Checking KDE 6.11 Platform/Sdk and Rust extension..."
flatpak install -y --noninteractive flathub org.kde.Platform//6.11 org.kde.Sdk//6.11 org.freedesktop.Sdk.Extension.rust-stable//25.08 2>/dev/null || true

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
flatpak build-bundle "$REPO_DIR" dev.mitchellrenouf.QT_llama.flatpak dev.mitchellrenouf.QT_llama 6.11

echo "=== Flatpak Build Complete! ==="
echo "To install locally: flatpak install --user -y dev.mitchellrenouf.QT_llama.flatpak"
echo "To run: flatpak run dev.mitchellrenouf.QT_llama"
