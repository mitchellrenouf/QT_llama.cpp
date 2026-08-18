#!/usr/bin/env bash
set -e

echo "=== Building MRML Flatpak (KDE 6.11 / Qt6 / CUDA) ==="

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

# 4. Optional: Build & install GPU SDK extensions if requested
if [ "${BUILD_GPU_EXTENSIONS:-0}" = "1" ]; then
    echo "Building GPU SDK Extensions (CUDA, ROCm, oneAPI)..."
    flatpak-builder --force-clean --user --install --repo="$REPO_DIR" .build-cuda flatpak/org.freedesktop.Sdk.Extension.cuda.yml || true
    flatpak-builder --force-clean --user --install --repo="$REPO_DIR" .build-rocm flatpak/org.freedesktop.Sdk.Extension.rocm.yml || true
    flatpak-builder --force-clean --user --install --repo="$REPO_DIR" .build-oneapi flatpak/org.freedesktop.Sdk.Extension.oneapi.yml || true
    rm -rf .build-cuda .build-rocm .build-oneapi tmp 2>/dev/null || true
fi

echo "Building package with flatpak-builder..."
flatpak-builder --force-clean --user --install-deps-from=flathub --repo="$REPO_DIR" "$BUILD_DIR" flatpak/dev.mitchellrenouf.mrml.yml

# 5. Create standalone .flatpak single-file bundle
echo "Creating standalone bundle dev.mitchellrenouf.mrml.flatpak..."
flatpak build-bundle "$REPO_DIR" dev.mitchellrenouf.mrml.flatpak dev.mitchellrenouf.mrml 6.11

echo "=== Flatpak Build Complete! ==="
echo "To install locally: flatpak install --user -y dev.mitchellrenouf.mrml.flatpak"
echo "To run: flatpak run dev.mitchellrenouf.mrml"
