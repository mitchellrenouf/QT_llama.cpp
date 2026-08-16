#!/bin/sh
# qt-llama-wrapper.sh: Multi-GPU compute & runtime initialization wrapper for Flatpak sandbox
set -e

# 1. Discover and configure Vulkan ICD and Layer paths (Mesa, NVIDIA, Intel, AMD)
VULKAN_ICD_DIRS=""
for dir in \
    /usr/share/vulkan/icd.d \
    /usr/lib/x86_64-linux-gnu/GL/default/vulkan/icd.d \
    /usr/lib/x86_64-linux-gnu/GL/nvidia-*/vulkan/icd.d \
    /usr/lib/GL/nvidia-*/vulkan/icd.d \
    /etc/vulkan/icd.d \
    /app/share/vulkan/icd.d; do
    if [ -d "$dir" ]; then
        if [ -z "$VULKAN_ICD_DIRS" ]; then
            VULKAN_ICD_DIRS="$dir"
        else
            VULKAN_ICD_DIRS="$VULKAN_ICD_DIRS:$dir"
        fi
    fi
done

if [ -n "$VULKAN_ICD_DIRS" ]; then
    export VK_DRIVER_FILES="$VULKAN_ICD_DIRS"
    export VK_ICD_FILENAMES="$VULKAN_ICD_DIRS"
fi

# 2. Configure Dynamic Linker search paths for GPU driver extensions, CUDA & ROCm libraries
EXTRA_LIB_PATHS=""
for libdir in \
    /app/lib \
    /app/cuda/lib64 \
    /app/cuda/lib \
    /app/rocm/lib \
    /app/rocm/lib64 \
    /usr/lib/x86_64-linux-gnu/GL/default/lib \
    /usr/lib/x86_64-linux-gnu/GL/nvidia-*/lib \
    /usr/lib/GL/default/lib \
    /usr/lib/GL/nvidia-*/lib \
    /usr/lib/x86_64-linux-gnu/dri \
    /usr/lib/dri; do
    if [ -d "$libdir" ]; then
        if [ -z "$EXTRA_LIB_PATHS" ]; then
            EXTRA_LIB_PATHS="$libdir"
        else
            EXTRA_LIB_PATHS="$EXTRA_LIB_PATHS:$libdir"
        fi
    fi
done

if [ -n "$EXTRA_LIB_PATHS" ]; then
    if [ -n "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$EXTRA_LIB_PATHS:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$EXTRA_LIB_PATHS"
    fi
fi

# 3. Add CUDA & ROCm bin to PATH if installed
if [ -d "/app/cuda/bin" ]; then
    export PATH="/app/cuda/bin:$PATH"
fi
if [ -d "/app/rocm/bin" ]; then
    export PATH="/app/rocm/bin:$PATH"
fi

# 4. Configure Qt6 QML import paths
export QML_IMPORT_PATH="/app/share/qt_llama/qml:/app/share/gemma/qml:/usr/lib/qt6/qml:/usr/lib/x86_64-linux-gnu/qt6/qml:${QML_IMPORT_PATH}"
export QML2_IMPORT_PATH="/app/share/qt_llama/qml:/app/share/gemma/qml:/usr/lib/qt6/qml:/usr/lib/x86_64-linux-gnu/qt6/qml:${QML2_IMPORT_PATH}"

# 5. Optional Debug diagnostics when QT_LLAMA_DEBUG=1
if [ "${QT_LLAMA_DEBUG:-0}" = "1" ]; then
    echo "=== QT_llama.cpp GPU Environment ==="
    echo "VK_DRIVER_FILES:  ${VK_DRIVER_FILES:-<default>}"
    echo "LD_LIBRARY_PATH:  $LD_LIBRARY_PATH"
    echo "PATH:             $PATH"
    echo "QML_IMPORT_PATH:  $QML_IMPORT_PATH"
    echo "Available DRI devices:"
    ls -l /dev/dri 2>/dev/null || echo "  No /dev/dri devices found"
    echo "Available NVIDIA devices:"
    ls -l /dev/nvidia* 2>/dev/null || echo "  No /dev/nvidia devices found"
    echo "====================================="
fi

# 6. Exec main application
exec /app/bin/qt_llama "$@"
