#!/bin/bash
set -euo pipefail

# Source emsdk if not already in PATH
if ! command -v emcc &> /dev/null; then
    source "${EMSDK:-$HOME/emsdk}/emsdk_env.sh" 2>/dev/null
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$ROOT_DIR/target/wasm32-unknown-emscripten/release"

echo "Building Rust staticlib..."
cargo build --target wasm32-unknown-emscripten --release -p highs-wasm

HIGHS_LIB="$TARGET_DIR/build/lio-highs-*/out/lib/libhighs.a"
WASM_LIB="$TARGET_DIR/libhighs_wasm.a"

echo "Linking with emcc..."
mkdir -p "$SCRIPT_DIR/dist"
emcc \
    $WASM_LIB \
    $HIGHS_LIB \
    -fwasm-exceptions \
    -O3 \
    -o "$SCRIPT_DIR/dist/highs_wasm.js" \
    -sMODULARIZE=1 \
    -sEXPORT_NAME=HiGHSWasm \
    -sALLOW_MEMORY_GROWTH=1 \
    -sEXPORTED_FUNCTIONS='["_highs_wasm_solve","_highs_wasm_get_obj_value","_highs_wasm_alloc","_highs_wasm_free","_malloc","_free"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","FS","UTF8ToString","stringToUTF8","lengthBytesUTF8"]' \
    -sFORCE_FILESYSTEM=1 \
    --js-library "$SCRIPT_DIR/src/highs_log.js" \
    --no-entry

cp "$SCRIPT_DIR/dist/highs_wasm.js" "$SCRIPT_DIR/demo/"
cp "$SCRIPT_DIR/dist/highs_wasm.wasm" "$SCRIPT_DIR/demo/"

echo "Built: $SCRIPT_DIR/dist/highs_wasm.js"
echo "Built: $SCRIPT_DIR/dist/highs_wasm.wasm"
echo ""
echo "To serve the demo: python3 -m http.server 8080 -d $SCRIPT_DIR/demo"
