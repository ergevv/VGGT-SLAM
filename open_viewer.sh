#!/usr/bin/env bash
# 在独立的 Chrome 配置中使用 CPU 渲染地图，绕过本机 WebGL 硬件初始化失败。
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIEWER_PORT="${1:-8080}"
if [[ ! "$VIEWER_PORT" =~ ^[0-9]+$ ]] || (( VIEWER_PORT < 1 || VIEWER_PORT > 65535 )); then
    echo "用法：./open_viewer.sh [端口号，默认 8080]" >&2
    exit 1
fi
if ! BROWSER_BIN="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"; then
    echo "未找到 Chrome 或 Chromium，请先安装浏览器。" >&2
    exit 1
fi

# 使用专用配置目录，确保参数生效；普通 Chrome 已运行时也不会复用其渲染配置。
mkdir -p "$PROJECT_DIR/.cache/viser-chrome"
echo "使用软件 WebGL 打开当前地图：http://127.0.0.1:$VIEWER_PORT"
# SwiftShader 仅用于本机可信的 Viser 页面，保留浏览器沙箱。
exec "$BROWSER_BIN" \
    --user-data-dir="$PROJECT_DIR/.cache/viser-chrome" \
    --no-first-run \
    --no-default-browser-check \
    --use-gl=angle \
    --use-angle=swiftshader \
    --enable-unsafe-swiftshader \
    --no-proxy-server \
    --new-window "http://127.0.0.1:$VIEWER_PORT/"
