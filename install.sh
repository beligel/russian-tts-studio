#!/usr/bin/env bash
# Russian TTS Studio — Linux installer
#
# Installs the project into ~/.local/share/russian-tts-studio/ with:
# - Python virtual environment
# - All dependencies
# - .desktop file for application menu
# - Optional systemd user service
# - Launcher symlink in ~/.local/bin/
#
# Usage:
#   bash install.sh              # Interactive install
#   bash install.sh --prefix /opt/russian-tts  # Custom prefix
#   bash install.sh --no-systemd               # Skip systemd service
#   bash install.sh --uninstall                # Remove everything
#
# Requires: python3 >= 3.10, pip, bash

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME="russian-tts-studio"
APP_DISPLAY_NAME="Russian TTS Studio"
APP_VERSION="0.3.0"

# Defaults
PREFIX="${HOME}/.local"
INSTALL_DIR="${PREFIX}/share/${APP_NAME}"
BIN_DIR="${PREFIX}/bin"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons"
SERVICE_DIR="${HOME}/.config/systemd/user"
CONFIG_DIR="${HOME}/.config/${APP_NAME}"

PYTHON_MIN_VERSION="3.10"
PORT=8129

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()   { echo -e "${YELLOW}[!]${NC} $*"; }
error()  { echo -e "${RED}[✗]${NC} $*" >&2; }
info()   { echo -e "${BLUE}[i]${NC} $*"; }

check_python() {
    if ! command -v python3 &>/dev/null; then
        error "python3 not found. Install Python >= ${PYTHON_MIN_VERSION}"
        exit 1
    fi
    local version
    version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local major minor
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    if (( major < 3 || (major == 3 && minor < 10) )); then
        error "Python >= ${PYTHON_MIN_VERSION} required, found ${version}"
        exit 1
    fi
    log "Python ${version} found"
}

check_pip() {
    if ! python3 -m pip --version &>/dev/null; then
        error "pip not found. Install python3-pip"
        exit 1
    fi
    log "pip found"
}

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

do_install() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Russian TTS Studio — Linux Installer   ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
    echo ""

    check_python
    check_pip

    # Detect project root (where this script lives)
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    info "Project root: ${script_dir}"
    info "Install dir:  ${INSTALL_DIR}"
    info "Port:         ${PORT}"
    echo ""

    # Create directories
    mkdir -p "${INSTALL_DIR}"
    mkdir -p "${BIN_DIR}"
    mkdir -p "${DESKTOP_DIR}"
    mkdir -p "${ICON_DIR}"
    mkdir -p "${CONFIG_DIR}"

    # Copy project files (excluding .git, .venv, __pycache__, output)
    info "Copying project files..."
    rsync -a --progress \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='output' \
        --exclude='.mimocode' \
        --exclude='*.pyc' \
        "${script_dir}/" "${INSTALL_DIR}/"

    # Create virtual environment
    info "Creating virtual environment..."
    python3 -m venv "${INSTALL_DIR}/.venv"
    source "${INSTALL_DIR}/.venv/bin/activate"

    # Install dependencies
    info "Installing dependencies..."
    pip install --upgrade pip --quiet
    pip install -r "${INSTALL_DIR}/requirements.txt" --quiet

    # Optional: install desktop dependencies
    if command -v apt-get &>/dev/null; then
        info "Detected Debian/Ubuntu — checking desktop dependencies..."
        if ! python3 -c "import gi" 2>/dev/null; then
            warn "PyGObject not found. For tray icon, install:"
            warn "  sudo apt install python3-gi gir1.2-ayatanaappindicator3-0.1"
        fi
    fi

    # Create launcher script
    info "Creating launcher..."
    cat > "${BIN_DIR}/${APP_NAME}" << LAUNCHER
#!/usr/bin/env bash
cd "${INSTALL_DIR}"
exec "${INSTALL_DIR}/.venv/bin/python" -m web.start --port ${PORT} "\$@"
LAUNCHER
    chmod +x "${BIN_DIR}/${APP_NAME}"
    log "Launcher: ${BIN_DIR}/${APP_NAME}"

    # Create .desktop file
    info "Creating .desktop entry..."
    local icon_path="${ICON_DIR}/${APP_NAME}.png"
    # Generate simple icon if not present
    if [ ! -f "${icon_path}" ]; then
        python3 -c "
import struct, zlib
w = h = 128
img = bytearray()
for y in range(h):
    img.append(0)
    for x in range(w):
        dx, dy = x - 64, y - 64
        r2 = dx*dx + dy*dy
        if r2 < 50*50:
            img += b'\x29\x7a\xe8\xff'
        elif r2 < 54*54:
            img += b'\x9a\xc0\xf5\xff'
        else:
            img += b'\x00\x00\x00\x00'
def chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n'
png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
png += chunk(b'IDAT', zlib.compress(bytes(img)))
png += chunk(b'IEND', b'')
with open('${icon_path}', 'wb') as f:
    f.write(png)
" 2>/dev/null || warn "Could not generate icon"
    fi

    cat > "${DESKTOP_DIR}/${APP_NAME}.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=${APP_DISPLAY_NAME}
GenericName=TTS Studio
Comment=Production TTS pipeline for Russian with VoxCPM2 voice cloning
Exec=${BIN_DIR}/${APP_NAME}
Icon=${icon_path}
Terminal=false
Categories=Audio;Utility;Development;
StartupNotify=true
Keywords=tts;voice;speech;russian;ai;
DESKTOP
    chmod +x "${DESKTOP_DIR}/${APP_NAME}.desktop"
    log "Desktop entry: ${DESKTOP_DIR}/${APP_NAME}.desktop"

    # Install systemd service (optional)
    if [ "${SKIP_SYSTEMD:-0}" != "1" ]; then
        info "Installing systemd user service..."
        mkdir -p "${SERVICE_DIR}"
        cat > "${SERVICE_DIR}/${APP_NAME}.service" << SERVICE
[Unit]
Description=${APP_DISPLAY_NAME}
After=network.target

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/.venv/bin/python -m web.run --host 127.0.0.1 --port ${PORT} --no-reload
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
SERVICE
        log "Systemd service: ${SERVICE_DIR}/${APP_NAME}.service"
        info "To enable: systemctl --user enable ${APP_NAME}"
        info "To start:  systemctl --user start ${APP_NAME}"
    fi

    # Save install metadata
    cat > "${CONFIG_DIR}/install.json" << META
{
    "version": "${APP_VERSION}",
    "install_dir": "${INSTALL_DIR}",
    "prefix": "${PREFIX}",
    "port": ${PORT},
    "installed_at": "$(date -Iseconds)",
    "python": "$(python3 --version 2>&1)",
    "systemd": $([ "${SKIP_SYSTEMD:-0}" != "1" ] && echo "true" || echo "false")
}
META

    echo ""
    log "Installation complete!"
    echo ""
    echo -e "  ${GREEN}Launch:${NC}   ${BIN_DIR}/${APP_NAME}"
    echo -e "  ${GREEN}Web UI:${NC}   http://localhost:${PORT}"
    echo -e "  ${GREEN}Desktop:${NC}  Search for '${APP_DISPLAY_NAME}' in your apps"
    if [ "${SKIP_SYSTEMD:-0}" != "1" ]; then
        echo -e "  ${GREEN}Service:${NC}  systemctl --user start ${APP_NAME}"
    fi
    echo -e "  ${GREEN}Uninstall:${NC} bash ${INSTALL_DIR}/install.sh --uninstall"
    echo ""
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

do_uninstall() {
    echo ""
    info "Uninstalling ${APP_DISPLAY_NAME}..."

    # Stop and disable systemd service
    if systemctl --user is-active "${APP_NAME}" &>/dev/null; then
        systemctl --user stop "${APP_NAME}"
        log "Stopped systemd service"
    fi
    if systemctl --user is-enabled "${APP_NAME}" &>/dev/null; then
        systemctl --user disable "${APP_NAME}"
        log "Disabled systemd service"
    fi
    rm -f "${SERVICE_DIR}/${APP_NAME}.service"
    rm -f "${SERVICE_DIR}/${APP_NAME}.service.timer"

    # Remove .desktop file
    rm -f "${DESKTOP_DIR}/${APP_NAME}.desktop"
    log "Removed .desktop entry"

    # Remove icon
    rm -f "${ICON_DIR}/${APP_NAME}.png"

    # Remove launcher
    rm -f "${BIN_DIR}/${APP_NAME}"
    log "Removed launcher"

    # Remove install directory
    if [ -d "${INSTALL_DIR}" ]; then
        rm -rf "${INSTALL_DIR}"
        log "Removed ${INSTALL_DIR}"
    fi

    # Remove config
    if [ -d "${CONFIG_DIR}" ]; then
        rm -rf "${CONFIG_DIR}"
        log "Removed config"
    fi

    # Remove desktop database cache
    if command -v update-desktop-database &>/dev/null; then
        update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
    fi

    log "Uninstall complete!"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SKIP_SYSTEMD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            INSTALL_DIR="${PREFIX}/share/${APP_NAME}"
            BIN_DIR="${PREFIX}/bin"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-systemd)
            SKIP_SYSTEMD=1
            shift
            ;;
        --uninstall)
            do_uninstall
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prefix DIR     Install prefix (default: ~/.local)"
            echo "  --port PORT      Web UI port (default: 8129)"
            echo "  --no-systemd     Skip systemd user service"
            echo "  --uninstall      Remove everything"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

do_install
