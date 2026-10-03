#!/bin/bash
# Oracle Cloud Always Free Ampere VM — speech-to-speech voice server (CPU-only)
# Run once on the VM after SSHing in.
set -euo pipefail

echo "=== system packages ==="
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg espeak-ng git iptables-persistent

echo "=== python venv + speech-to-speech (CPU extras) ==="
python3 -m venv ~/s2s
# shellcheck disable=SC1091
source ~/s2s/bin/activate
pip install -U pip wheel
pip install "speech-to-speech[faster-whisper,kokoro]"

echo "=== tool API dir ==="
mkdir -p ~/s2s-tools
cp "$(dirname "$0")/tool_server.py" ~/s2s-tools/tool_server.py

echo "=== open firewall ports (persisted) ==="
sudo iptables -I INPUT -p tcp --dport 8765 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8766 -j ACCEPT
sudo netfilter-persistent save

echo
echo "DONE. Next steps:"
echo "  1. echo 'HF_TOKEN=<your-hf-token>' > ~/.s2s_env && chmod 600 ~/.s2s_env"
echo "  2. sudo cp $(dirname "$0")/*.service /etc/systemd/system/"
echo "  3. sudo systemctl daemon-reload && sudo systemctl enable --now s2s s2s-tools"
echo "  4. systemctl status s2s   # check it's up"
echo "  5. journalctl -u s2s -f   # watch the logs"
