#!/bin/bash
# Quick test of CartPole physics

cd "$(dirname "$0")"

# Run the main app in headless mode for a short time
echo "=== Testing CartPole Physics (Headless) ==="
cargo run --quiet 2>&1 | head -30

echo -e "\n=== Running Physics Tests ==="
cargo test -p physics cartpole --quiet

echo -e "\nTo run with graphics, use: cargo run --features render"
echo "Controls when running with graphics:"
echo "  - M: Toggle manual control mode"
echo "  - 1-6: Select cartpole (when manual control is on)"
echo "  - Left/Right arrows: Apply force to selected cartpole"
echo "  - Space: Stop applying force"
echo "  - R: Reset all cartpoles"
echo "  - P: Take screenshot"
echo "  - F: Toggle fullscreen"
echo "  - Escape: Release mouse"