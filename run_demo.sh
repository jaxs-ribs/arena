#!/bin/bash
# Run the CartPole demo with rendering

echo "=== JAXS CartPole Demo ==="
echo
echo "Starting the CartPole gym demo with rendering enabled..."
echo
echo "Controls:"
echo "  Camera: WASD to move, Mouse to look around"
echo "  M: Toggle manual control mode"
echo "  1-6: Select cartpole (when manual control is on)"
echo "  Left/Right arrows: Apply force to selected cartpole"
echo "  Space: Stop applying force"
echo "  R: Reset all cartpoles"
echo "  P: Take screenshot"
echo "  F: Toggle fullscreen"
echo "  Escape: Release mouse capture"
echo
echo "The demo shows 6 CartPoles with different control strategies:"
echo "  1. Oscillating control (sine wave)"
echo "  2. Bang-bang control (switches direction)"
echo "  3. No control (falls over)"
echo "  4. Slow oscillation"
echo "  5. Simple feedback control"
echo "  6. Complex pattern"
echo
echo "CartPoles will automatically reset when they fall over or go out of bounds."
echo

cargo run --features render