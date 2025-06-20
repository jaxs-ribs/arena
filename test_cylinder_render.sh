#!/bin/bash
# Test cylinder rendering

echo "=== Testing Cylinder Rendering ==="
echo
echo "Running headless to check cylinder counts..."
cargo run --quiet 2>&1 | grep -E "(cylinder|Cylinder)" | head -20

echo
echo "To visually verify cylinders are rendering:"
echo "1. Run: cargo run --features render"
echo "2. Use WASD + mouse to move camera around"
echo "3. Look for vertical poles attached to carts"
echo "4. Press P to take a screenshot"
echo
echo "Expected: 6 cylinders (poles) attached to 6 boxes (carts)"