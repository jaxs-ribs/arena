fn main() {
    // Test setup from test_momentum_conservation
    // Initial: sphere 0 mass=?, vel=2.0; sphere 1 mass=?, vel=0.0
    // Expected momentum with unit masses: 1.0 * 2.0 + 1.0 * 0.0 = 2.0
    // But test shows 8377.581
    
    // This suggests masses are not 1.0
    // With density 1000 (water) and radius 1.0:
    // Volume = 4/3 * pi * r^3 = 4.189
    // Mass = 4.189 * 1000 = 4189
    // Momentum = 4189 * 2.0 = 8378 ✓
    
    println\!("Sphere volume: {}", (4.0/3.0) * std::f32::consts::PI * 1.0_f32.powi(3));
    println\!("Expected mass with density 1000: {}", (4.0/3.0) * std::f32::consts::PI * 1.0_f32.powi(3) * 1000.0);
}
