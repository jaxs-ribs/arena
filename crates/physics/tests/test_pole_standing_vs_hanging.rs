use physics::{PhysicsSim, CartPole, CartPoleConfig};
use physics::types::Vec3;

#[test]
fn test_pole_is_standing_not_hanging() {
    let mut sim = PhysicsSim::new();
    
    let config = CartPoleConfig {
        cart_size: Vec3::new(0.4, 0.2, 0.2),
        pole_length: 2.0,
        pole_radius: 0.05,
        initial_angle: 0.1, // Small tilt
        ..Default::default()
    };
    
    let mut cartpole = CartPole::new(&mut sim, Vec3::ZERO, config.clone());
    
    // Get key positions
    let cart = &sim.boxes[cartpole.cart_idx];
    let pole = &sim.cylinders[cartpole.pole_idx];
    
    let cart_top_y = cart.pos.y + cart.half_extents.y;
    let pole_center_y = pole.pos.y;
    let pole_top_y = pole.pos.y + pole.half_height;
    let pole_bottom_y = pole.pos.y - pole.half_height;
    
    println!("=== STANDING vs HANGING TEST ===");
    println!("Cart top Y: {}", cart_top_y);
    println!("Pole bottom Y: {}", pole_bottom_y);
    println!("Pole center Y: {}", pole_center_y);
    println!("Pole top Y: {}", pole_top_y);
    
    // CRITICAL TESTS:
    println!("\n=== KEY CHECKS ===");
    
    // 1. Is pole bottom at cart top?
    let bottom_at_joint = (pole_bottom_y - cart_top_y).abs() < 0.01;
    println!("1. Pole bottom at cart top (joint)? {}", bottom_at_joint);
    
    // 2. Is pole center ABOVE cart top?
    let center_above_cart = pole_center_y > cart_top_y;
    println!("2. Pole center ABOVE cart top? {}", center_above_cart);
    
    // 3. Is pole top the highest point?
    let top_is_highest = pole_top_y > pole_center_y && pole_top_y > cart_top_y;
    println!("3. Pole top is highest point? {}", top_is_highest);
    
    // 4. Height relationships
    println!("\n=== HEIGHT ORDER (bottom to top) ===");
    let mut heights = vec![
        ("Cart bottom", cart.pos.y - cart.half_extents.y),
        ("Cart center", cart.pos.y),
        ("Cart top (joint)", cart_top_y),
        ("Pole bottom", pole_bottom_y),
        ("Pole center", pole_center_y),
        ("Pole top", pole_top_y),
    ];
    heights.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    for (name, y) in heights {
        println!("  {}: {}", name, y);
    }
    
    // DEFINITIVE TEST
    println!("\n=== DEFINITIVE ANSWER ===");
    if center_above_cart && bottom_at_joint {
        println!("✓ The pole is STANDING UP on the cart");
    } else {
        println!("✗ The pole is HANGING DOWN from somewhere");
    }
    
    // Visual ASCII representation
    println!("\n=== VISUAL REPRESENTATION ===");
    println!("Height 2.5:                    {}", if pole_top_y > 2.5 { "?" } else { " " });
    println!("Height 2.0:          {}         ", if pole_top_y > 2.0 { "|" } else { " " });
    println!("Height 1.5:          {}         ", if pole_center_y > 1.5 { "|" } else { " " });
    println!("Height 1.0:          {}         ", if pole_center_y > 1.0 { "|" } else { " " });
    println!("Height 0.5:          {}         ", if pole_bottom_y > 0.5 { "|" } else { " " });
    println!("Height 0.0:     [==={}===]      ", if pole_bottom_y > 0.0 { "|" } else { " " });
    println!("               Cart (y={:.2})", cart.pos.y);
    
    // Run physics to see motion
    println!("\n=== AFTER 20 PHYSICS STEPS ===");
    for _ in 0..20 {
        sim.step_cpu();
    }
    
    let pole_after = &sim.cylinders[cartpole.pole_idx];
    let angle_after = cartpole.get_pole_angle(&sim);
    
    println!("Pole angle: {:.3} rad ({:.1}°)", angle_after, angle_after.to_degrees());
    println!("Pole still above cart? {}", pole_after.pos.y > cart_top_y);
    
    // Final assertions
    assert!(bottom_at_joint, "Pole bottom should be at joint");
    assert!(center_above_cart, "Pole center should be above cart");
    assert!(top_is_highest, "Pole top should be highest point");
}