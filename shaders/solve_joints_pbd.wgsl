struct Vec3 {
    x : f32,
    y : f32,
    z : f32,
};

struct Quat {
    x : f32,
    y : f32,
    z : f32,
    w : f32,
};

struct Body {
    pos : Vec3,
    _pad1 : f32,
    vel : Vec3,
    _pad2 : f32,
    orientation : Quat,
    angular_vel : Vec3,
    _pad3 : f32,
};

struct Joint {
    body_a : u32,
    body_b : u32,
    joint_type : u32, // 0: Distance, 1: Hinge, 2: Slider
    rest_length : f32,
    local_anchor_a : Vec3,
    local_anchor_b : Vec3,
    local_axis_a : Vec3,
    local_axis_b : Vec3,
};

struct SolveParams {
    compliance : f32,
    _pad : vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> bodies : array<Body>;
@group(0) @binding(1) var<storage, read> joints : array<Joint>;
@group(0) @binding(2) var<uniform> _params : SolveParams;

fn quat_mul(q1 : Quat, q2 : Quat) -> Quat {
    let w1 = q1.w; let x1 = q1.x; let y1 = q1.y; let z1 = q1.z;
    let w2 = q2.w; let x2 = q2.x; let y2 = q2.y; let z2 = q2.z;
    return Quat(
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    );
}

fn quat_inv(q : Quat) -> Quat {
    let len_sq = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    if (len_sq == 0.0) {
        return Quat(0.0, 0.0, 0.0, 1.0);
    }
    return Quat(-q.x/len_sq, -q.y/len_sq, -q.z/len_sq, q.w/len_sq);
}

fn quat_mul_vec3(q : Quat, v_struct : Vec3) -> Vec3 {
    let v = vec3<f32>(v_struct.x, v_struct.y, v_struct.z);
    let u = vec3<f32>(q.x, q.y, q.z);
    let s = q.w;
    let res = 2.0 * dot(u, v) * u + (s*s - dot(u, u)) * v + 2.0 * s * cross(u, v);
    return Vec3(res.x, res.y, res.z);
}

@compute @workgroup_size(1)
fn main() {
    let nj = arrayLength(&joints);
    for (var i : u32 = 0u; i < nj; i = i + 1u) {
        let jnt = joints[i];
        if (jnt.body_a >= arrayLength(&bodies) || jnt.body_b >= arrayLength(&bodies)) {
            continue;
        }

        var ba = bodies[jnt.body_a];
        var bb = bodies[jnt.body_b];

        // Body positions
        var pa = vec3<f32>(ba.pos.x, ba.pos.y, ba.pos.z);
        var pb = vec3<f32>(bb.pos.x, bb.pos.y, bb.pos.z);

        if (jnt.joint_type == 0u) { // Distance Joint
            let dx = pb - pa;
            let len_sq = dot(dx, dx);
            if (len_sq == 0.0) { continue; }
            let len = sqrt(len_sq);
            let diff = (len - jnt.rest_length) / len * 0.5;
            
            pa = pa + dx * diff;
            pb = pb - dx * diff;

        } else if (jnt.joint_type == 1u) { // Hinge Joint
            // 1. Get world space anchor points and axes
            let anchor_a_world = quat_mul_vec3(ba.orientation, jnt.local_anchor_a);
            let anchor_b_world = quat_mul_vec3(bb.orientation, jnt.local_anchor_b);
            let anchor_a = ba.pos + anchor_a_world;
            let anchor_b = bb.pos + anchor_b_world;
            let diff = anchor_a - anchor_b;
            ba.pos += diff * 0.5;
            bb.pos -= diff * 0.5;

            // Rotational constraint for hinge joint
            let axis_a_world = quat_mul_vec3(ba.orientation, jnt.local_axis_a);
            let axis_b_world = quat_mul_vec3(bb.orientation, jnt.local_axis_b);

            let rot = cross(axis_a_world, axis_b_world);
            let rot_len = length(rot);
            if (rot_len > 0.0001) {
                let rot_axis = rot / rot_len;
                let angle = asin(rot_len);
                let half_angle = angle * 0.5;
                let sin_half = sin(half_angle);
                let q_rot = vec4<f32>(rot_axis * sin_half, cos(half_angle));
                
                let q_rot_inv = quat_inv(q_rot);
                ba.orientation = quat_mul(q_rot, ba.orientation);
                bb.orientation = quat_mul(q_rot_inv, bb.orientation);
            }
            */

        } else if (jnt.joint_type == 2u) { // Slider Joint
            // Placeholder for Slider joint logic
        }

        bodies[jnt.body_a].pos.x = pa.x;
        bodies[jnt.body_a].pos.y = pa.y;
        bodies[jnt.body_a].pos.z = pa.z;
        bodies[jnt.body_a].orientation = ba.orientation; // Write back orientation
        
        bodies[jnt.body_b].pos.x = pb.x;
        bodies[jnt.body_b].pos.y = pb.y;
        bodies[jnt.body_b].pos.z = pb.z;
        bodies[jnt.body_b].orientation = bb.orientation; // Write back orientation
    }
}
