use std::f32::consts::E;

pub fn sigmoid(x: f32) -> f32 {
    return 1. / (1. + E.powf(-x));
}
