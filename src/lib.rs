use nalgebra::{Vector2, Vector3};

pub mod ml;
pub mod optimierung;

/// Type alias for the result of a drawing function.
pub type DrawResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn vec2(x: f32, y: f32) -> Vector2<f32> {
    [x, y].into()
}

pub fn vec3(x: f32, y: f32, z: f32) -> Vector3<f32> {
    [x, y, z].into()
}
