use nalgebra::SVector;

pub fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

pub fn squared_error(x: f32, y: f32) -> f32 {
    (x - y).powf(2.)
}

pub fn rmse<const SIZE: usize>(x: &SVector<f32, SIZE>, y: &SVector<f32, SIZE>) -> f32 {
    ((x - y).map(|x| x.powf(2.)).sum() / SIZE as f32).sqrt()
}

pub fn relu(x: f32) -> f32 {
    if x > 0. {
        x
    } else {
        0.
    }
}

pub fn linear(x: f32) -> f32 {
    x
}

pub fn approx_equal(a: f32, b: f32, dp: u8) -> bool {
    let p = 10f32.powi(-(dp as i32));
    (a - b).abs() < p
}


#[derive(Debug, Clone)]
pub struct Split {
    pub variable: usize,
    pub value: f32,
}

impl Split {
    pub fn new(variables: usize, value: f32) -> Self {
        Self {
            variable: variables,
            value,
        }
    }
}