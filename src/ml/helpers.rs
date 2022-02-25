use std::f32::consts::E;

use nalgebra::SVector;

pub fn sigmoid(x: f32) -> f32 {
    1. / (1. + E.powf(-x))
}

pub fn squared_error(x: f32, y: f32) -> f32 {
    (x - y).powf(2.)
}


pub fn rmse<const SIZE: usize>(x: &SVector<f32, SIZE>, y: &SVector<f32, SIZE>) -> f32 {
    ((x-y).map(|x| x.powf(2.)).sum() / SIZE as f32).sqrt()
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