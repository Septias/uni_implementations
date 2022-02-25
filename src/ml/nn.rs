#![allow(unused)]

use assert_approx_eq::assert_approx_eq;
use nalgebra::{DMatrix, Matrix, Matrix2, SMatrix, SVector, Vector, Vector2};
use std::f32::consts::E;

use crate::vec2;

use super::helpers::{linear, relu};

struct Layer<const INPUT: usize, const NODES: usize> {
    values: SVector<f32, NODES>,
    weights: SMatrix<f32, NODES, INPUT>,
    applied_fn: fn(f32) -> f32,
    bias: Option<SVector<f32, NODES>>,
}

impl<const INPUT: usize, const NODES: usize> Layer<INPUT, NODES> {
    fn from_values(
        values: SVector<f32, NODES>,
        weights: SMatrix<f32, NODES, INPUT>,
        applied_fn: fn(f32) -> f32,
        bias: Option<SVector<f32, NODES>>,
    ) -> Self {
        Self {
            values,
            weights,
            applied_fn,
            bias,
        }
    }

    fn from_weights(weights: SMatrix<f32, NODES, INPUT>, applied_fn: fn(f32) -> f32) -> Self {
        Self {
            values: SVector::zeros(),
            weights,
            applied_fn,
            bias: None,
        }
    }

    fn new(applied_fn: fn(f32) -> f32) -> Self {
        Self {
            values: SVector::zeros(),
            weights: SMatrix::new_random(),
            applied_fn,
            bias: Some(SVector::repeat(1.)),
        }
    }

    fn forward(&mut self, input: &SVector<f32, INPUT>) {
        let weight_x_input = self.weights * input;
        self.values = weight_x_input.map(|element| (self.applied_fn)(element));
    }

    fn values(&self) -> &SVector<f32, NODES> {
        &self.values
    }
}

struct Nn1 {
    layers: (Layer<1, 2>, Layer<2, 2>, Layer<2, 1>),
}

impl Nn1 {
    fn new() -> Self {
        Self {
            layers: (
                Layer::from_weights(vec2(1., -1.), relu),
                Layer::from_weights(Matrix2::new(-1., 1., 1., -1.), relu),
                Layer::from_weights(vec2(-1., 1.).transpose(), relu),
            ),
        }
    }

    fn forward(&mut self, input: SVector<f32, 1>) -> f32 {
        //let neural_values = [];
        self.layers.0.forward(&input);
        self.layers.1.forward(self.layers.0.values());
        self.layers.2.forward(self.layers.1.values());
        self.layers.2.values()[0]
    }
}

struct Nn2 {
    layers: (Layer<2, 2>, Layer<2, 2>),
}

impl Nn2 {
    fn new() -> Self {
        let w1 = Matrix2::new(3.21, -2.34, 3.21, -2.34);
        let w2 = Matrix2::new(3.19, -2.68, 4.64, -3.44);
        Self {
            layers: (
                Layer::from_values(SVector::zeros(), w1, relu, None),
                Layer::from_values(SVector::zeros(), w2, relu, None),
            ),
        }
    }

    fn forward(&mut self, input: SVector<f32, 2>) -> &SVector<f32, 2> {
        self.layers.0.forward(&input);
        self.layers.1.forward(self.layers.0.values());
        self.layers.1.values()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::helpers::sigmoid;
    use itertools::Itertools;
    use nalgebra::{Matrix2, Matrix2x4, Matrix4x2, RowVector2, Vector2};

    #[test]
    fn test_layer() {
        let mut layer = Layer::<2, 2>::from_values(
            [0.0, 0.0].into(),
            Matrix2::<f32>::new(1., 0., 0., 1.),
            sigmoid,
            None,
        );
        let input: SVector<f32, 2> = [5., 1.].into();

        layer.forward(&input);
        assert_eq!(
            *layer.values(),
            <SVector<f32, 2>>::from([0.00669285236, 0.268941432])
        );
    }

    #[test]
    fn test_nn_forward() {
        let mut nn = Nn1::new();
        let out = nn.forward([3.0].into());
        assert_eq!(out, 3.0);
    }

    #[test]
    fn test_ex_6() {
        let w1: Matrix2<f32> = Matrix2::new(3.21, -2.34, 3.21, -2.34);
        let w2: Matrix2<f32> = Matrix2::new(3.19, -2.68, 4.64, -3.44);

        let b1: RowVector2<f32> = vec2(-3.21, 2.34).transpose();
        let b2: RowVector2<f32> = vec2(-4.08, 4.42).transpose();

        let input: Matrix4x2<_> = Matrix4x2::from_rows(&[
            [0., 0.].into(),
            [0., 1.].into(),
            [1., 0.].into(),
            [1., 1.].into(),
        ]);
        let b1: Matrix4x2<f32> = Matrix4x2::from_rows(&[b1, b1, b1, b1]);
        let b2: Matrix4x2<f32> = Matrix4x2::from_rows(&[b2, b2, b2, b2]);

        let t = ((input * w1 + b1).map(|elem| elem.max(0.0)) * w2) + b2;
        println!("{:?}", t);

        let mut nn = Nn2::new();
    }
}
