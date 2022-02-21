#![allow(unused)]

use std::f32::consts::E;
use assert_approx_eq::assert_approx_eq;
use nalgebra::{DMatrix, Matrix, SMatrix, SVector, Vector};

struct Layer<const INPUT: usize, const NODES: usize> {
    values: SVector<f32, NODES>,
    weights: SMatrix<f32, NODES, INPUT>,
    applied_fn: fn(&f32) -> f32,
}

impl<const INPUT: usize, const NODES: usize> Layer<INPUT, NODES> {
    fn from_values(
        values: SVector<f32, NODES>,
        weights: SMatrix<f32, NODES, INPUT>,
        applied_fn: fn(&f32) -> f32,
    ) -> Self {
        Self {
            values,
            weights,
            applied_fn,
        }
    }

    fn new(applied_fn: fn(&f32) -> f32) -> Self {
        Self {
            values: SVector::new_random(),
            weights: SMatrix::new_random(),
            applied_fn,
        }
    }

    fn forward(&mut self, input: SVector<f32, INPUT>) {
        let weight_x_input = self.weights * input;
        self.values = weight_x_input.map(|element| (self.applied_fn)(&element));;
    }

    fn values(&self) -> &SVector<f32, NODES> {
        &self.values
    }
}

struct NN {
    input: SVector<f32, 2>,
    hidden: [SVector<f32, 2>; 2],
    output: SVector<f32, 1>,
}

impl NN {
    fn forward(&mut self) {
        //let neural_values = [];
        for layer in self.hidden {}
    }
}

fn sigmoid(x: &f32) -> f32 {
    1.0 / (1.0 + E.powf(*x))
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix2, Vector2};

    use super::*;

    #[test]
    fn test_layer() {
        let mut layer = Layer::<2, 2>::from_values(
            [0.0, 0.0].into(),
            Matrix2::<f32>::new(1., 0., 0., 1.),
            sigmoid,
        );
        let input: SVector<f32, 2> = [5., 1.].into();
    
        layer.forward(input);
        assert_eq!(*layer.values(), <SVector<f32, 2>>::from([0.00669285236, 0.268941432]));
    }
}
