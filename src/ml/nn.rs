#![allow(unused)]

use assert_approx_eq::assert_approx_eq;
use nalgebra::{DMatrix, Matrix, SMatrix, SVector, Vector, Vector2, Matrix2};
use std::f32::consts::E;


fn vec2(x: f32, y: f32) -> Vector2<f32> {
    [x, y].into()
}


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
    
    fn from_weights(weights: SMatrix<f32, NODES, INPUT>, applied_fn: fn(&f32) -> f32) -> Self {
        Self {
            values: SVector::zeros(),
            weights,
            applied_fn,
        }
    }

    fn new(applied_fn: fn(&f32) -> f32) -> Self {
        Self {
            values: SVector::zeros(),
            weights: SMatrix::new_random(),
            applied_fn,
        }
    }

    fn forward(&mut self, input: &SVector<f32, INPUT>) {
        let weight_x_input = self.weights * input;
        self.values = weight_x_input.map(|element| (self.applied_fn)(&element));
    }

    fn values(&self) -> &SVector<f32, NODES> {
        &self.values
    }
}

struct NN {
    layers: (Layer<1, 2>, Layer<2, 2>, Layer<2, 1>),
}

impl NN {
    fn new() -> Self {
        Self {
            layers: (
                Layer::from_weights(vec2(1., -1.),relu),
                Layer::from_weights(Matrix2::new(-1., 1., 1., -1.),relu),
                Layer::from_weights(vec2(-1., 1.).transpose(),relu),
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

fn relu(x: &f32) -> f32 {
    if *x > 0. {*x} else {0.}
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

        layer.forward(&input);
        assert_eq!(
            *layer.values(),
            <SVector<f32, 2>>::from([0.00669285236, 0.268941432])
        );
    }

    #[test]
    fn test_nn_forward() {
        let mut nn = NN::new();
        let out = nn.forward([3.0].into());
        assert_eq!(out, 3.0);
    }
}
