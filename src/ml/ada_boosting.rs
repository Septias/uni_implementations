#![allow(unused)]
use std::f32::consts::E;

use itertools::Itertools;
use nalgebra::{DMatrix, DVector, SMatrix, Vector1, Vector2, Vector3, SVector, ComplexField};

use super::helpers::approx_equal;

struct Stump {
    error: f32,
    correctness: f32,
}

impl Stump {
    fn update_error(&mut self) {}

    fn update_correctness(&mut self) {}
}

#[derive(Clone, PartialEq, Debug)]
pub struct TrainingExample {
    pub position: Vector3<f32>,
    pub classification: i32,
}

impl TrainingExample {
    pub fn new(position: Vector3<f32>, classification: i32) -> Self {
        Self {
            position,
            classification,
        }
    }
}

pub struct Tree<const TRAINING_POINTS: usize> {
    depth: usize,
    stumps: Vec<Stump>,
    splits: Vec<Vec<f32>>,
    training_points: [TrainingExample; TRAINING_POINTS],
}

impl<const TRAINING_POINTS: usize> Tree<TRAINING_POINTS> {
    pub fn new(depth: usize, training_points: [TrainingExample; TRAINING_POINTS]) -> Self {

        let iterator = training_points.iter().map(|tp | tp.position).collect_vec();
        let matrix = SMatrix::<f32, 3, TRAINING_POINTS>::from_columns(&iterator).transpose();
            
        let splits = matrix
            .column_iter()
            .map(|column| {
                let mut items = column
                    .iter()
                    .dedup_by(|a, b| approx_equal(**a, **b, 10))
                    .collect_vec();

                items.sort_by(|a, b| a.partial_cmp(b).unwrap());

                items
                    .iter()
                    .tuple_windows()
                    .filter(|(a, b)| !approx_equal(***a, ***b, 10))
                    .map(|(x1, x2)| (**x1 + **x2) / 2.)
                    .collect_vec()
            })
            .collect_vec();

        Self {
            stumps: vec![],
            depth,
            splits,
            training_points,
        }
    }

    pub fn train(&mut self) {
        for i in 0..self.depth {
            let split_value = 
            self.stumps[i].update_error();
        }
    }

    fn train_new_stump(&self) {}

    pub fn predict() {}
}

fn loss_fn<const T: usize> (y: SVector<f32, T>, y_hat: SVector<f32, T>, w: SVector<f32, T> ) {
    y.component_mul(&y_hat).map(|elem| E.powf(elem)) ;
}

#[cfg(test)]
mod tests {
    use nalgebra::{SVector, Vector4};

    use crate::vec3;

    use super::*;

    #[test]
    fn test_tree_constructor() {
        let training_points = [
            TrainingExample::new(vec3(-1., 1., 1.), -1),
            TrainingExample::new(vec3(1., 1., 1.), -1),
            TrainingExample::new(vec3(-1., 1., -1.), -1),
            TrainingExample::new(vec3(1., 1., -1.), -1),
        ];

        let tree = Tree::new(5, training_points);
        assert_eq!(tree.splits, vec![vec![0.0], vec![], vec![0.0]])
    }
}
