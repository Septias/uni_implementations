use std::f32::consts::E;

use itertools::Itertools;
use nalgebra::{SVector, RowSVector, SMatrix};
use super::helpers::approx_equal;

// TODO: Splits in both directions
#[derive(Default)]
struct Stump {
    alpha: f32,
    variable: usize,
    split: f32,
}

impl Stump {
    pub fn new(variable: usize, split: f32) -> Self {
        Self {
            alpha: 0.,
            variable,
            split,
        }
    }

    pub fn compute_alpha<const TRAINING_POINTS: usize, const FEATURE_SIZE: usize>(
        &mut self,
        training_points: &[TrainingExample<FEATURE_SIZE>; TRAINING_POINTS],
        weights: &SVector<f32, TRAINING_POINTS>,
    ) {
        let wrong_classified: f32 = training_points
            .iter()
            .zip(weights)
            .map(|(point, weight)| {
                if !approx_equal(self.predict(&point.position), point.classification, 10) {
                    *weight
                } else {
                    0.
                }
            })
            .sum();

        let err_k = wrong_classified / weights.iter().sum::<f32>();

        self.alpha = ((1. - err_k) / err_k).log(E)
    }

    pub fn predict<const FEATURE_SIZE: usize>(&self, x: &SVector<f32, FEATURE_SIZE>) -> f32 {
        if x[self.variable] > self.split {
            1.
        } else {
            -1.
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct TrainingExample<const FEATURE_SIZE: usize> {
    pub position: SVector<f32, FEATURE_SIZE>,
    pub classification: f32,
}

impl<const FEATURE_SIZE: usize> TrainingExample<FEATURE_SIZE> {
    pub fn new(position: SVector<f32, FEATURE_SIZE>, classification: f32) -> Self {
        Self {
            position,
            classification,
        }
    }
}

pub struct Tree<const TRAINING_POINTS: usize, const FEATURE_SIZE: usize> {
    depth: usize,
    stumps: Vec<Stump>,
    splits: Vec<Vec<f32>>,
    training_points: [TrainingExample<FEATURE_SIZE>; TRAINING_POINTS],
}

impl<const TRAINING_POINTS: usize, const FEATURE_SIZE: usize> Tree<TRAINING_POINTS, FEATURE_SIZE> {
    pub fn new(
        depth: usize,
        training_points: [TrainingExample<FEATURE_SIZE>; TRAINING_POINTS],
    ) -> Self {
        let rows: Vec<RowSVector<f32, FEATURE_SIZE>> = training_points
            .iter()
            .map(|tp| tp.position.transpose())
            .collect_vec();

        let matrix = SMatrix::<f32, TRAINING_POINTS, FEATURE_SIZE>::from_rows(&rows);

        let splits = matrix
            .column_iter()
            .map(|column| {
                column
                    .iter()
                    .tuple_windows()
                    .filter(|(a, b)| !approx_equal(**a, **b, 10))
                    .map(|(x1, x2)| (x1 + x2) / 2.)
                    .dedup_by(|a, b| approx_equal(*a, *b, 10))
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
        let mut w: SVector<f32, TRAINING_POINTS> = SVector::repeat( 1. / TRAINING_POINTS as f32);

        let y: SVector<f32, TRAINING_POINTS> = SMatrix::from_iterator(
            self.training_points.iter().map(|tp| tp.classification),
        );
        
        for _i in 0..self.depth {
            // 1. Find best split
            let mut best = (f32::MAX, Stump::default());
            for (x, split_side) in self.splits.iter().enumerate() {
                for (_y, split) in split_side.iter().enumerate() {
                    let decision_stump = Stump::new(x, *split);
                    let y_hat: SVector<f32, TRAINING_POINTS> = SMatrix::from_iterator(
                        self.training_points
                            .iter()
                            .map(|tp| decision_stump.predict(&tp.position)),
                    );
                    
                    let value = loss_fn(y, y_hat, w);

                    if value < best.0 {
                        best.0 = value;
                        best.1 = decision_stump;
                    }
                }
            }

            // 2. Compute alpha for decision tree
            let mut tree = best.1;
            tree.compute_alpha(&self.training_points, &w);

            // 3. Update w_i
            // w_i = w_i * exp(a_k * I(y_i \ne f^(k)(x_i)))
            w.iter_mut().enumerate().for_each(|(i, w)| {
                *w *= (tree.alpha
                    * indicate_ne(
                        self.training_points[i].classification,
                        tree.predict(&self.training_points[i].position),
                    ))
                .exp()
            });
            self.stumps.push(tree);
        }
    }

    pub fn predict(&self, x: SVector<f32, FEATURE_SIZE>) -> f32 {
        let sum: f32 = self
            .stumps
            .iter()
            .map(|stump| stump.predict(&x) * stump.alpha)
            .sum();
        if sum > 0. {
            1.
        } else {
            -1.
        }
    }
}

fn indicate_ne(y: f32, y_hat: f32) -> f32 {
    if approx_equal(y, y_hat, 10) {
        0.
    } else {
        1.
    }
}

fn loss_fn<const T: usize>(y: SVector<f32, T>, y_hat: SVector<f32, T>, w: SVector<f32, T>) -> f32 {
    y.component_mul(&y_hat)
        .map(|elem| (-elem).exp())
        .component_mul(&w)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3;
    

    #[test]
    fn test_tree_constructor() {
        let training_points = [
            TrainingExample::new(vec3(-1., 1., 1.), -1.),
            TrainingExample::new(vec3(1., 1., 1.), 1.),
            TrainingExample::new(vec3(-1., 1., -1.), 1.),
            TrainingExample::new(vec3(1., 1., -1.), 1.),
        ];

        let tree = Tree::new(5, training_points);
        assert_eq!(tree.splits, vec![vec![0.0], vec![], vec![0.0]])
    }

    #[test]
    fn test_tree_learning() {
        let training_points = [
            TrainingExample::new(vec3(-1., 1., 1.), -1.),
            TrainingExample::new(vec3(1., 1., 1.), 1.),
            TrainingExample::new(vec3(-1., 1., -1.), 1.),
            TrainingExample::new(vec3(1., 1., -1.), 1.),
        ];

        let mut tree = Tree::new(5, training_points);
        tree.train();
        

        let prediction = vec![[1., -1., -1.], [-1., -1., -1.], [1., -1., 1.]].iter().map(|tp|{
            tree.predict((*tp).into())
        }).collect_vec();

        println!("{:?}", prediction);
    }
}
