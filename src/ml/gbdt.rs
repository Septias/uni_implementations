#![allow(unused)]

use itertools::Itertools;
use nalgebra::{Matrix2x6, SMatrix, SVector};

use super::helpers::{approx_equal, sigmoid, Split};


struct Stump {
    split: Split,
    w1: f32,
    w2: f32,
}

impl Stump {
    fn new(split: Split, w1: f32, w2: f32) -> Self {
        Self {
            split,
            w1,
            w2,
        }
    }

    pub fn predict<const FEATURE_SIZE: usize>(&self, x: &SVector<f32, FEATURE_SIZE>) -> f32 {
        if x[self.split.variable] > self.split.value {
            self.w1
        } else {
            self.w2
        }
    }
}
pub struct GBDT<const TRAINING_POINTS: usize, const FEATURE_SIZE: usize> {
    depth: usize,
    splits: Vec<Split>,
    training_points: [SVector<f32, FEATURE_SIZE>; TRAINING_POINTS],
    y: [f32; TRAINING_POINTS],
    g_n: fn(f32, f32) -> f32,
    h_n: fn(f32) -> f32,
    lamda: f32,
    gamma: f32,
    stumps: Vec<Stump>,
}

impl<const TRAINING_POINTS: usize, const FEATURE_SIZE: usize> GBDT<TRAINING_POINTS, FEATURE_SIZE> {
    pub fn new(
        depth: usize,
        training_points: [SVector<f32, FEATURE_SIZE>; TRAINING_POINTS],
        y: [f32; TRAINING_POINTS],
        g_n: fn(f32, f32) -> f32,
        h_n: fn(f32) -> f32,
    ) -> Self {
        let matrix = SMatrix::<f32, FEATURE_SIZE, TRAINING_POINTS>::from_columns(&training_points)
            .transpose();

        let splits = matrix
            .column_iter()
            .enumerate()
            .map(|(index, column)| {
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
                    .map(|split_var| Split::new(index, split_var))
                    .collect_vec()
            })
            .flatten()
            .collect_vec();

        Self {
            depth,
            splits,
            training_points,
            y,
            g_n,
            h_n,
            lamda: 1.,
            gamma: 1.,
            stumps: vec![],
        }
    }

    fn train(&mut self) {
        let mut prediction = vec![0.; TRAINING_POINTS];

        for i in 0..self.depth {
            let prediction = GBDT::predict_samples(&self.stumps, &self.training_points);
            let gain = self.single_gain(&self.y.to_vec(), &prediction.to_vec());
            println!(
                "prediction in iteration {} from whole tree: {:?}",
                i, prediction
            );
            println!("choosing split from {:?}", self.splits);

            let split = self
                .splits
                .iter()
                .map(|split| {
                    let (left, right): (
                        Vec<(usize, &SVector<f32, FEATURE_SIZE>)>,
                        Vec<(usize, &SVector<f32, FEATURE_SIZE>)>,
                    ) = self
                        .training_points
                        .iter()
                        .enumerate()
                        .partition(|(index, tp)| tp[split.variable] > split.value);

                    let (left_y, left_y_hat): (Vec<f32>, Vec<f32>) = left
                        .iter()
                        .map(|(index, elem)| (self.y[*index], prediction[*index]))
                        .unzip();

                    let (right_y, right_y_hat): (Vec<f32>, Vec<f32>) = right
                        .iter()
                        .map(|(index, elem)| (self.y[*index], prediction[*index]))
                        .unzip();

                    let left_gain = self.single_gain(&left_y, &left_y_hat);
                    let right_gain = self.single_gain(&right_y, &right_y_hat);
                    let score = self.gain(
                        gain,
                        left_gain,
                        right_gain,
                    );
                    
                    // unperformant to compute this event thoug it's only needed for one 
                    // tree
                    let w1 = self.best_leaf_value(&left_y, &left_y_hat); 
                    let w2 = self.best_leaf_value(&right_y, &right_y_hat);

                    // this is a little hacky
                    ((score * 1000.) as i32, split, w1, w2)
                    
                })
                .inspect(|(score, split, ..)| println!("{:?} has score {}", split, score))
                .max_by_key(|elem| elem.0);

            if let Some((score, split, w1, w2)) = split {
                println!("chose best split: {:?}, with score: {}", split, score);
                let stump = Stump::new(split.clone(), w1, w2);
                self.stumps.push(stump);
            } else {
                println!("no more splits possible, quitting");
                return;
            }
        }
    }


    fn predict_samples(
        stumps: &Vec<Stump>,
        training_points: &[SVector<f32, FEATURE_SIZE>; TRAINING_POINTS],
    ) -> [f32; TRAINING_POINTS] {
        let mut prediction = [0.; TRAINING_POINTS];
        prediction
            .iter_mut()
            .enumerate()
            .for_each(|(index, element)| {
                let tp = &training_points[index];
                *element = stumps
                    .iter()
                    .fold(*element, |acc, stump| acc + stump.predict(tp))
            });
        prediction
    }

    fn gain(&self, gain: f32, left: f32, right: f32) -> f32 {
        0.5 * (-gain + left + right) - self.gamma
    }

    fn single_gain(&self, y: &Vec<f32>, y_hat: &Vec<f32>) -> f32 {
        let top = y
            .iter()
            .zip(y_hat)
            .map(|(y, y_hat)| ((self.g_n)(*y, *y_hat)))
            .sum::<f32>()
            .powi(2);

        let bottom = (y_hat.iter().map(|y_hat| (self.h_n)(*y_hat)).sum::<f32>() + self.lamda);
        top / bottom
    }

    fn best_leaf_value(&self, y: &Vec<f32>, y_hat: &Vec<f32>) -> f32 {
        let top = y
            .iter()
            .zip(y_hat)
            .map(|(y, y_hat)| ((self.g_n)(*y, *y_hat)))
            .sum::<f32>();

        let bottom = (y_hat.iter().map(|y_hat| (self.h_n)(*y_hat)).sum::<f32>() + self.lamda);
        -top / bottom
    }
}

fn g_n(y: f32, y_hat: f32) -> f32 {
    sigmoid(y_hat) - y
}

fn h_n(y_hat: f32) -> f32 {
    let sigm_y = sigmoid(y_hat);
    sigm_y * (1. - sigm_y)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix6x2, Vector2, Vector6};
    use super::*;

    #[test]
    fn test_gdbt_ex() {
        let x: [Vector2<f32>; 6] = [
            [1., 2.].into(),
            [1., 3.].into(),
            [4., 1.].into(),
            [2., 1.].into(),
            [3., 2.].into(),
            [4., 3.].into(),
        ];

        let y = [1., 1., 1., 0., 0., 0.];
        let y_hat = Vector6::<f32>::zeros();

        // test if initial h_n is computed correctly
        assert_eq!(
            y_hat.iter().map(|y_hat| h_n(*y_hat)).collect_vec(),
            vec![0.25; 6]
        );

        // test if initial g_n is computed correctly
        assert_eq!(
            y.iter()
                .zip(&y_hat)
                .inspect(|value| println!("{:?}", value))
                .map(|(y, y_hat)| g_n(*y, *y_hat))
                .collect_vec(),
            vec![-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
        );

        let mut tree = GBDT::new(3, x.into(), y, g_n, h_n);
        tree.train();
    }
}
