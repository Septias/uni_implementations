#![allow(unused)]

use itertools::Itertools;
use nalgebra::{SMatrix, SVector};

use super::helpers::{approx_equal, sigmoid};

#[derive(Debug, Clone)]
struct Split {
    variable: usize,
    value: f32,
}

impl Split {
    fn new(variables: usize, value: f32) -> Self {
        Self {
            variable: variables,
            value,
        }
    }
}

struct Stump {
    split: Split,
    w1: f32,
    w2: f32,
}

impl Stump {
    fn new(split: Split) -> Self {
        Self {
            split,
            w1: f32::default(),
            w2: f32::default(),
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
        let splits = training_points
            .iter()
            .enumerate()
            .map(|(index, column)| {
                column
                    .iter()
                    .tuple_windows()
                    .filter(|(a, b)| !approx_equal(**a, **b, 10))
                    .map(|(x1, x2)| (x1 + x2) / 2.)
                    .dedup_by(|a, b| approx_equal(*a, *b, 10))
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
            let gain_comparison = [prediction.to_vec(), self.y.to_vec()];
            println!("prediction in iteration {} from whole tree: {:?}", i ,prediction);
            println!("choosing split from {:?}", self.splits);

            let split =self.splits.iter().map(|split| {
                let (left, right): (
                    Vec<(usize, &SVector<f32, FEATURE_SIZE>)>,
                    Vec<(usize, &SVector<f32, FEATURE_SIZE>)>,
                ) = self
                    .training_points
                    .iter()
                    .enumerate()
                    .partition(|(index, tp)| tp[split.variable] > split.value);
                
                
                let (left_y_hat, left_y): (Vec<f32>, Vec<f32>) = left
                    .iter()
                    .map(|(index, elem)| (self.y[*index], prediction[*index]))
                    .unzip();
                
                let (right_y, right_y_hat): (Vec<f32>, Vec<f32>) = right
                    .iter()
                    .map(|(index, elem)| (self.y[*index],  prediction[*index]))
                    .unzip();

                ( (self.gain(&gain_comparison, [left_y_hat, left_y], [right_y_hat, right_y]) * 1000.) as usize, split)
            }).max_by_key(|elem| elem.0);

            if let Some((score, split)) = split{
                println!("chose best split: {:?}, with score: {}", split, score);
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

    fn gain(&self, [a, b]: &[Vec<f32>; 2], [c, d]: [Vec<f32>; 2], [e, f]: [Vec<f32>; 2]) -> f32 {
        1. / 2. * (self.single_gain(a, b) - self.single_gain(&c, &d) - self.single_gain(&e, &f))
            - self.gamma
    }

    fn single_gain(&self, y: &Vec<f32>, y_hat: &Vec<f32>) -> f32 {
        let top = y
            .iter()
            .zip(y_hat)
            .map(|(y, y_hat)| ((self.g_n)(*y, *y_hat)))
            .sum::<f32>()
            .powf(2.);

        let bottom = (y.iter().map(|y| (self.h_n)(*y)).sum::<f32>() + self.lamda).powi(2);
        top / bottom
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
    fn test_gdbt() {
        let x = Matrix6x2::from_columns(&[
            [1., 2., 3., 1., 2., 3.].into(),
            [2., 1., 2., 3., 2., 3.].into(),
        ]);

        let y: Vector6<f32> = [0., 0., 0., 1., 1., 1.].into();

        //GBDT::new(4, X, g_n, h_n)
    }

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
                .map(|(y, y_hat)| g_n(*y, *y_hat))
                .collect_vec(),
            vec![-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
        );

        let mut tree = GBDT::new(1, x.into(), y, g_n, h_n);
        tree.train();
    }
}
