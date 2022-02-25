use super::helpers::squared_error;
use itertools::Itertools;

/// Decision tree learning
pub struct TrainingExample {
    pub prediction: f32,
    pub features: f32,
}

pub struct Decision<'a> {
    examples: Vec<&'a TrainingExample>,
    children: Option<[Box<Decision<'a>>; 2]>,
}

fn approx_equal(a: f32, b: f32, dp: u8) -> bool {
    let p = 10f32.powi(-(dp as i32));
    (a - b).abs() < p
}

impl<'a> Decision<'a> {
    pub fn new(
        examples: Vec<&'a TrainingExample>,
        children: Option<[Box<Decision<'a>>; 2]>,
    ) -> Self {
        Self { examples, children }
    }

    pub fn split(&mut self) {
        let splits = self.generate_split();

        let split_scores = splits
            .iter()
            .map(|split| {
                self.examples
                    .iter()
                    .partition(|example| example.features < *split)
            })
            .map(
                |(left, right): (Vec<&TrainingExample>, Vec<&TrainingExample>)| {
                    let purity_left = compute_purity(&left, get_prediction(&left));
                    let purity_right = compute_purity(&right, get_prediction(&right));
                    (purity_left + purity_right, (left, right))
                },
            );

        let best_split = split_scores.min_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

        if let Some((_, (left, right))) = best_split {
            self.children = Some([
                Box::new(Decision::new(left, None)),
                Box::new(Decision::new(right, None)),
            ]);
        }
    }

    fn generate_split(&self) -> Vec<f32> {
        self.examples
            .iter()
            .map(|x| x.features)
            .dedup_by(|a, b| approx_equal(*a, *b, 100))
            .tuple_windows()
            .map(|(a, b)| (a + b) / 2.)
            .collect()
    }
}

fn log_purity(examples: &[&TrainingExample]) {}

fn compute_purity(examples: &[&TrainingExample], prediction: f32) -> f32 {
    examples
        .iter()
        .map(|elem| squared_error(prediction, elem.prediction))
        .sum()
}

fn get_prediction(examples: &[&TrainingExample]) -> f32 {
    (1. / examples.len() as f32) * examples.iter().map(|example| example.features).sum::<f32>()
}
