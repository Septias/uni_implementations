#![allow(unused)]
use itertools::Itertools;
use nalgebra::{Matrix2, RowSVector, SMatrix, SVector, Vector};
use std::{fmt::Debug, hash::Hash};

use super::helpers::sigmoid;


// TODO: make X_WIDTH and X_HEIGHT mirror matrix-syntax mxn
struct LogRes<const X_HEIGHT: usize, const X_WIDTH: usize> {
    x: SMatrix<f32, X_HEIGHT, X_WIDTH>,
    y: SVector<f32, X_HEIGHT>,
    w: SVector<f32, X_WIDTH>,
    alpha: f32,
}

impl<const X_WIDTH: usize, const X_HEIGHT: usize>
    LogRes<X_HEIGHT, X_WIDTH>
{
    fn new(
        x: SMatrix<f32, X_HEIGHT, X_WIDTH>,
        y: SVector<f32, X_HEIGHT>,
        alpha: f32
    ) -> Self {
        Self { x, y, w: SVector::zeros(), alpha }
    }

    /// we use the loss-function
    /// J = - \sum{n=1}N y_n log p_n + (1 - y_n) log (1-p_n)
    /// transforms to
    /// - \sum{n=1}N y_n * x_n * w - log(1 + e^{x_n * w})
    /// This derived with respect to w is
    /// -X^T(y - p)
    /// p is this Vector: (p(y = 1 | X = x_1; w),... ,p(y = 1 | X = x_N; w))^T
    fn step(&mut self) {
        let yprop = self.y.sum() / self.y.len() as f32;

        let propability_iter = self.x.row_iter().map(|row| sigmoid((row * self.w)[0]));
        let propability_vec: SVector<f32, X_HEIGHT> = SVector::from_iterator(propability_iter);

        let xt_y_m_p = self.x.transpose() * (self.y - propability_vec);
        
        self.w += self.alpha * xt_y_m_p;
    }

    fn predict(&self, x: RowSVector<f32, X_WIDTH>) -> f32 {
        return sigmoid((self.w * x)[0])
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix4x3};

    use super::*;

    #[test]
    fn test_log_res() {

        let x: Matrix4x3<f32> = SMatrix::from_columns(&[
            SVector::repeat(1.),
            [2., 3., -4., -2.].into(),
            [4., 3., -2., -6.].into(),
        ]);
        let y: SVector<f32, 4> = [1., 1., 0., 0.].into(); 
        
        let reg_res = LogRes::<4, 3>::new(x, y, 0.5); 
        let res = reg_res.predict([1.0, -1., 1.].into());
    }
}
