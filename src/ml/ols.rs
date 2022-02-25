use nalgebra::{RowSVector, SMatrix, SVector};

pub struct OLS<const X_HEIGHT: usize, const X_WIDTH: usize> {
    w: SVector<f32, X_WIDTH>,
}

impl<const X_HEIGHT: usize, const X_WIDTH: usize> OLS<X_HEIGHT, X_WIDTH> {
    pub fn new() -> Self {
        Self {
            w: SVector::zeros(),
        }
    }

    pub fn solve(
        &mut self,
        x: SMatrix<f32, X_HEIGHT, X_WIDTH>,
        y: SVector<f32, X_HEIGHT>,
    ) -> SVector<f32, X_WIDTH> {
        let xt: SMatrix<f32, X_WIDTH, X_HEIGHT> = x.transpose();
        let t1: SMatrix<f32, X_WIDTH, X_WIDTH> = xt * x;
        let inv = (t1).try_inverse().unwrap();
        self.w = inv * xt * y;
        self.w
    }

    pub fn predict(&self, x: RowSVector<f32, X_WIDTH>) -> f32 {
        (x * self.w)[0]
    }
}

/// Solve ordinary-least-squares problems

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ml::helpers::rmse, vec2, vec3};
    use itertools::Itertools;
    use nalgebra::Vector3;

    #[test]
    fn test_ols() {
        let x: SMatrix<f32, 3, 2> =
            SMatrix::from_columns(&[[-0.8, 0.3, 1.5].into(), [2.8, -2.2, 1.1].into()]);
        let y: Vector3<f32> = [-8.5, 12.8, 3.8].into();

        let mut ols = OLS::new();
        assert_eq!(ols.solve(x, y), vec2(4.19881678, -3.06202126));

        let predictions = x
            .row_iter()
            .map(|row| ols.predict(row.into_owned()))
            .collect_vec();

        assert_eq!(predictions, [-11.9327126, 7.99609184, 2.93000197])
    }

    #[test]
    fn test_old_bias() {
        let x: SMatrix<f32, 3, 3> = SMatrix::from_columns(&[
            SVector::repeat(1.),
            [-0.8, 0.3, 1.5].into(),
            [2.8, -2.2, 1.1].into(),
        ]);
        let y: Vector3<f32> = [-8.5, 12.8, 3.8].into();

        let mut ols = OLS::new();
        let sol = ols.solve(x, y);
        assert_eq!(sol, vec3(3.91121507, 2.62616825, -3.68224335));

        let predictions = x
            .row_iter()
            .map(|row| ols.predict(row.into_owned()))
            .collect_vec();

        assert_eq!(predictions, [-8.5, 12.8000011, 3.79999971])
    }

    #[test]
    fn no_test() {
        let x: SMatrix<f32, 3, 3> = SMatrix::from_columns(&[
            SVector::repeat(1.),
            [-0.8, 0.3, 1.5].into(),
            [2.8, -2.2, 1.1].into(),
        ]);
        let y: Vector3<f32> = [-8.5, 12.8, 3.8].into();

        let mut ols = OLS::new();
        ols.solve(x, y);

        let with_bias: SVector<f32, 2> = [
            ols.predict(vec3(1.0, -2., 2.).transpose()),
            ols.predict(vec3(1.0, -4., 15.).transpose()),
        ]
        .into();

        let x: SMatrix<f32, 3, 2> =
            SMatrix::from_columns(&[[-0.8, 0.3, 1.5].into(), [2.8, -2.2, 1.1].into()]);
        let y: Vector3<f32> = [-8.5, 12.8, 3.8].into();
        let mut ols = OLS::new();
        ols.solve(x, y);

        let no_bias: SVector<f32, 2> = [
            ols.predict(vec2(-2., 2.).transpose()),
            ols.predict(vec2(-4., 15.).transpose()),
        ]
        .into();

        let func = |x: f32, y: f32| -> f32 { 5. + 2. * x - 4. * y };
        let ground_truth = vec2(func(-2., 2.), func(-4., 15.));

        let rmse = [
            rmse(&with_bias, &ground_truth),
            rmse(&no_bias, &ground_truth),
        ];
        assert_eq!(rmse, [1.46368885, 5.32216644]);
    }
}
