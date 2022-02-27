use nalgebra::{RowSVector, SMatrix, SVector};

pub struct LinearRegression<const X_HEIGHT: usize, const X_WIDTH: usize> {
    w: SVector<f32, X_WIDTH>,
}

impl<const X_HEIGHT: usize, const X_WIDTH: usize> LinearRegression<X_HEIGHT, X_WIDTH> {
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
    fn test_lin_reg() {
        let x: SMatrix<f32, 3, 2> =
            SMatrix::from_columns(&[[-0.8, 0.3, 1.5].into(), [2.8, -2.2, 1.1].into()]);
        let y: Vector3<f32> = [-8.5, 12.8, 3.8].into();

        let mut lin_reg = LinearRegression::new();
        assert_eq!(lin_reg.solve(x, y), vec2(4.198_817, -3.062_021_3));

        let predictions = x
            .row_iter()
            .map(|row| lin_reg.predict(row.into_owned()))
            .collect_vec();

        assert_eq!(predictions, [-11.932_713, 7.996_092, 2.930_002])
    }

    #[test]
    fn test_lin_reg_bias() {
        let x: SMatrix<f32, 3, 3> = SMatrix::from_columns(&[
            SVector::repeat(1.),
            [-0.8, 0.3, 1.5].into(),
            [2.8, -2.2, 1.1].into(),
        ]);
        let y: Vector3<f32> = [-8.5, 12.8, 3.8].into();

        let mut lin_reg = LinearRegression::new();
        let sol = lin_reg.solve(x, y);
        assert_eq!(sol, vec3(3.911_215, 2.626_168_3, -3.682_243_3));

        let predictions = x
            .row_iter()
            .map(|row| lin_reg.predict(row.into_owned()))
            .collect_vec();

        assert_eq!(predictions, [-8.5, 12.800_001, 3.799_999_7])
    }

    #[test]
    fn no_test() {
        let x: SMatrix<f32, 3, 3> = SMatrix::from_columns(&[
            SVector::repeat(1.),
            [-0.8, 0.3, 1.5].into(),
            [2.8, -2.2, 1.1].into(),
        ]);
        let y: Vector3<f32> = [-8.5, 12.8, 3.8].into();

        let mut lin_reg = LinearRegression::new();
        lin_reg.solve(x, y);

        let with_bias: SVector<f32, 2> = [
            lin_reg.predict(vec3(1.0, -2., 2.).transpose()),
            lin_reg.predict(vec3(1.0, -4., 15.).transpose()),
        ]
        .into();

        let x: SMatrix<f32, 3, 2> =
            SMatrix::from_columns(&[[-0.8, 0.3, 1.5].into(), [2.8, -2.2, 1.1].into()]);
        let y: Vector3<f32> = [-8.5, 12.8, 3.8].into();
        let mut lin_reg = LinearRegression::new();
        lin_reg.solve(x, y);

        let no_bias: SVector<f32, 2> = [
            lin_reg.predict(vec2(-2., 2.).transpose()),
            lin_reg.predict(vec2(-4., 15.).transpose()),
        ]
        .into();

        let func = |x: f32, y: f32| -> f32 { 5. + 2. * x - 4. * y };
        let ground_truth = vec2(func(-2., 2.), func(-4., 15.));

        let rmse = [
            rmse(&with_bias, &ground_truth),
            rmse(&no_bias, &ground_truth),
        ];
        assert_eq!(rmse, [1.463_688_9, 5.322_166_4]);
    }
}
