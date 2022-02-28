use itertools::Itertools;
use nalgebra::{RowSVector, SMatrix, SVector};
use std::{fmt::Debug, hash::Hash};

pub fn solve_lda<T, const X_WIDTH: usize, const X_HEIGHT: usize>(
    x: SMatrix<f32, X_HEIGHT, X_WIDTH>,
    y: SVector<T, X_HEIGHT>,
) -> (f32, SVector<f32, X_WIDTH>)
where
    T: Eq + Hash + Debug,
{
    // calculate the mean of the classes
    let mut m1: RowSVector<f32, X_WIDTH> = RowSVector::zeros();
    let mut m2: RowSVector<f32, X_WIDTH> = RowSVector::zeros();

    let unique: [&T; 2] = y
        .iter()
        .unique()
        .collect_vec()
        .try_into()
        .expect("no more than two classes supported");

    let [mut c_1, mut c_2] = [0., 0.];
    for (i, row) in x.row_iter().enumerate() {
        if y[i] == *unique[0] {
            m1 += row;
            c_1 += 1.;
        } else {
            m2 += row;
            c_2 += 1.;
        }
    }

    m1 /= c_1;
    m2 /= c_2;

    // Calculate the covariances
    // S_k = \frac1{N_k−1} \sum_{n∈C_k} (x_n − m_k)(x_n − m_k)^T

    let mut sk1 = SMatrix::zeros();
    let mut sk2 = SMatrix::zeros();

    for (i, row) in x.row_iter().enumerate() {
        if y[i] == *unique[0] {
            sk1 += (row - m1).transpose() * (row - m1);
        } else {
            sk2 += (row - m2).transpose() * (row - m2);
        }
    }

    sk1 /= c_1 - 1.;
    sk2 /= c_2 - 1.;

    // Calculate the Inbetween
    // S_b = 1/2 (S_1 + S_2)
    let s_w = 0.5 * (sk1 + sk2);

    // calculate the w
    // J(w) = \frac{(m2 - m1)^2}{s_1^2 + s_2^2}
    // w = S^(-1)_W(m2 - m1)
    let w = s_w.try_inverse().unwrap() * (m2 - m1).transpose();
    let b = (-0.5 * w.transpose() * (m1 + m2).transpose())[0];
    (b, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec2;

    #[derive(PartialEq, Eq, Debug, Hash, Clone)]
    enum Logo {
        NASA,
        ALDI,
    }

    #[test]
    fn test_lda() {
        let x = SMatrix::<f32, 12, 2>::from_columns(&[
            [4.9, 4.7, 4.5, 5.8, 4.9, 4., 5., 8.2, 8.7, 6.9, 7.2, 9.].into(),
            [10, 15, 12, 25, 10, 15, 43, 45, 50, 55, 52, 51]
                .map(|x| x as f32)
                .into(),
        ]);

        let y = [
            Logo::NASA,
            Logo::NASA,
            Logo::NASA,
            Logo::NASA,
            Logo::NASA,
            Logo::NASA,
            Logo::ALDI,
            Logo::ALDI,
            Logo::ALDI,
            Logo::ALDI,
            Logo::ALDI,
            Logo::ALDI,
        ]
        .into();

        let b = solve_lda(x, y);
        assert_eq!(b, (-42.2037201, [-0.216408253, 1.36400926].into()));
        assert_eq!((b.1.transpose() * vec2(6., 25.))[0] + b.0, -9.401939);
    }
}
