use approx::abs_diff_eq;
use ndarray::{arr1, arr2, Array1, Array2, ShapeBuilder};
use ndarray_matops::Ger;
use sprs::{CsMat, CsVec};

#[test]
fn ger_f32() {
    let mut m: Array2<f32> = Array2::zeros((3, 5));
    let x: Array1<f32> = arr1(&[1.0, 2.0, 3.0]);
    let y: Array1<f32> = arr1(&[1.5, 2.5, 3.5, 4.5, 5.5]);
    m.ger(2.0, &x, &y);
    abs_diff_eq!(
        m,
        arr2(&[
            [3.0, 5.0, 7.0, 9.0, 11.0],
            [6.0, 10.0, 14.0, 18.0, 22.0],
            [9.0, 15.0, 21.0, 27.0, 33.0]
        ])
    );
}

#[test]
fn ger_f64() {
    let mut m: Array2<f64> = Array2::zeros((3, 5));
    let x: Array1<f64> = arr1(&[1.0, 2.0, 3.0]);
    let y: Array1<f64> = arr1(&[1.5, 2.5, 3.5, 4.5, 5.5]);
    m.ger(2.0, &x, &y);
    abs_diff_eq!(
        m,
        arr2(&[
            [3.0, 5.0, 7.0, 9.0, 11.0],
            [6.0, 10.0, 14.0, 18.0, 22.0],
            [9.0, 15.0, 21.0, 27.0, 33.0]
        ])
    );
}

#[test]
fn ger_f32_f() {
    let mut m: Array2<f32> = Array2::zeros((3, 5).f());
    let x: Array1<f32> = arr1(&[1.0, 2.0, 3.0]);
    let y: Array1<f32> = arr1(&[1.5, 2.5, 3.5, 4.5, 5.5]);
    m.ger(2.0, &x, &y);
    abs_diff_eq!(
        m,
        arr2(&[
            [3.0, 5.0, 7.0, 9.0, 11.0],
            [6.0, 10.0, 14.0, 18.0, 22.0],
            [9.0, 15.0, 21.0, 27.0, 33.0]
        ])
    );
}

#[test]
fn ger_f64_f() {
    let mut m: Array2<f64> = Array2::zeros((3, 5).f());
    let x: Array1<f64> = arr1(&[1.0, 2.0, 3.0]);
    let y: Array1<f64> = arr1(&[1.5, 2.5, 3.5, 4.5, 5.5]);
    m.ger(2.0, &x, &y);
    abs_diff_eq!(
        m,
        arr2(&[
            [3.0, 5.0, 7.0, 9.0, 11.0],
            [6.0, 10.0, 14.0, 18.0, 22.0],
            [9.0, 15.0, 21.0, 27.0, 33.0]
        ])
    );
}

#[test]
fn ger_f32_sparse() {
    let mut m: CsMat<f32> = CsMat::zero((3, 5));
    let x: CsVec<f32> = CsVec::new(3, vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
    let y: CsVec<f32> = CsVec::new(5, vec![0, 2, 4], vec![1.5, 3.5, 5.5]);
    m.ger(2.0, &x, &y);
    abs_diff_eq!(
        m,
        CsMat::csr_from_dense(
            arr2(&[
                [3.0, 0.0, 7.0, 0.0, 11.0],
                [6.0, 0.0, 14.0, 0.0, 22.0],
                [9.0, 0.0, 21.0, 0.0, 33.0]
            ])
            .view(),
            0.0
        )
    );
}

#[test]
fn ger_f64_sparse() {
    let mut m: CsMat<f64> = CsMat::zero((3, 5));
    let x: CsVec<f64> = CsVec::new(3, vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
    let y: CsVec<f64> = CsVec::new(5, vec![0, 2, 4], vec![1.5, 3.5, 5.5]);
    m.ger(2.0, &x, &y);
    abs_diff_eq!(
        m,
        CsMat::csr_from_dense(
            arr2(&[
                [3.0, 0.0, 7.0, 0.0, 11.0],
                [6.0, 0.0, 14.0, 0.0, 22.0],
                [9.0, 0.0, 21.0, 0.0, 33.0]
            ])
            .view(),
            0.0
        )
    );
}
