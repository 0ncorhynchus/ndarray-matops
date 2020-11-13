use approx::abs_diff_eq;
use ndarray::{arr1, arr2, Array1, Array2};
use ndarray_matops::Ger;

#[test]
fn ger_f32() {
    let mut m: Array2<f32> = Array2::zeros((3, 5));
    let x: Array1<f32> = arr1(&[1.0, 2.0, 3.0]);
    let y: Array1<f32> = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    m.ger(2.0, &x, &y);
    abs_diff_eq!(
        m,
        arr2(&[
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [4.0, 8.0, 12.0, 16.0, 20.0],
            [6.0, 12.0, 18.0, 24.0, 30.0]
        ])
    );
}

#[test]
fn ger_f64() {
    let mut m: Array2<f64> = Array2::zeros((3, 5));
    let x: Array1<f64> = arr1(&[1.0, 2.0, 3.0]);
    let y: Array1<f64> = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    m.ger(2.0, &x, &y);
    abs_diff_eq!(
        m,
        arr2(&[
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [4.0, 8.0, 12.0, 16.0, 20.0],
            [6.0, 12.0, 18.0, 24.0, 30.0]
        ])
    );
}
