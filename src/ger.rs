use ndarray::{ArrayBase, Data, DataMut, Ix1, Ix2, LinalgScalar};

#[cfg(feature = "blas")]
use crate::blas_utils::*;
#[cfg(feature = "blas")]
use cblas_sys as blas_sys;

#[cfg(feature = "blas")]
#[allow(non_camel_case_types)]
type blas_index = std::os::raw::c_int;

pub trait Ger<V1, V2> {
    type Elem;
    fn ger(&mut self, alpha: Self::Elem, x: &V1, y: &V2);
}

impl<A, S1, S2, S3> Ger<ArrayBase<S2, Ix1>, ArrayBase<S3, Ix1>> for ArrayBase<S1, Ix2>
where
    S1: DataMut<Elem = A>,
    S2: Data<Elem = A>,
    S3: Data<Elem = A>,
    A: LinalgScalar,
{
    type Elem = A;

    fn ger(&mut self, alpha: Self::Elem, x: &ArrayBase<S2, Ix1>, y: &ArrayBase<S3, Ix1>) {
        let (m, n) = self.dim();
        assert!(m == x.len());
        assert!(n == y.len());
        ger_impl(self, alpha, x, y);
    }
}

fn ger_generic<A, S1, S2, S3>(
    z: &mut ArrayBase<S1, Ix2>,
    alpha: A,
    x: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix1>,
) where
    S1: DataMut<Elem = A>,
    S2: Data<Elem = A>,
    S3: Data<Elem = A>,
    A: LinalgScalar,
{
    let (m, _) = z.dim();
    for i in 0..m {
        unsafe {
            z.row_mut(i).scaled_add(alpha * *x.uget(i), y);
        }
    }
}

#[cfg(not(feature = "blas"))]
fn ger_impl<A, S1, S2, S3>(
    z: &mut ArrayBase<S1, Ix2>,
    alpha: A,
    x: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix1>,
) where
    S1: DataMut<Elem = A>,
    S2: Data<Elem = A>,
    S3: Data<Elem = A>,
    A: LinalgScalar,
{
    ger_generic(z, alpha, x, y);
}

#[cfg(feature = "blas")]
fn ger_impl<A, S1, S2, S3>(
    z: &mut ArrayBase<S1, Ix2>,
    alpha: A,
    x: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix1>,
) where
    S1: DataMut<Elem = A>,
    S2: Data<Elem = A>,
    S3: Data<Elem = A>,
    A: LinalgScalar,
{
    macro_rules! ger {
        ($ty:ty, $ger:ident) => {{
            if same_type::<A, $ty>()
                && is_blas_compat_2d(z)
                && is_blas_compat_1d(x)
                && is_blas_compat_1d(y)
            {
                if let Some(layout) = memory_layout(z) {
                    let (m, n) = z.dim();
                    let stride = layout.stride(z);
                    let layout = layout.into();
                    unsafe {
                        let (x_ptr, _, incx) = blas_1d_params(x.as_ptr(), x.len(), x.strides()[0]);
                        let (y_ptr, _, incy) = blas_1d_params(y.as_ptr(), y.len(), y.strides()[0]);
                        blas_sys::$ger(
                            layout,
                            m as blas_index,
                            n as blas_index,
                            cast_as(&alpha),
                            x_ptr as *const _,
                            incx,
                            y_ptr as *const _,
                            incy,
                            z.as_mut_ptr() as *mut _,
                            stride,
                        );
                    }
                    return;
                }
            }
        }};
    }

    ger! {f32, cblas_sger};
    ger! {f64, cblas_dger};

    ger_generic(z, alpha, x, y);
}
