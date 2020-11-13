use ndarray::{ArrayBase, Data, Ix2};

#[cfg(feature = "blas")]
pub use blas_utils::*;

#[derive(Clone, Copy, PartialEq)]
pub enum MemoryOrder {
    C,
    F,
}

pub fn memory_layout<S>(a: &ArrayBase<S, Ix2>) -> Option<MemoryOrder>
where
    S: Data,
{
    let (m, n) = a.dim();
    let s0 = a.strides()[0];
    let s1 = a.strides()[1];

    if s1 == 1 || n == 1 {
        return Some(MemoryOrder::C);
    }
    if s0 == 1 || m == 1 {
        return Some(MemoryOrder::F);
    }

    None
}

#[cfg(feature = "blas")]
mod blas_utils {
    use super::*;
    use cblas_sys::CBLAS_LAYOUT;
    use ndarray::{ArrayBase, Data, Ix1, Ix2};
    use std::any::TypeId;
    use std::os::raw::c_int;

    /// Return a pointer to the starting element in BLAS's view.
    ///
    /// BLAS wants a pointer to the element with lowest address,
    /// which agrees with our pointer for non-negative strides, but
    /// is at the opposite end for negative strides.
    pub unsafe fn blas_1d_params<A>(
        ptr: *const A,
        len: usize,
        stride: isize,
    ) -> (*const A, c_int, c_int) {
        // [x x x x]
        //        ^--ptr
        //        stride = -1
        //  ^--blas_ptr = ptr + (len - 1) * stride
        if stride >= 0 || len == 0 {
            (ptr, len as c_int, stride as c_int)
        } else {
            let ptr = ptr.offset((len - 1) as isize * stride);
            (ptr, len as c_int, stride as c_int)
        }
    }

    #[inline(always)]
    /// Return `true` if `A` and `B` are the same type
    pub fn same_type<A: 'static, B: 'static>() -> bool {
        TypeId::of::<A>() == TypeId::of::<B>()
    }

    // Read pointer to type `A` as type `B`.
    //
    // **Panics** if `A` and `B` are not the same type
    pub fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
        assert!(same_type::<A, B>());
        unsafe { ::std::ptr::read(a as *const _ as *const B) }
    }

    pub fn is_blas_compat_1d<S: Data>(a: &ArrayBase<S, Ix1>) -> bool {
        if !is_blas_compat_dim(a.len()) {
            return false;
        }
        if !is_blas_compat_stride(a.strides()[0]) {
            return false;
        }
        true
    }

    impl MemoryOrder {
        pub fn stride<S: Data>(&self, a: &ArrayBase<S, Ix2>) -> c_int {
            match self {
                MemoryOrder::C => a.strides()[0] as c_int,
                MemoryOrder::F => a.strides()[1] as c_int,
            }
        }
    }

    impl From<MemoryOrder> for CBLAS_LAYOUT {
        fn from(order: MemoryOrder) -> Self {
            match order {
                MemoryOrder::C => CBLAS_LAYOUT::CblasRowMajor,
                MemoryOrder::F => CBLAS_LAYOUT::CblasColMajor,
            }
        }
    }

    fn is_blas_compat_stride(s: isize) -> bool {
        s <= c_int::max_value() as isize && s >= c_int::min_value() as isize
    }

    fn is_blas_compat_dim(n: usize) -> bool {
        n <= c_int::max_value() as usize
    }

    pub fn is_blas_compat_2d<S: Data>(a: &ArrayBase<S, Ix2>) -> bool {
        let (m, n) = a.dim();
        let strides = a.strides();

        if strides[0] < 1 || strides[1] < 1 {
            return false;
        }
        if !is_blas_compat_stride(strides[0]) || !is_blas_compat_stride(strides[1]) {
            return false;
        }
        if !is_blas_compat_dim(m) || !is_blas_compat_dim(n) {
            return false;
        }
        true
    }
}
