use cblas_sys::CBLAS_LAYOUT;
use ndarray::{ArrayBase, Data, Ix1, Ix2};
use std::any::TypeId;

#[allow(non_camel_case_types)]
type blas_index = std::os::raw::c_int;

/// Return a pointer to the starting element in BLAS's view.
///
/// BLAS wants a pointer to the element with lowest address,
/// which agrees with our pointer for non-negative strides, but
/// is at the opposite end for negative strides.
pub unsafe fn blas_1d_params<A>(
    ptr: *const A,
    len: usize,
    stride: isize,
) -> (*const A, blas_index, blas_index) {
    // [x x x x]
    //        ^--ptr
    //        stride = -1
    //  ^--blas_ptr = ptr + (len - 1) * stride
    if stride >= 0 || len == 0 {
        (ptr, len as blas_index, stride as blas_index)
    } else {
        let ptr = ptr.offset((len - 1) as isize * stride);
        (ptr, len as blas_index, stride as blas_index)
    }
}

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
pub fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
    assert!(same_type::<A, B>());
    unsafe { ::std::ptr::read(a as *const _ as *const B) }
}

pub fn blas_compat_1d<A, S>(a: &ArrayBase<S, Ix1>) -> bool
where
    S: Data,
    A: 'static,
    S::Elem: 'static,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    if a.len() > blas_index::max_value() as usize {
        return false;
    }
    let stride = a.strides()[0];
    if stride > blas_index::max_value() as isize || stride < blas_index::min_value() as isize {
        return false;
    }
    true
}

pub enum MemoryOrder {
    C,
    F,
}

fn blas_row_major_2d<A, S>(a: &ArrayBase<S, Ix2>) -> bool
where
    S: Data,
    A: 'static,
    S::Elem: 'static,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    is_blas_2d(a.dim(), a.strides(), MemoryOrder::C)
}

fn blas_column_major_2d<A, S>(a: &ArrayBase<S, Ix2>) -> bool
where
    S: Data,
    A: 'static,
    S::Elem: 'static,
{
    if !same_type::<A, S::Elem>() {
        return false;
    }
    is_blas_2d(a.dim(), a.strides(), MemoryOrder::F)
}

fn is_blas_2d((m, n): (usize, usize), stride: &[isize], order: MemoryOrder) -> bool {
    let s0 = stride[0];
    let s1 = stride[1];
    let (inner_stride, outer_dim) = match order {
        MemoryOrder::C => (s1, n),
        MemoryOrder::F => (s0, m),
    };
    if !(inner_stride == 1 || outer_dim == 1) {
        return false;
    }
    if s0 < 1 || s1 < 1 {
        return false;
    }
    if (s0 > blas_index::max_value() as isize || s0 < blas_index::min_value() as isize)
        || (s1 > blas_index::max_value() as isize || s1 < blas_index::min_value() as isize)
    {
        return false;
    }
    if m > blas_index::max_value() as usize || n > blas_index::max_value() as usize {
        return false;
    }
    true
}

pub fn blas_layout<A, S>(a: &ArrayBase<S, Ix2>) -> Option<CBLAS_LAYOUT>
where
    S: Data,
    A: 'static,
    S::Elem: 'static,
{
    if blas_row_major_2d::<A, _>(a) {
        Some(CBLAS_LAYOUT::CblasRowMajor)
    } else if blas_column_major_2d::<A, _>(a) {
        Some(CBLAS_LAYOUT::CblasColMajor)
    } else {
        None
    }
}
