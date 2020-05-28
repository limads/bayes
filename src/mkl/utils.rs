use nalgebra as na;
use std::ffi::c_void;
use nalgebra::base::storage::Storage;
use nalgebra::*;

pub unsafe fn fetch_ptrs<N1, R1, C1, S1, N2, R2, C2, S2>(
    a : &na::Matrix<N1, R1, C1, S1>,
    b : &mut na::Matrix<N2, R2, C2, S2>
) -> (*mut std::ffi::c_void, usize, *mut std::ffi::c_void, usize)
    where
        N1 : Scalar,
        R1 : Dim,
        C1 : Dim,
        S1 : Storage<N1,R1,C1>,
        N2 : Scalar,
        R2 : Dim,
        C2 : Dim,
        S2 : Storage<N2,R2,C2>
{
    let ptr_a = a.data.ptr() as *mut std::ffi::c_void;
    let n_a   = a.nrows() * a.ncols();
    let ptr_b = b.data.ptr() as *mut std::ffi::c_void;
    let n_b = b.nrows() * b.ncols();
    (ptr_a, n_a, ptr_b, n_b)
}

pub unsafe fn fetch_ptr<N, R, C, S>(
    a : &na::Matrix<N, R, C, S>
) -> (*mut std::ffi::c_void, usize)
    where
        N : Scalar,
        R : Dim,
        C : Dim,
        S : Storage<N,R,C>
{
    let ptr = a.data.ptr() as *mut std::ffi::c_void;
    let n = a.nrows() * a.ncols();
    (ptr, n)
}


