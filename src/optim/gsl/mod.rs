use crate::gsl::multimin::*;
use nalgebra::*;
use crate::gsl::vector_double::*;
// use crate::gsl::matrix_double::*;
use std::ffi::c_void;
use std::mem;
use crate::gsl::utils::*;

pub mod nls;

pub mod bfgs;

pub mod nelder;

#[derive(Debug)]
struct OptimizationStep {
    obj : f64,
    jac_norm : f64,
    param_norm : f64
}

#[repr(C)]
struct MinProblem<T : Sized> {
    user_data : T,
    user_params : DVector<f64>,
    user_grad : DVector<f64>,
    fn_update : Box<dyn FnMut(&DVector<f64>, &mut T)->f64>,
    grad_update : Option<Box<dyn FnMut(&DVector<f64>, &mut T)->DVector<f64>>>
}

unsafe extern "C" fn objective<T : Sized>(
    p : *const gsl_vector,
    extra : *mut c_void
) -> f64 {
    let extra_ptr : *mut MinProblem<T> = mem::transmute(extra);
    let extra = &mut *extra_ptr;
    let user_p : DVector<f64> = (*p).clone().into();
    (extra.fn_update)(&user_p, &mut extra.user_data)
}





