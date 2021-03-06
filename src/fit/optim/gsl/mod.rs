use crate::foreign::gsl::multimin::*;
use nalgebra::*;
use crate::foreign::gsl::vector_double::*;
use std::ffi::c_void;
use std::mem;
use crate::foreign::gsl::utils::*;

// This module contains private structures that are useful both for the
// LBFGS and Nelder-Mean minimizers. Those structures are used internally
// by the minimizer functions inside each module.

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
struct MinProblem<T : Sized + Clone> {
    user_data : T,
    user_params : DVector<f64>,
    user_grad : DVector<f64>,
    fn_update : Box<dyn FnMut(&DVector<f64>, &mut T)->f64>,
    grad_update : Option<Box<dyn FnMut(&DVector<f64>, &mut T)->DVector<f64>>>
}

// Transform the use function into a void pointer to be passed to the C routine.
// Used by both minimizers.
unsafe extern "C" fn objective<T : Sized + Clone>(
    p : *const gsl_vector,
    extra : *mut c_void
) -> f64 {
    let extra_ptr : *mut MinProblem<T> = mem::transmute(extra);
    let extra = &mut *extra_ptr;
    let user_p : DVector<f64> = (*p).clone().into();
    //let user_p = DVectorSlice::slice_from_gsl(&*p);
    (extra.fn_update)(&user_p, &mut extra.user_data)
}




