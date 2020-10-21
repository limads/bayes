use std::ffi::c_void;
use std::mem;
use crate::model::Model;
use crate::distr::*;
use std::convert::AsMut;
use std::ptr;
use std::slice;
use nalgebra::DVectorSlice;

#[repr(C)]
pub struct DistrPtr {
    pub model : *mut c_void,
    pub lp_func : extern "C" fn(*mut c_void, *const f64, usize)->f64
}

/// Calculate the model log-probability wrt. natural parameter param.
#[no_mangle]
pub extern "C" fn model_log_prob(model : *mut c_void, param : *const f64, param_len : usize) -> f64 {
    unsafe {
        if model == ptr::null_mut() {
            panic!("Informed null pointer to log_prob. Aborting.");
        }
        if param == ptr::null() {
            panic!("Informed null pointer to param. Aborting.");
        }
        let model : *mut Model = mem::transmute(model);
        let distr : &mut dyn Distribution = (&mut *model).into();
        let param_vals = slice::from_raw_parts(param, param_len);
        let vs : DVectorSlice<'_, f64> = param_vals.into();
        distr.set_parameter(vs, true);
        if let Some(obs) = distr.observations() {
            distr.log_prob(obs.into(), None)
        } else {
            panic!("Distribution does not have observations");
        }
    }
}

