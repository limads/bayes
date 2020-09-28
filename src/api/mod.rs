use std::os::raw::c_char;
use std::ffi::CStr;
use crate::parse::*;
use crate::distr::*;

mod server;

use server::*;

#[no_mangle]
pub extern "C" fn log_prob(distr_txt : *const text) -> f64 {
    // Use this for nul-terminated
    // let distr_str = unsafe{ CStr::from_ptr(text.vl_dat).to_str().unwrap() };
    let distr_s : &str = server::utf8_to_str(distr_txt);
    let model : AnyLikelihood = distr_s.parse().unwrap();
    let distr : &Distribution = (&model).into();
    let obs = distr.observations().unwrap();
    distr.log_prob(obs.into(), None)
}


