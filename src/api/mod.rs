use std::os::raw::c_char;
use std::ffi::CStr;
use crate::parse::*;
use crate::distr::*;

mod pgserver;

use pgserver::*;

// The symbols of PG_MODULE_MAGIC already live in the compiled pg_helper module.
// We declare them here so they will be re-exported to the crate .so target.
// #[no_mangle]
// pub mod pgsymbols;

// Compile and install the postgres extension with:
// pg_install -d --link "-lgsl -lgslcblas -lm -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -ldl -liomp5"
#[no_mangle]
pub extern "C" fn log_prob(distr_txt : &Text) -> f64 {
    let distr_s : &str = distr_txt.as_ref();
    let model : AnyLikelihood = distr_s.parse().unwrap();
    let distr : &Distribution = (&model).into();
    let obs = distr.observations().unwrap();
    distr.log_prob(obs.into(), None)
}


