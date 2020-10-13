use std::os::raw::c_char;
use std::ffi::CStr;
use crate::parse::*;
use crate::distr::*;
use pgdatum::{Text, Bytea};
use std::convert::TryInto;
use std::panic::catch_unwind;

/// Parses the distribution from text, assuming its validity.
fn parse_distr(distr_txt : &Text) -> AnyLikelihood {
    let distr_s : &str = distr_txt.as_ref();
    let model : AnyLikelihood = distr_s.parse().unwrap();
    model
}

#[no_mangle]
pub extern "C" fn log_prob(distr_txt : &Text) -> f64 {
    let f = || {
        let model = parse_distr(distr_txt);
        let distr : &Distribution = (&model).into();
        if let Some(obs) = distr.observations() {
            distr.log_prob(obs.into(), None)
        } else {
            std::f64::NAN
        }
    };
    match catch_unwind(f) {
        Ok(d) => d,
        Err(e) => { println!("{:?}", e); std::f64::NAN }
    }
}

#[no_mangle]
pub extern "C" fn sample_from(distr_txt : &Text) -> Text /* *const Text*/ {
    println!("calling sample_from");
    let f = || {
        let model = parse_distr(distr_txt);
        let distr : &Distribution = (&model).into();
        let sample = distr.sample();
        let mat_v = crate::parse::matrix_to_value(&sample);
        mat_v.to_string()
    };
    /*let mat_string = match catch_unwind(f) {
        Ok(d) => d,
        Err(e) => { println!("{:?}", e); String::from("(Function error)") }
    };*/
    let mat_string = f();
    let mut txt_bytes = Bytea::palloc(mat_string.as_bytes().len());
    println!("mat_string = {}", mat_string);
    txt_bytes.as_mut().copy_from_slice(mat_string.as_bytes());
    let txt : Text = txt_bytes.try_into().unwrap();
    // txt.release()
    txt
}

