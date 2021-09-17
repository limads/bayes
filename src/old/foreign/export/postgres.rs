use std::os::raw::c_char;
use std::ffi::CStr;
use crate::model::{self, *};
use crate::distr::*;
use pgserver::datum::{self, Text, Bytea};
use std::convert::TryInto;
use std::panic::catch_unwind;
use pgserver::log;

/// Parses the distribution from text, assuming its validity.
fn parse_distr(distr_txt : &Text) -> Model {
    let distr_s : &str = distr_txt.as_ref();
    let model : Model;
    if let Ok(model) = distr_s.parse() {
        model
    } else {
        log::Error::raise("Unable to parse distribution")
    }
}

#[no_mangle]
pub extern "C" fn log_prob(distr_txt : Text) -> f64 {
    let model = parse_distr(&distr_txt);
    let distr : &Distribution = (&model).into();
    if let Some(obs) = distr.observations() {
        distr.log_prob(obs.into(), None)
    } else {
        log::Error::raise("Distribution do not have any observations")
    }
}

#[no_mangle]
pub extern "C" fn sample_from(distr_txt : Text) -> Text {
    let mat_string = model::parse::sample_to_string(distr_txt.as_ref());
    let txt = Text::from(&mat_string[..]);
    txt
}

#[test]
fn test_sample() {
    println!("{}", model::parse::sample_to_string("{ \"prop\" : 0.5, \"n\" : 10 }"));
}
