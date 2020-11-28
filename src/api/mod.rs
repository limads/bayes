use std::os::raw::c_char;
use std::ffi::CStr;
use crate::model::*;
use crate::prob::*;
// use pgdatum::{self, Text, Bytea};
use std::convert::TryInto;
use std::panic::catch_unwind;
// use pgdatum::log;

pub mod c_api;

#[cfg(feature="pgext")]
pub mod postgres;

