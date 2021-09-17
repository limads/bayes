use std::os::raw::c_char;
use std::ffi::CStr;
use crate::model::*;
use crate::prob::*;
use std::convert::TryInto;
use std::panic::catch_unwind;

/// Functions exported to C, and which can be used by compiling this library as a dynamic library.
/// The functions here are also used by the C++ modules statically compiled with this library,
/// living at foreign/gcem/gcem.cpp and foreign/mcmc/mcmc.cpp.
pub mod clang;

// PostgreSQL server-side functions. WIP.
// #[cfg(feature="pgext")]
// pub mod postgres;

