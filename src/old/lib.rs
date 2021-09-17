//! This create offers composable abstractions to build
//! probabilistic models and inference algorithms.
//!
//! The trait [Distribution](distr/trait.Distribution.html) offer the basic random sampling and
//! calculation of summary statistic functionality for the typical parametric
//! distributions. Implementors of this trait are located at the `distr` module.
//!
//! The trait [Estimator](distr/trait.Estimator.html) offer the `fit` method, which is implemented
//! by the distributions themselves (conjugate inference) and by generic estimation
//! algorithms. Two algorithms will be provided: [ExpectMax](optim/em/struct.ExpectMax.html)
//! (expectation maximization) which returns a gaussian approximation for each node
//! of a generic distribution graph; and [Metropolis](sim/metropolis/struct.Metropolis.html)
//! (Metropolis-Hastings posterior sampler) which returns
//! a non-parametric marginal histogram for each node.

// #![feature(vec_into_raw_parts)]
// #![feature(extern_types)]
// #![feature(is_sorted)]
// #![feature(min_const_generics)] (Perhaps implement MultiNormal<N : usize>)

#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/bayes/master/assets/bayes-logo.png")]

/// Probability distributions used to build models.
pub mod prob;

//pub mod basis;

// Auto-generated bindings to Intel MKL (mostly for basis transformation).
// mod mkl;

// Data structures and generic traits to load and save data into/from dynamically-allocated matrices.
pub mod sample;

/// Feature extraction traits, structures and algorithms.
pub mod feature;

// Probability models defined at runtime.
pub mod model;

/// Estimation algorithms
pub mod fit;

// Foreign source code to interface with MKL, GSL and mcmclib.
mod foreign;

// Utilities to parse and express probabilistic models graphically.
// pub mod graph;

/// Mathematical functions useful for probability-related calculations
pub mod calc;

/// Non-parametric distribution representations
pub mod approx;

// C API to be used from other environments.
// cbindgen src/lib.rs -o include/bayes.h
mod export {

    use crate::prob::*;
    use std::ffi;
    use std::mem;

    // type DynDistr = *const (dyn Distribution + Predictive);
    // type DynDistrMut = *mut (dyn Distribution + Predictive);

    #[no_mangle]
    pub extern "C" fn normal_prior(mean : f64, var : f64) -> *mut Box<dyn Predictive<f64>> {
        // let ptr = Box::into_raw(Box::new(Normal::prior((mean, var))));
        // println!("Exported pointer : {:?}", ptr);
        // ptr
        // Box::into_raw(box_twice(Normal::prior((mean, var))))
        // Box::into_raw(Box::new(Box::new(Normal::prior((mean, var)))))
        Normal::prior((mean, var)).export()
    }

    // TODO build crate exports and import with "use exports::Export;"
    pub trait Export<D : ?Sized>
    where
        Self : Sized
    {

        /*fn thin(self) -> Box<Box<D>>
        where
            Box<D> : From<Box<Self>>
        {
            Box::new(Box::new(self).into())
        }*/

        fn export(self) -> *mut Box<D> {
            Box::into_raw(Box::new(Self::pack(self)))
        }

        // User-defined function used to pack a type into Box<T>. Must
        // be implemented as Box::new(self) by the user always.
        fn pack(self) -> Box<D>;

        /*fn receive(ptr : *mut Box<D>) -> Box<D> {
            unsafe { *Box::from_raw(ptr) }
        }*/

    }

    pub unsafe trait Receive<D>
    where
        D : ?Sized
    {

        fn assume_ref<'a>(self) -> &'a D;

        fn receive(self) -> Box<D>;

    }

    unsafe impl<D> Receive<D> for *mut Box<D>
    where
        D : ?Sized
    {

        /// Panics if pointer is null.
        fn assume_ref<'a>(self) -> &'a D {
            unsafe { self.as_ref().unwrap() }
        }

        // To be called from data at the FFI. Unwraps a *mut Box<D> into
        // a Box<D> to be dropped at the end of the current call.
        fn receive(self) -> Box<D> {
            unsafe { *Box::from_raw(self) }
        }
    }

    impl Export<dyn Distribution> for Normal {
        fn pack(self) -> Box<dyn Distribution> {
            Box::new(self)
        }
    }

    impl Export<dyn Predictive<f64>> for Normal {
        fn pack(self) -> Box<dyn Predictive<f64>> {
            Box::new(self)
        }
    }

    #[no_mangle]
    pub extern "C" fn export_slice() -> Box<[f64]> {
        vec![1.0, 2.0, 3.0].into()
    }

    #[no_mangle]
    pub extern "C" fn with_capacity_double(cap : u64) -> Vec<f64> {
        Vec::with_capacity(cap as usize)
    }

    #[no_mangle]
    pub extern "C" fn push_double(vec : *mut Vec<f64>, d : f64) {
        unsafe { (*vec).push(d); }
    }

    #[no_mangle]
    pub extern "C" fn free_double_vec(vec : Vec<f64>) {
        mem::drop(vec);
    }

    #[no_mangle]
    pub extern "C" fn free_double_box(bx : Box<f64>) {
        mem::drop(bx);
    }

    #[no_mangle]
    pub extern "C" fn into_double_boxed(bx : Vec<f64>) -> Box<[f64]> {
        bx.into_boxed_slice()
    }

    // None, integers, bytes objects and (unicode) strings are the only native Python objects
    // that can directly be used as parameters in function calls
    // If pointer is to be de-allocated, use Option<Box<dyn Trait>> as argument.

    // *mut ffi::c_void*/
    #[no_mangle]
    pub extern "C" fn predict(distr : *mut Box<dyn Predictive<f64>>) -> f64 {
        unsafe { distr.assume_ref().predict() }
    }

    /* To deallocate, when receiving distr : *mut dyn [Something] : drop(Box::from_raw(distr)); */

}

