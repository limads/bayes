use std::env;
use std::fs;
use cc;
use std::path::Path;

// Use export so the libs will be available to any executable, such as test (not only the root crate).
// export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/diego/Software/bayes/lib/mcmc:/home/diego/Software/bayes/lib/mcmc/mcmclib

fn main() {
    println!("cargo:rustc-link-lib=gsl");
    println!("cargo:rustc-link-lib=gslcblas");
    println!("cargo:rustc-link-lib=m");
    if let Ok(_) = env::var("CARGO_FEATURE_MKL") {
        println!("cargo:rustc-link-lib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=mkl_intel_thread");
        println!("cargo:rustc-link-lib=mkl_core");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=iomp5");
    }
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").display());
    // println!("cargo:rustc-link-lib=capi");

    println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").join("mcmc").display());
    println!("cargo:rustc-link-lib=mcmcwrapper");

    println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").join("mcmc").join("mcmclib").display());
    println!("cargo:rustc-link-lib=mcmc");
}



