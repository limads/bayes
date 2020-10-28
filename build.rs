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

    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").display());
    // println!("cargo:rustc-link-lib=capi");

    println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").join("mcmc").display());
    println!("cargo:rustc-link-lib=mcmcwrapper");

    println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").join("mcmc").join("mcmclib").display());
    println!("cargo:rustc-link-lib=mcmc");
}



