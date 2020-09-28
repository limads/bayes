use std::env;
use std::fs;

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
    let man_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_dir = format!("{}/{}", man_dir, "target/c");
    let can_dir = fs::canonicalize(&lib_dir).unwrap().to_string_lossy().to_string();
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=static={}", "pg_helper");
}


