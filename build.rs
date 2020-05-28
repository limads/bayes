/*
Also, use links="<lib>" on Cargo.toml
*/

// -c gnu_c -libs

// use std::process::Command;

fn main() {
    /*println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=mkl_core");
    println!("cargo:rustc-link-lib=mkl_intel_ilp64");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=mkl_gnu_thread");
    println!("cargo:rustc-link-lib=mkl_mc");
    println!("cargo:rustc-link-lib=mkl_rt");
    println!("cargo:rustc-link-lib=mkl_avx");
    println!("cargo:rustc-link-lib=mkl_def");
    println!("cargo:rustc-link-lib=mkl_vml_mc");
    println!("cargo:rustc-link-lib=mkl_def");
    println!("cargo:rustc-link-lib=mkl_scalapack_ilp64");
    println!("cargo:rustc-link-lib=mkl_cdft_core");*/
    //println!("cargo:rustc-link-lib=mkl_tbb_thread");
    //println!("cargo:rustc-link-lib=lkl_blacs_intelmpi_ilp6");
    //println!("cargo:rustc-link-lib=tbb");
    // mkl_link_tool -c gnu_c -libs

    //let mkl_link_tool_err = "mkl_link_tool failed to execute. \
    //    Check if utility is installed an living on $PATH.";
    //let _output = Command::new("mkl_link_tool")
    //    .args(&["-c", "gnu_c", "-libs"])
    //    .output()
    //    .expect(mkl_link_tool_err);
    // let cmd_out = std::str::from_utf8(&output.stdout).unwrap();
    // let cmd_split : Vec<&str> = cmd_out.rsplit(' ').collect();
    // let libs : Vec<&&str> = cmd_split.iter().filter(|c| (*c).find(&"-l") != None).collect();
    // println!("Intel MKL output");
    // println!("{:?}", libs);

    println!("cargo:rustc-link-lib=mkl_intel_lp64");
    println!("cargo:rustc-link-lib=mkl_intel_thread");
    println!("cargo:rustc-link-lib=mkl_core");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=iomp5");
    println!("cargo:rustc-link-lib=gsl");
    println!("cargo:rustc-link-lib=gslcblas");
}


