use std::env;
use std::fs;
use cc;
use std::path::Path;

// Use export so the libs will be available to any executable, such as test (not only the root crate).
// export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/diego/Software/bayes/lib/mcmc:/home/diego/Software/bayes/lib/mcmc/mcmclib

// git clone https://github.com/kthohr/gcem.git
// git clone https://github.com/kthohr/mcmc

fn link_gsl() {
    println!("cargo:rustc-link-lib=gsl");
    println!("cargo:rustc-link-lib=gslcblas");
    println!("cargo:rustc-link-lib=m");
}

fn compile_mcmclib() {
    cc::Build::new()
        .cpp(true)
        .warnings(false)
        .extra_warnings(false)
        .includes(&["/usr/include", "target/mcmc/include"])
        .flag("-O3")
        .flag("-march=native") 
        .flag("-ffp-contract=fast")
        .file("target/mcmc/src/aees.cpp")
        .file("target/mcmc/src/de.cpp")
        .file("target/mcmc/src/hmc.cpp")
        .file("target/mcmc/src/mala.cpp")
        .file("target/mcmc/src/rmhmc.cpp")
        .file("target/mcmc/src/rwmh.cpp")
        .file("src/foreign/mcmc/mcmc.cpp")
        .cpp_link_stdlib("stdc++")
        .compile("libmcmc.a");
}

fn compile_gcem() {
    cc::Build::new()
        .cpp(true)
        .warnings(false)
        .extra_warnings(false)
        .includes(&["/usr/include", "target/gcem/include"])
        .flag("-O3")
        .flag("-march=native") 
        .flag("-ffp-contract=fast")
        .file("src/foreign/gcem/gcem.cpp")
        .cpp_link_stdlib("stdc++")
        .compile("libmcmc.a");
}

/*fn try_link_lib(build_folder : &Path, lib : &str) -> bool {
    if let Ok(entries) = fs::read_dir(build_folder) {
        for entry in entries {
            if entry.starts_with("bayes-") {
                if let Ok(inner_entries) = fs::read_dir(entry) {
                    for inner_entry in inner_entries {
                        if inner_entry.filename() == "out" {
                            if let Ok(folder_entries) = fs::read_dir(inner_entry) {
                                for folder_entry in folder_entries {
                                    let is_entry = folder_entry.extension() == Some("a") && 
                                        folder_entry.filename() == &format!("lib{}", lib)[..];
                                    if is_entry {
                                        println!("cargo:rustc-link-search={}", folder_entry.display());
                                        println!("cargo:rustc-link-lib={}", lib);
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    false
}*/

fn main() {

    println!("cargo:rerun-if-changed=src/foreign/gcem/gcem.cpp");
    println!("cargo:rerun-if-changed=src/foreign/mcmc/mcmc.cpp");
    println!("cargo:rerun-if-changed=src/foreign/stats/stats.cpp");
    
    link_gsl();
    compile_mcmclib();
    compile_gcem();
        
    /*let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_folder = Path::new(&dir).join("target").join("debug").join("build");
    if !try_link_lib(build_folder, "mcmc") {
    }
    if !try_link_lib(build_folder, "gcem") { 
    }*/
    
    // println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").display());
    // println!("cargo:rustc-link-lib=capi");

    // println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").join("mcmc").display());
    // println!("cargo:rustc-link-lib=mcmcwrapper");

    // println!("cargo:rustc-link-search=native={}", Path::new(&dir).join("lib").join("mcmc").join("mcmclib").display());
    // println!("cargo:rustc-link-lib=mcmc");
    
}



