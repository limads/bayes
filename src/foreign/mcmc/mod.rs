use crate::foreign::export::clang::{DistrPtr, model_log_prob};

// Linking happens at build.rs file
extern "C" {

    pub fn distr_mcmc(
        init_vals : *const f64,
        out : *mut f64,
        n : usize,
        p : usize,
        burn : usize,
        distr : *mut DistrPtr
    ) -> bool;

}

