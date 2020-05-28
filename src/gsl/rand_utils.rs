use super::rng::*;
use super::randist::*;

pub struct GslRng {
    rng : *mut gsl_rng
}

impl GslRng {

    pub fn new() -> Self {
        unsafe {
            gsl_rng_env_setup();
            let rng_type : *const gsl_rng_type = gsl_rng_default;
            let rng : *mut gsl_rng = gsl_rng_alloc(rng_type);
            Self{ rng }
        }
    }

    pub fn get(&self) -> *mut gsl_rng {
        unsafe { self.rng  }
    }

}

impl Drop for GslRng {

    fn drop(&mut self) {
        unsafe {
            gsl_rng_free (self.rng);
        }
    }
}

