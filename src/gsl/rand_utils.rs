use super::rng::*;
use super::randist::*;

#[derive(Clone, Debug)]
pub struct GslRng {
    rng : *mut gsl_rng
}

impl GslRng {

    pub fn new(seed : u64) -> Self {
        unsafe {
            gsl_rng_env_setup();
            //let rng_type : *const gsl_rng_type = gsl_rng_default;
            let rng_type : *const gsl_rng_type = gsl_rng_rand;
            let mut rng : *mut gsl_rng = gsl_rng_alloc(rng_type);
            gsl_rng_set(rng, seed);
            Self{ rng }
        }
    }

    pub fn get(&self) -> *mut gsl_rng {
        unsafe { self.rng  }
    }

    pub fn normal(&self, mu : f64, sigma : f64) -> f64 {
        unsafe {
            mu + gsl_ran_gaussian(self.rng, sigma)
        }
    }

}

impl Drop for GslRng {

    fn drop(&mut self) {
        unsafe {
            // This is triggering a double free at test norm_approx
            // gsl_rng_free (self.rng);
        }
    }
}

