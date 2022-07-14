use crate::prob::*;
use special::*;

// based on stats::dpois.ipp
fn poisson_log_prob(x : u32, rate : f64) -> f64 {
    x as f64 * rate.ln() - rate - ((x+1) as f64).ln_gamma().0
}

pub struct Poisson(f64);

impl Univariate for Poisson {

}
