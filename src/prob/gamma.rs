use crate::prob::*;
use special::*;

// Reference: stats::dgamma.ipp
fn gamma_log_prob(x : f64, shape : f64, scale : f64) -> f64 {
    -1.0*shape.ln() - shape*scale.ln() + (shape - 1.0)*x.ln() - x / scale
}

pub struct Gamma {

}

impl Univariate for Gamma {

}
