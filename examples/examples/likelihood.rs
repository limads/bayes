// runner examples/likelihood.rs -s

extern crate nalgebra;
extern crate bayes;
extern crate rand_distr;
extern crate rand;

use bayes::prob::Normal;
use bayes::fit::Likelihood;
use std::default::Default;
use rand_distr::Distribution;

/// Exemplifies maximum likelihood estimation for univariate and multivariate distributions.
fn main() {
    let n = Normal::likelihood(&[1.0, 2.0, 1.1, 2.0]);
    println!("{:?}", n);
    for _ in 0..10 {
        println!("{:?}", n.sample(&mut rand::thread_rng()));
    }
}

