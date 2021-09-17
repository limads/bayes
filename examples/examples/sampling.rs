extern crate nalgebra;
extern crate bayes;
extern crate rand_distr;
extern crate rand;

use bayes::prob::{Normal, Prior};
use std::default::Default;
use rand_distr::Distribution;

fn main() {
    let n = Normal::prior(1.0, Some(1.0));
    for _ in 0..10 {
        println!("{:?}", n.sample(&mut rand::thread_rng()));
    }
}


