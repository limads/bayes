// runner examples/mixture.rs -s

extern crate nalgebra;
extern crate bayes;
extern crate rand_distr;
extern crate rand;

use bayes::prob::{Normal, MultiNormal};
use bayes::fit::Marginal;
use std::default::Default;
use rand_distr::Distribution;

fn main() -> Result<(), String> {
    let y = [0.1, 0.2, 0.3, 10.1, 10.2, 10.3];

    // Out observations are conditional on two unknown categorical realizations.
    // Calling marginal cache the y values instead of calculating the MLE.
    let mut n = Normal::marginal(&y, 2);
    n.condition(Categorical::new(0.1, 0.2));

    let cat : &Categorical = n.fit()?;
    let probs : [f64; 3] = cat.location();

    println!("{:?}", beta);
    for i in 0..3 {
        let p = probs[i];

        // Any calls to sample, location(.) and scale(.) dispatch to the ith sister marginal factor.
        n.set_marginal(i);

        let s : f64 = n.sample(&mut rand::thread_rng());
        println!("Probability = {:?}; Average at () = {:?}", p, s);
    }
    Ok(())
}


