use bayes::distr::*;
use nalgebra::*;

// Usage:
// cargo run --example approx > [dst].csv

// View distribution as a function of the data sample
// select y::real, log_prob::real, prob::real, score::real from poiss;
fn prob_sample() {
    // Distributions have a fixed sample size associated with them when they are created.
    // To observe a single log-probability value, create a distribution that relates to
    // samples of size one.
    // The log-probability might be linear as a function of the sample but convex as a function of the parameter.
    let p = Poisson::new(1, Some(5.));
    let y = DVector::from_fn(100, |r, _| r as f64);
    println!("y,log_prob,prob,score");
    for i in 0..100 {
        let lp = p.log_prob(y.slice((i, 0), (1, 1)), None);
        let prob = p.prob(y.slice((i, 0), (1, 1)), None);
        let score = p.grad(y.slice((i, 0), (1, 1)), None);
        println!("{},{},{},{}", y[i], lp, prob, score[0]);
    }
}

// View distribution as a function of the natural parameter
// select eta::real, log_prob::real, score::real from poiss;
// select * from poiss
//    order by abs(score::real);
fn log_prob() {
    println!("eta,log_prob,score");
    let y = DVector::from_column_slice(&[1., 2., 3., 1., 2.]);
    let mut p = Poisson::new(5, None);
    for i in 1..1000 {
        let eta = DVector::from_element(5, -2.3 + i as f64 * 0.01);
        p.set_parameter((&eta).into(), true);
        let lp = p.log_prob(y.slice((0, 0), (5, 1)), None);
        // let prob = p.prob(y.slice((0, 0), (5, 1)), None);
        let score = p.grad(y.slice((0, 0), (5, 1)), None);
        println!("{},{},{}", eta[0], lp, score[0]);
    }
}

fn main() {
    log_prob();
}


