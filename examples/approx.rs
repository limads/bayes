use bayes::prob::*;
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

// select eta::real, log_prob::real, score::real from bern;
fn bernoulli_log_prob() {
    println!("theta,eta,log_prob,score");
    let y = DVector::from_column_slice(&[1., 0., 1., 1., 0., 1., 1., 0., 1., 1.]);
    let mut p = Bernoulli::new(10, None);
    for i in 1..1000 {
        let eta = DVector::from_element(10, -5. + (i as f64)*0.01);
        p.set_parameter((&eta).into(), true);
        let lp = p.log_prob(y.slice((0, 0), (10, 1)), None);
        let score = p.grad(y.slice((0, 0), (10, 1)), None);
        println!("{},{},{},{}", p.view_parameter(false)[0], eta[0], lp, score[0]);
    }
}

// View distribution as a function of the natural parameter
// select eta::real, log_prob::real, score::real from poiss;
// select * from poiss
//    order by abs(score::real);
fn poisson_log_prob() {
    println!("eta,log_prob,score");
    let y = DVector::from_column_slice(&[1., 2., 3., 1., 2.]);
    let mut p = Poisson::new(5, None);
    for i in 1..1000 {
        let eta = DVector::from_element(5, -2.3 + i as f64 * 0.01);
        p.set_parameter((&eta).into(), true);
        let lp = p.log_prob(y.slice((0, 0), (5, 1)), None);
        let score = p.grad(y.slice((0, 0), (5, 1)), None);
        println!("{},{},{}", eta[0], lp, score[0]);
    }
}

// View distribution as a function of the natural parameter
// select mu::real, scaled_mu::real, log_prob::real, score::real from norm;
// select mu::real, scaled_mu::real, log_prob::real, score::real from norm order by log_prob desc;
fn normal_log_prob() {
    println!("mu,scaled_mu,log_prob,score");
    let mut n = Normal::new(1000, Some(1.0), Some(2.0));
    let y = n.sample();
    // println!("{}", y);
    // let y = DVector::from_column_slice(&[1.1, 2.2, 3.2, 1.3, 2.1, 2.2, 3.1, 2.2, 1.1, 1.0]);
    let mle = Normal::mle(y.slice((0, 0), (1000, 1))).unwrap();
    // println!("{}", mle);
    // println!("MLE: {}", mle);
    // let var = mle.var()[0];
    // let var = y.map(|y| (y - mle.mean()[0]).powf(2.) / 100. ).sum();
    n.set_var(mle.var()[0]);
    let prec = 1. / n.var()[0];
    for i in 1..1000 {
        let mu = DVector::from_element(1000, -25.0 + i as f64 * 0.05);
        n.set_parameter((&mu).into(), true);
        let lp = n.log_prob(y.slice((0, 0), (1000, 1)), None);
        let score = n.grad(y.slice((0, 0), (1000, 1)), None);
        println!("{},{},{},{}", mu[0], prec*mu[0], lp, score[0]);
    }
}

// cargo run --example approx > /home/diego/Downloads/norm.csv
fn main() {
    normal_log_prob();
}


