use nalgebra::*;
use bayes::gsl::randist::*;
use bayes::gsl::vector_double::*;
use bayes::gsl::matrix_double::*;
use bayes::gsl::utils::*;
use bayes::distr::*;
use bayes::inference::sim::Histogram;
use bayes::distr::multinormal;
// use bayes::io::{Sequence, Surface};
use rand;
//use crate::utils;

const EPS : f64 = 10E-8;

fn unit_interval_seq(n : usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, 1, |i,j| (i+1) as f64 * (1. / n as f64) )
}

fn count_seq(n : usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, 1, |i,j| (i+1) as f64)
}

#[test]
fn bernoulli() {
    for y in [0.0, 1.0].iter() {
        for p in (1..100).map(|p| 0.01 * p as f64) {
            let gsl_p = unsafe{ gsl_ran_bernoulli_pdf(*y as u32, p) };
            let mut b = Bernoulli::new(1, Some(0.5));
            let pv = DVector::from_element(1,p);
            b.set_parameter(pv.rows(0,1), false);
            let ym = DMatrix::from_element(1,1,*y);
            assert!( (gsl_p - b.prob(ym.rows(0,1), None)).abs() < EPS);
        }
    }
}

#[test]
fn poisson() {
    let lambda = unit_interval_seq(100);
    let count = count_seq(100);
    unsafe {
        for i in 0..100 {
            for l in lambda.iter() {
                let gsl_prob = gsl_ran_poisson_pdf(count[(i,0)] as u32, *l);
                let poiss = Poisson::new(1, Some(*l));
                let bayes_prob = poiss.prob(count.slice((i,0), (1,1)), None);
                assert!( (gsl_prob -  bayes_prob).abs() < EPS);
            }
        }
    }
}

#[test]
fn beta() {
    let theta = unit_interval_seq(100);
    let count = count_seq(10);
    unsafe {
        for i in 0..100 {
            for (a, b) in count.iter().zip(count.iter()) {
                let gsl_prob = gsl_ran_beta_pdf(theta[(i,0)], *a, *b);
                let beta = Beta::new(*a as usize, *b as usize);
                let bayes_prob = beta.prob(theta.slice((i,0), (1,1)), None);
                //println!("Gsl prob: {}", gsl_prob);
                //println!("Bayes prob: {}", bayes_prob);
                //println!("a: {}; b: {}; theta: {}", a, b, theta[(i,0)]);
                assert!( (gsl_prob -  bayes_prob).abs() < EPS);
            }
        }
    }
}

#[test]
fn gamma() {
    let theta = unit_interval_seq(100);
    let count = count_seq(10);
    unsafe {
        for i in 0..100 {
            for (a, b) in count.iter().zip(count.iter()) {
                // GSL follows the shape/scale parametrization;
                // bayes follows the shape/inv-scale parametrization
                let gsl_prob = gsl_ran_gamma_pdf(theta[(i,0)], *a, 1. / *b);
                let gamma = Gamma::new(*a, *b);
                let bayes_prob = gamma.prob(theta.slice((i,0), (1,1)), None);
                assert!( (gsl_prob -  bayes_prob).abs() < EPS);
            }
        }
    }
}

fn gsl_multinormal_pdf(x : &DVector<f64>, mu : &DVector<f64>, sigma : &DMatrix<f64>) -> f64 {
    let mut gsl_prob : f64 = 0.0;
    let lu = Cholesky::new(sigma.clone()).unwrap();
    let lower = lu.l();
    unsafe {
        let lower_gsl : gsl_matrix = lower.into();
        let ws_orig = DVector::<f64>::zeros(5);
        let mut ws : gsl_vector = ws_orig.into();
        let mu_vec : gsl_vector = mu.clone().into();
        let x_vec : gsl_vector = x.clone().into();
        let ans = gsl_ran_multivariate_gaussian_pdf(
            &x_vec as *const _,
            &mu_vec as *const _,
            &lower_gsl as *const _,
            &mut gsl_prob as *mut _,
            &mut ws as *mut _
        );
        if ans != 0 {
            panic!("Error calculating multivariate density");
        }
    }
    gsl_prob
}

#[test]
fn multinormal() {

    // We pass samples organized row-by-row to bayes; but as a vector to gsl (as long as we
    // evaluate only a single probability).
    let mut x = DMatrix::from_fn(1, 5, |_,_| rand::random() );
    let xt = x.row(0).clone_owned().transpose();

    let mut mu = DVector::from_fn(5, |_,_| rand::random() );

    /*let mut sigma = DMatrix::from_element(5, 5, 0.0);
    sigma.set_diagonal(&DVector::from_element(5, 2.));
    sigma.row_mut(0).copy_from_slice(&[1.0, 0.5, 0.0, 0.0, 0.0]);
    sigma.row_mut(1).copy_from_slice(&[0.5, 1.0, 0.0, 0.0, 0.0]);*/
    let sigma = multinormal::approx_pd(DMatrix::from_fn(5, 5, |_,_| rand::random() ) );
    println!("mu = {}", mu);
    println!("sigma = {}", sigma);
    println!("x = {}", x);

    let gsl_prob = gsl_multinormal_pdf(&xt, &mu, &sigma);
    let mn : MultiNormal = MultiNormal::new(mu.clone(), sigma).unwrap();
    let bayes_prob = mn.prob(x.slice((0,0), (x.nrows(), x.ncols())), None);
    println!("GSL Prob: {}", gsl_prob);
    println!("Bayes Prob: {}", bayes_prob);
    assert!((gsl_prob - bayes_prob).abs() < EPS);
}

#[test]
fn normal() {
    let mu = unit_interval_seq(100);
    let values = mu.clone();
    let vars = unit_interval_seq(100);
    unsafe {
        //for m in mu.iter() {
        for (i, (y, v)) in values.iter().zip(vars.iter()).enumerate() {
            // GSL parametrizes gaussians by the standard deviation;
            // bayes by the variance.
            let gsl_prob = gsl_ran_gaussian_pdf(*y - *v, (v).sqrt());
            let norm = Normal::new(1, Some(*v), Some(*v));
            let bayes_prob = norm.prob(values.slice((i,0), (1,1)), None);
            // println!("Gsl prob: {} (evaluated at {})", gsl_prob, *y);
            // println!("Bayes prob: {} (evaluated at {})", bayes_prob, *y);
            println!("mu: {}, var: {}, gsl prob: {}; bayes prob: {}", v, v, gsl_prob, bayes_prob);
            assert!( (gsl_prob -  bayes_prob).abs() < EPS);
        }
        //}
    }
    /*let n = Normal::new(1, Some(0.), Some(1.));
    let ym = DMatrix::from_element(1, 1, 0.0);
    unsafe {
        assert!( (gsl_ran_gaussian_pdf(0., 1.) - n.prob(ym.rows(0,1))).abs()  < EPS );
    }*/
}

#[test]
fn categorical() {
    // Obs: Categorical = Multinomial when n=1. Also note that bayes parametrizes
    // categorical with k-1 classes.
    let c : Categorical = Categorical::new(4, Some(&[0.2, 0.2, 0.2, 0.2]));
    let probs : [f64; 5] = [0.2,0.2,0.2,0.2,0.2];
    let outcome : [u32; 5] = [1, 0, 0, 0, 0];
    unsafe {
        let gsl_cat_p = gsl_ran_multinomial_pdf(5, &probs[0] as *const _, &outcome[0] as *const _);
        let vprobs = DMatrix::from_iterator(1, 4, outcome[0..4].iter().map(|o| *o as f64));
        let cat_p = c.prob(vprobs.slice((0, 0), (1, 4)), None);
        // println!("Categorical: {} {}", gsl_cat_p, cat_p);
        // println!("{}", gsl_cat_p - cat_p);
        assert!((gsl_cat_p - cat_p).abs() < EPS);
    }
}

#[test]
fn histogram() {
    // Generate histogram to reproduced sample uniform distribution
    let sample = DVector::from_iterator(100, (0..100).map(|s| s as f64));
    let hist = Histogram::build(&sample);
    println!("Mean = {}", hist.mean());
    println!("Median = {}", hist.median());
    println!("Variance = {}", hist.var());
    println!("Full = {:?}", hist.full(5, false));
    //assert!(hist.mean() == hist.median());
}

#[test]
fn gradients() {

    let p = Poisson::new(5, Some(0.5));
    let data = DVector::from_column_slice(&[1., 1., 2., 1., 2.]);
    println!("gradient = {}", p.grad((&data).into(), None) );
}

/*#[test]
fn sequence() {
    let mut s : Sequence<'_, f32> = Sequence::new(&[0, 1, 0]);
    println!("{}", s.extract());
    s.reposition(1,2);
    println!("{}", s.extract());
}

#[test]
fn surface() {
    let mut s : Surface<'_, f32> = Surface::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3);
    println!("{}", s.extract());
    s.reposition((2,1),(2,2));
    println!("{}", s.extract());
}*/
