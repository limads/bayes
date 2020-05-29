use nalgebra::*;
use bayes::gsl::randist::*;
use bayes::gsl::vector_double::*;
use bayes::gsl::matrix_double::*;
use bayes::gsl::utils::*;
use bayes::distr::*;

const EPS : f64 = 10E-8;

#[test]
fn bernoulli() {
    for y in [0.0, 1.0].iter() {
        for p in (1..100).map(|p| 0.01 * p as f64) {
            let gsl_p = unsafe{ gsl_ran_bernoulli_pdf(*y as u32, p) };
            let mut b = Bernoulli::new(1, Some(0.5));
            let pv = DVector::from_element(1,p);
            b.set_parameter(pv.rows(0,1), false);
            let ym = DMatrix::from_element(1,1,*y);
            assert!( (gsl_p - b.prob(ym.rows(0,1))).abs() < EPS);
        }
    }
}

#[test]
fn poisson() {
    let p = Poisson::new(1, Some(1.));
    let ym = DMatrix::from_element(1,1,1.0);
    unsafe {
        assert!( (gsl_ran_poisson_pdf(1, 1.0) -  p.prob(ym.rows(0,1))).abs() < EPS);
    }
}

#[test]
fn beta() {
    let beta = Beta::new(1, 1);
    let ym = DMatrix::from_element(1 ,1 ,0.5);
    unsafe {
        assert!( (gsl_ran_beta_pdf(0.5, 1., 1.) -  beta.prob(ym.rows(0,1))).abs() < EPS);
    }
}

#[test]
fn gamma() {
    let gamma = Gamma::new(1., 1.);
    let ym = DMatrix::from_element(1 ,1 , 1.0);
    unsafe {
        assert!( (gsl_ran_gamma_pdf(1.0, 1., 1.) -  gamma.prob(ym.rows(0,1))).abs() < EPS);
    }
}

#[test]
fn multinormal() {
    let mu = DVector::from_element(5, 0.0);
    let mut sigma = DMatrix::from_element(5, 5, 0.0);
    sigma.set_diagonal(&DVector::from_element(5, 1.));
    let mn : MultiNormal = MultiNormal::new(mu.clone(), sigma.clone());
    let lu = Cholesky::new(sigma).unwrap();
    let lower = lu.l();
    println!("{}", lower);
    unsafe {
        let lower_gsl : gsl_matrix = lower.into();
        //let mu : [f64; 5] = [0., 0., 0., 0., 0.];
        let ws_orig = DVector::<f64>::zeros(5);
        let mut ws : gsl_vector = ws_orig.into();
        let mut gsl_prob : f64 = 0.0;

        let mu_vec : gsl_vector = mu.clone().into();
        let x_vec : gsl_vector = mu.clone().into();
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
        let x = DMatrix::<f64>::from_iterator(1, 5, mu.iter().map(|x| *x));
        println!("Prob: {}", gsl_prob);
        assert!((gsl_prob - mn.prob(x.slice((0,0), (x.nrows(), x.ncols())))).abs() < EPS);
    }
}

#[test]
fn normal() {
    let n = Normal::new(1, Some(0.), Some(1.));
    let ym = DMatrix::from_element(1, 1, 0.0);
    unsafe {
        assert!( (gsl_ran_gaussian_pdf(0., 1.) - n.prob(ym.rows(0,1))).abs()  < EPS );
    }
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
        let cat_p = c.prob(vprobs.slice((0, 0), (1, 4)));
        println!("Categorical: {} {}", gsl_cat_p, cat_p);
        //println!("{}", gsl_cat_p - cat_p);
        assert!((gsl_cat_p - cat_p).abs() < EPS);
    }
}



