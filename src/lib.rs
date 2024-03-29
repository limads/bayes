#![allow(warnings)]
#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/bayes/master/assets/bayes-logo.png")]

#[abiparse::export_abi]
mod c {

    use crate::prob::*;

    #[no_mangle]
    pub extern "C" fn histogram_closest_median_mode_partition(
        probs : &[f64],
        firsts : &mut [i64],
        modes : &mut [i64],
        lasts : &mut [i64],
        n_ret_modes : &mut i64,
        min_mode : i64,
        max_mode : i64,
        min_range : i64,
        max_range : i64,
        min_prob : f64,
        max_med_mode_dist : i64
    ) -> i64 {
        let partitions = crate::approx::closest_median_mode_partition(
            probs,
            min_mode as usize,
            max_mode as usize,
            min_range as usize,
            max_range as usize,
            min_prob,
            max_med_mode_dist as usize
        );
        println!("Returned {} partitions", partitions.len());
        *n_ret_modes = partitions.len() as i64;
        for (i, p) in partitions.iter().enumerate() {
            firsts[i] = p.first as i64;
            modes[i] = p.mode as i64;
            lasts[i] = p.last as i64;
        }
        0
    }
    
    #[no_mangle]
    pub extern "C" fn histogram_hdis(
        probs : &[f64],
        min_interval : i64,
        step_increment : i64,
        min_global_prob : f64,
        min_mode_ratio : f64,
        min_modes : i64,
        max_modes : i64,
        firsts : &mut [i64],
        modes : &mut [i64],
        lasts : &mut [i64],
        n_ret_modes : &mut i64,
        recursive : bool
    ) -> i64 {
        let crit = crate::approx::DensityCriteria::SmallestArea { 
            min_modes : min_modes as usize,
            max_modes : max_modes as usize
        };
        let partitions = crate::approx::highest_density_partition(
            probs, 
            min_interval as usize, 
            step_increment as usize,
            min_global_prob,
            min_mode_ratio,
            crit,
            recursive
        );
        *n_ret_modes = partitions.len() as i64;
        for (i, p) in partitions.iter().enumerate() {
            firsts[i] = p.first as i64;
            modes[i] = p.mode as i64;
            lasts[i] = p.last as i64;
        }
        0
     }
    
    #[no_mangle]
    pub extern "C" fn histogram_partition(
        probs : &[f64],
        firsts : &mut [i64],
        modes : &mut [i64],
        lasts : &mut [i64],
        n_ret_modes : &mut i64,
        min_mode : i64,
        max_mode : i64,
        min_range : i64,
        max_range : i64
    ) -> i64 {
        let partitions = crate::approx::min_partial_entropy_partition(
            probs,
            min_mode as usize,
            max_mode as usize,
            min_range as usize,
            max_range as usize
        );
        println!("Returned {} partitions", partitions.len());
        *n_ret_modes = partitions.len() as i64;
        for (i, p) in partitions.iter().enumerate() {
            firsts[i] = p.first as i64;
            modes[i] = p.mode as i64;
            lasts[i] = p.last as i64;
        }
        0
    }

    #[no_mangle]
    pub extern "C" fn binned_histogram(
        data : &[f64],
        min : &mut f64,
        max : &mut f64,
        bin_width : &mut f64,
        dst : &mut [f64],
        is_rel : bool
    ) -> i64 {
        use crate::approx::Histogram;
        let hist = crate::approx::SparseCountHistogram::calculate(data.iter(), dst.len());
        for i in 0..dst.len() {
            if is_rel {
                dst[i] = hist.bin(i).unwrap().prop as f64;
            } else {
                dst[i] = hist.bin(i).unwrap().count as f64;
            }
        }
        let (h_min, h_max) = hist.limits();
        *min = h_min;
        *max = h_max;
        *bin_width = hist.interval();
        0
    }

    #[no_mangle]
    pub extern "C" fn error(code : i64, out : &mut [u8]) -> i64 {
        if code < 0 {
            return -1;
        }
        if let Some(msg) = ERRORS.get(code as usize) {
            if msg.len() <= out.len() {
                out[..msg.len()].copy_from_slice(&msg.as_bytes());
                0
            } else {
                -1
            }
        } else {
            -1
        }
    }

    pub static ERRORS : [&'static str; 2] = [
        "No error\0",
        "Invalid distribution code\0"
    ];

    #[no_mangle]
    pub static ERR_INVALID_DISTR : i64 = 1;

    #[no_mangle]
    pub static NORMAL : i64 = 0;

    #[no_mangle]
    pub static BINOMIAL : i64 = 1;

    #[no_mangle]
    pub static POISON : i64 = 2;

    #[no_mangle]
    pub static GAMMA : i64  = 3;

    #[no_mangle]
    pub extern "C" fn random_sample(distr : i64, params : &[f64], dst : &mut [f64]) -> i64 {
        if distr == NORMAL {
            let mut n = Normal::new(params[0], params[1]);
            for dst in dst.iter_mut() {
                *dst = n.sample_with_default();
            }
        } else if distr == BINOMIAL {

        } else if distr ==  POISON {

        } else if distr == GAMMA {

        } else {
            return ERR_INVALID_DISTR;
        }
        0
    }

}

pub mod prob;

pub mod fit;

pub mod approx;

pub mod calc;

// pub mod ffi;

// use nalgebra::DVector;

/*
pub struct LocationExponential {
    loc : f64,
    obs : f64
}

pub struct ScaledExponential {
    loc : f64,
    scale : f64,
    obs : f64
} */

/*/// Represents an ordered sequence of distribution realizations, that are independent when they have no
/// parent factors, and when they do they are conditionally-independent given the parent factor.
/// The Joint<D> is used to represent convergent graphs without reference-counting.
/// While having many independent realizations sharing Rc<RefCell<ParentFactor>> is one way to do it,
/// we miss out on Rust static memory safety guarantees and the possibility to vectorize calculations.
/// Having impl Condition<F> for Joint<D> lets us represent a generic directed graph with as tree, since
/// a convergent many-to-one relationship is collapsed into a one-to-one relationship. Conceptually,
/// from the user point-of-view and in the exported API, we still have a generic DAG,
/// but in terms of the data structure, it is represented as a tree, with a single top-level likelihood
/// node and the branches diverging to form the scale and location factors. Rust static memory guarantees
/// hold only for tree-like data structures, not generic DAGs.
pub struct Join<D>
where
    D : Distribution
{

    obs : DVector<f64>,

    loc : DVector<f64>,

    scale : DVector<f64>,

    d : PhantomData<D>

}*/

// impl From<[D]> for Joint<D>
// impl FromIterator<D> for Joint<D>

/*// Implemented only for univariate distributions. May also be called Univariate.
// If Distribution does not have a scale parameter, scale(&self) always return 1.0.
pub trait Exponential {

    type Location;

    type Scale;

    fn location(&self) -> Self::Location;

    fn scale(&self) -> Option<Self::Scale>;

}*/

/*enum Factor {

    Fixed(DVector<f64>, )
}

use std::io;
use std::fmt;
use std::fs;
use either::Either;

#[test]
fn condition() {

    use crate::prob::Normal;
    use crate::fit::Likelihood;

    let a : Box<[Normal]> = [0.0].iter()
        .map(|v| Normal::likelihood([v]) )
        .collect();
}*/

// extern "C" fn avg()

/*

// Represent a stochastic process. Each realization of a process yields a univariate or joint distribution,
// which in turn can be sampled.
// Some processes are markov (depend only on the previous realization). Some can also be strict-sense
// stationary (parameters of realizations are exchangeable, such as white noise). Or wide-sense stationary
// (parameters are not exchangeable, but are dependent only on the distance between samples, not on distance
// between samples and origin).
pub trait Stochastic {

    pub fn realize(&mut self) -> &Joint<Normal>;

}

pub struct Process<D> {

    distr : D

}

impl Stochastic for Process<Normal>

impl Stochastic for Process<Joint<Normal>>;

impl Stochastic for Process<Dirichlet>

*/

use nalgebra::*;

pub fn homoscedastic(n : usize, var : f64) -> DMatrix<f64> {
    let mut c = DMatrix::zeros(n, n);
    c.fill_with_identity();
    c.scale_mut(var);
    c
}

// cargo test --lib -- rwmh --nocapture
#[test]
fn rwmh() {
    use statrs::distribution::*;
    use mcmclib::*;
    let init_vals = DVector::zeros(2);
    let lp = |vals : &DVector<f64>|->f64 {
        // let mvn = statrs::distribution::MultivariateNormal::new(vals.clone().data.into(), homoscedastic(2, 10.0).data.into()).unwrap();
        // mvn.ln_pdf(&vec![10.0, 10.0].into())
        let mvn = statrs::distribution::MultivariateNormal::new(vec![0.0, 0.0].into(), homoscedastic(2, 10.0).data.into()).unwrap();
        mvn.ln_pdf(&vals.data.as_vec().clone().into())
    };
    let out = rwmh_init(&init_vals, lp, &RWMHSettings::default()).unwrap();
    println!("{:?}", out.draws_out.row_mean());
}

#[macro_export]
macro_rules! save {
    ( $path:expr, $( $x:ident ),* ) => {
        let mut env = std::collections::BTreeMap::<String, serde_json::Value>::new();
        {
            $(
                env.insert(stringify!($x).to_string(), serde_json::to_value(&$x).unwrap());
            )*
        }
        std::fs::write($path, &serde_json::to_string_pretty(&env).unwrap()[..]);
    };
}

#[macro_export]
macro_rules! show {
    ( $( $x:ident ),* ) => {
        let mut env = std::collections::BTreeMap::<String, serde_json::Value>::new();
        {
            $(
                env.insert(stringify!($x).to_string(), serde_json::to_value(&$x).unwrap());
            )*
        }
        print!("{}", serde_json::to_string_pretty(&env).unwrap());
    };
}

