/* Credit to the "sample" module goes to Keith O'hara, which released the original
C++ implementation of mcmclib via Apache 2.0. Original source at https://github.com/kthohr/mcmc */

/* TODO
struct Bounds { lower : DVector<f64>, upper : DVector<f64>, ty : Vec<Bound> } then carry Option<Bounds>
*/

use nalgebra::*;
use std::f64::EPSILON;
use std::f64::NEG_INFINITY;
use std::f64::INFINITY;
use std::fmt::Debug;
use num_traits::Zero;

fn boxed_log_prob_kernel_with_grad<'a>(
    vals_bound : bool,
    bounds_type : &'a DVector<u32>,
    lower_bounds : &'a DVector<f64>,
    upper_bounds : &'a DVector<f64>,
    target_log_kernel : impl Fn(&DVector<f64>, Option<&mut DVector<f64>>)->f64 + 'a
) -> Box<dyn Fn(&DVector<f64>, Option<&mut DVector<f64>>)->f64 + 'a> {
    if vals_bound {
        Box::new(move |vals_inp : &DVector<f64>, grad : Option<&mut DVector<f64>>|->f64 {
            let vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            target_log_kernel(&vals_inv_trans, None) + log_jacobian(vals_inp,bounds_type, lower_bounds, upper_bounds)
        })
    } else {
        Box::new(move |vals_inp : &DVector<f64>, grad : Option<&mut DVector<f64>>|->f64 {
            target_log_kernel(&vals_inp, None)
        })
    }
}

fn boxed_log_prob_kernel<'a>(
    vals_bound : bool,
    bounds_type : &'a DVector<u32>,
    lower_bounds : &'a DVector<f64>,
    upper_bounds : &'a DVector<f64>,
    target_log_kernel : impl Fn(&DVector<f64>)->f64 + 'a
) -> Box<dyn Fn(&DVector<f64>)->f64 + 'a> {
    if vals_bound {
        Box::new(move |vals_inp : &DVector<f64>|->f64 {
            let vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            target_log_kernel(&vals_inv_trans) + log_jacobian(vals_inp,bounds_type, lower_bounds, upper_bounds)
        })
    } else {
        Box::new(target_log_kernel)
    }
}

mod mala;

mod de;

mod rwmh;

mod aees;

mod hmc;

mod rmhmc;

pub use rwmh::*;

pub use aees::*;

pub use de::*;

pub use hmc::*;

pub use mala::*;

pub use rmhmc::*;

/* To be used instead of tensors, but avoiding Vec<DMatrix> to minimize allocations.
Each matrix is stored by concatenating it horizontally with the previous one, resulting
in one wide matrix. Individual access semantics are provided by slice(.) and slice_mut(.) */
#[derive(Clone)]
pub struct Matrices<T>
where
    T : Scalar + Copy + Debug + Zero
{

    mats : DMatrix<T>,

    slice_ncol : usize

}

impl<T> Matrices<T>
where
    T : Scalar + Copy + Debug + Zero
{

    pub fn zeros(nrow : usize, ncol : usize, nslice : usize) -> Self {
        Self { mats : DMatrix::zeros(nrow, ncol*nslice), slice_ncol : ncol }
    }

    pub fn slice(&self, i : usize) -> DMatrixSlice<T> {
        self.mats.columns(self.slice_ncol*i, self.slice_ncol)
    }

    pub fn slice_mut(&mut self, i : usize) -> DMatrixSliceMut<T> {
        self.mats.columns_mut(self.slice_ncol*i, self.slice_ncol)
    }

}

// Original impl use a cube: nrow x ncol x nslice. We use a very wide matrix
// instead. To index the ith draw, use X_out[:,(i*K)..(i*K + K)]
fn get_ith_draw_mut(draws : &mut DMatrix<f64>, K : usize, i : usize) -> DMatrixSliceMut<f64> {
    draws.columns_mut(K*i, K)
}

fn get_ith_draw_ref(draws : &DMatrix<f64>, K : usize, i : usize) -> DMatrixSlice<f64> {
    draws.columns(K*i, K)
}

fn fill_with_std_normal<R : Dim, C : Dim, S : RawStorageMut<f64, R, C>>(v : &mut Matrix<f64, R, C, S>) {
    use rand::prelude::*;
    let mut rng = ThreadRng::default();
    v.iter_mut().for_each(|d| *d = rng.sample(rand_distr::StandardNormal) );
}

fn fill_with_uniform<R : Dim, C : Dim, S : RawStorageMut<f64, R, C>>(v : &mut Matrix<f64, R, C, S>) {
    use rand::prelude::*;
    v.iter_mut().for_each(|d| *d = rand::random() );
}

pub enum Bound {

    // 1
    Unbounded,

    // 2
    Lower,

    // 3
    Upper,

    // 4
    Both

}

/*
struct algo_settings_t
{
    // general
    bool vals_bound = false;

    arma::vec lower_bounds;
    arma::vec upper_bounds;

    // AEES
    int aees_n_draws = 1E04;
    int aees_n_initial_draws = 1E03;
    int aees_n_burnin = 1E03;

    arma::vec aees_temper_vec;
    double aees_prob_par = 0.10;
    int aees_n_rings = 5;

    // DE
    bool de_jumps = false;

    int de_n_pop = 100;
    int de_n_gen = 1000;
    int de_n_burnin = 1000;

    double de_par_b = 1E-04;
    double de_par_gamma = 1.0;
    double de_par_gamma_jump = 2.0;

    arma::vec de_initial_lb;
    arma::vec de_initial_ub;

    double de_accept_rate; // will be returned by the function

    // HMC
    int hmc_n_draws = 1E04;
    int hmc_n_burnin = 1E04;

    double hmc_step_size = 1.0;
    int hmc_leap_steps = 1; // number of leap frog steps
    arma::mat hmc_precond_mat;

    double hmc_accept_rate; // will be returned by the function

    // RM-HMC: HMC options + below
    int rmhmc_fp_steps = 5; // number of fixed point iteration steps

    // MALA
    int mala_n_draws = 1E04;
    int mala_n_burnin = 1E04;

    double mala_step_size = 1.0;
    arma::mat mala_precond_mat;

    double mala_accept_rate; // will be returned by the function

    // RWMH
    int rwmh_n_draws = 1E04;
    int rwmh_n_burnin = 1E04;

    double rwmh_par_scale = 1.0;
    arma::mat rwmh_cov_mat;

    double rwmh_accept_rate; // will be returned by the function
};
*/


// TODO return Vec<Bound> here.
fn determine_bounds_type(
    vals_bound : bool,
    n_vals : usize,
    lower_bounds : &DVector<f64>,
    upper_bounds : &DVector<f64>
) -> DVector<u32> {
    let mut ret_vec = DVector::from_element(n_vals, 1);
    if vals_bound {
        for i in 0..n_vals {
            if lower_bounds[i].is_finite() && upper_bounds[i].is_finite() {
                ret_vec[i] = 4;
            } else if lower_bounds[i].is_finite() && !upper_bounds[i].is_finite() {
                ret_vec[i] = 2;
            } else if !lower_bounds[i].is_finite() && upper_bounds[i].is_finite() {
                ret_vec[i] = 3;
            }
        }
    }
    ret_vec
}

fn transform(
    vals_inp : &DVector<f64>,
    bounds_type : &DVector<u32>,
    lower_bounds : &DVector<f64>,
    upper_bounds : &DVector<f64>
) -> DVector<f64> {
    let n_vals = bounds_type.nrows();
    let mut vals_trans_out = DVector::zeros(n_vals);
    for i in 0..n_vals {
        match bounds_type[i] {
            1 => { // no bounds
                vals_trans_out[i] = vals_inp[i];
            },
            2 => { // lower bound only
                vals_trans_out[i] = (vals_inp[i] - lower_bounds[i] + EPSILON).ln();
            },
            3 => { // upper bound only
                vals_trans_out[i] = -1.0*(upper_bounds[i] - vals_inp[i] + EPSILON).ln();
            },
            4 => { // upper and lower bounds
                vals_trans_out[i] = (vals_inp[i] - lower_bounds[i] + EPSILON).ln() - (upper_bounds[i] - vals_inp[i] + EPSILON).ln();
            },
            _ => panic!()
        }
    }
    vals_trans_out
}

fn inv_transform(
    vals_trans_inp : &DVector<f64>,
    bounds_type : &DVector<u32>,
    lower_bounds : &DVector<f64>,
    upper_bounds : &DVector<f64>
) -> DVector<f64> {
    let n_vals = bounds_type.nrows();
    let mut vals_out = DVector::zeros(n_vals);
    for i in 0..n_vals {
        match bounds_type[i] {
            1 => { // unbounded
                vals_out[i] = vals_trans_inp[i];
            },
            2 => { // lower bound
                if !vals_trans_inp[i].is_finite() {
                    vals_out[i] = lower_bounds[i] + EPSILON;
                } else {
                    vals_out[i] = lower_bounds[i] + EPSILON + vals_trans_inp[i].exp();
                }
            },
            3 => { // upper bound
                if !vals_trans_inp[i].is_finite() {
                    vals_out[i] = upper_bounds[i] - EPSILON;
                } else {
                    vals_out[i] = upper_bounds[i] - EPSILON - (-1.0*vals_trans_inp[i]).exp();
                }
            },
            4 => { // upper and lower bounds
                if !vals_trans_inp[i].is_finite() {
                    if vals_trans_inp[i].is_nan() {
                        vals_out[i] = (upper_bounds[i] - lower_bounds[i]) / 2.0;
                    } else if vals_trans_inp[i] < 0.0 {
                        vals_out[i] = lower_bounds[i] + EPSILON;
                    } else {
                        vals_out[i] = upper_bounds[i] - EPSILON;
                    }
                } else {
                    vals_out[i] =  (lower_bounds[i] + EPSILON + (upper_bounds[i] - EPSILON)*vals_trans_inp[i].exp() ) /
                                    ( 1.0 + vals_trans_inp[i].exp() );
                }
            },
            _ => panic!()
        }
        if vals_out[i].is_nan() {
            vals_out[i] = upper_bounds[i] - EPSILON;
        }
    }
    vals_out
}

fn log_jacobian(
    vals_trans_inp : &DVector<f64>,
    bounds_type : &DVector<u32>,
    lower_bounds : &DVector<f64>,
    upper_bounds : &DVector<f64>
) -> f64 {
    let n_vals = bounds_type.nrows();
    let mut ret_val = 0.0;
    for i in 0..n_vals {
        match bounds_type[i] {
            2 => { // lower bound only
                ret_val += vals_trans_inp[i];
            },
            3 => { // upper bound only
                ret_val += - vals_trans_inp[i];
            },
            4 => { // upper and lower bounds
                let exp_inp = vals_trans_inp[i].exp();
                if exp_inp.is_finite() {
                    ret_val += (upper_bounds[i] - lower_bounds[i]).ln() + vals_trans_inp[i] - 2.0 * (1.0 + exp_inp).ln();
                } else {
                    ret_val += (upper_bounds[i] - lower_bounds[i]).ln() - vals_trans_inp[i];
                }
            },
            _ => panic!()
        }
    }
    return ret_val;
}

fn inv_jacobian_adjust(
    vals_trans_inp : &DVector<f64>,
    bounds_type : &DVector<u32>,
    lower_bounds : &DVector<f64>,
    upper_bounds : &DVector<f64>
) -> DMatrix<f64> {
    let n_vals = bounds_type.nrows();
    let mut ret_mat = DMatrix::<f64>::identity(n_vals,n_vals);
    for i in 0..n_vals {
        match bounds_type[i] {
            2 => { // lower bound only
                ret_mat[(i,i)] = 1.0 / vals_trans_inp[i].exp();
            },
            3 => { // upper bound only
                ret_mat[(i,i)] = 1.0 / (-1.0*vals_trans_inp[i]).exp();
            },
            4 => { // upper and lower bounds
                let exp_inp = vals_trans_inp[i].exp();
                ret_mat[(i,i)] = 1.0 / ( exp_inp*(upper_bounds[i] - lower_bounds[i]) / (exp_inp + 1.0).powf(2.) as f64 );
            },
            _ => panic!()
        }
    }
    ret_mat
}

