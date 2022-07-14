/* Credit goes to mcmclib (Keith O'hara), which released the original C++ implementation via Apache 2.0 */

use super::*;

pub struct RWMHSettings {
    pub n_draws : usize,
    pub n_burnin : usize,
    pub par_scale : f64,
    pub vals_bound : bool,
    pub lower_bounds : DVector<f64>,
    pub upper_bounds : DVector<f64>,
    pub cov_mat : Option<DMatrix<f64>>
}

pub struct RWMCOutput {
    pub draws_out : DMatrix<f64>,
    pub accept_rate : f64
}

pub struct RWMHState {

    krand : DVector<f64>,

    prev_draw : DVector<f64>,

    new_draw : DVector<f64>,

    // Proposed new log-probability
    prop_LP : f64,

    // Old log-probability
    prev_LP : f64,

    draws_out : DMatrix<f64>

}

impl RWMHState {

    pub fn init() -> Self {
        unimplemented!()
    }

    /* Returns true when algotithm is done */
    pub fn step(&mut self) -> bool {
        false
    }

    pub fn finish(&mut self) -> Option<RWMCOutput> {
        None
    }

}

fn rwmh_int(
    initial_vals : &DVector<f64>,
    target_log_kernel : impl Fn(&DVector<f64>)->f64,
    settings : &RWMHSettings
) -> Result<RWMCOutput, ()> {
    let n_vals = initial_vals.nrows();
    let n_draws_keep = settings.n_draws;
    let n_draws_burnin = settings.n_burnin;
    let par_scale = settings.par_scale;
    let cov_mcmc : DMatrix<f64> = settings.cov_mat.clone().unwrap_or(DMatrix::identity(n_vals, n_vals));
    assert!(cov_mcmc.nrows() == cov_mcmc.ncols() && cov_mcmc.nrows() == initial_vals.len());
    let vals_bound = settings.vals_bound;
    let lower_bounds = &settings.lower_bounds;
    let upper_bounds = &settings.upper_bounds;
    let bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);
    let box_log_kernel = boxed_log_prob_kernel(vals_bound, &bounds_type, &lower_bounds, &upper_bounds, target_log_kernel);
    let mut first_draw = initial_vals.clone();
    if vals_bound {
        first_draw = transform(initial_vals, &bounds_type, lower_bounds, upper_bounds);
    }
    let mut draws_out = DMatrix::zeros(n_draws_keep, n_vals);

    let mut prev_LP = box_log_kernel(&first_draw);
    let mut prop_LP = prev_LP;

    let mut prev_draw = first_draw.clone();
    let mut new_draw  = first_draw.clone();

    let mut cov_mcmc_sc   = par_scale * par_scale * cov_mcmc;
    let mut cov_mcmc_chol = linalg::Cholesky::new(cov_mcmc_sc).unwrap().l();

    let mut n_accept = 0;
    let mut krand = DVector::zeros(n_vals);

    for jj in 0..(n_draws_keep+n_draws_burnin) {
        fill_with_std_normal(&mut krand);
        new_draw = prev_draw.clone() + (cov_mcmc_chol.clone() * &krand);
        prop_LP = box_log_kernel(&new_draw);

        if !prop_LP.is_finite() {
            prop_LP = NEG_INFINITY;
        }

        let comp_val = (prop_LP - prev_LP).min(0.0);
        let z : f64 = rand::random();
        if z < comp_val.exp() {
            prev_draw = new_draw.clone();
            prev_LP = prop_LP;
            if jj >= n_draws_burnin {
                // TODO transpose_copy_from
                draws_out.row_mut(jj - n_draws_burnin).copy_from(&new_draw.clone().transpose());
                n_accept+=1;
            }
        } else {
            if jj >= n_draws_burnin {
                // TODO transpose_copy_from
                draws_out.row_mut(jj - n_draws_burnin).copy_from(&prev_draw.clone().transpose());
            }
        }
    }

    if vals_bound {
        for jj in 0..n_draws_keep {
            let inv = inv_transform(&draws_out.row(jj).transpose(), &bounds_type, lower_bounds, upper_bounds);
            draws_out.row_mut(jj).copy_from(&inv.transpose());
        }
    }

    let accept_rate = n_accept as f64 / n_draws_keep as f64;
    return Ok(RWMCOutput { draws_out, accept_rate });
}
