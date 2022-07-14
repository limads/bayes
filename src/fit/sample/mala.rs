/* Credit goes to mcmclib (Keith O'hara), which released the original C++ implementation via Apache 2.0 */

use super::*;

pub struct MALASettings {
    pub n_draws : usize,
    pub n_burnin : usize,
    pub step_size : f64,
    pub vals_bound : bool,
    pub lower_bounds : DVector<f64>,
    pub upper_bounds : DVector<f64>,
    pub precond_matrix : Option<DMatrix<f64>>
}

pub struct MALAOutput {
    pub draws_out : DMatrix<f64>,
    pub accept_rate : f64
}

fn hmc_int(
    initial_vals : &DVector<f64>,
    target_log_kernel : impl Fn(&DVector<f64>, Option<&mut DVector<f64>>)->f64,
    settings : &MALASettings
) -> Result<MALAOutput, ()> {
    let n_vals = initial_vals.nrows();
    let precond_matrix = settings.precond_matrix.clone().unwrap_or(DMatrix::identity(n_vals, n_vals));
    let sqrt_precond_matrix = Cholesky::new(precond_matrix.clone()).unwrap().l();
    let lower_bounds = &settings.lower_bounds;
    let upper_bounds = &settings.upper_bounds;
    let bounds_type = determine_bounds_type(settings.vals_bound, n_vals, lower_bounds, upper_bounds);
    let box_log_kernel = boxed_log_prob_kernel_with_grad(settings.vals_bound, &bounds_type, &lower_bounds, &upper_bounds, &target_log_kernel);

    let mala_mean_fn : Box<dyn Fn(&DVector<f64>, f64, Option<&mut DMatrix<f64>>)->DVector<f64>> = if settings.vals_bound {
        Box::new(|vals_inp, step_size, jacob_matrix_out| {
            let n_vals = vals_inp.nrows();
            let mut grad_obj = DVector::zeros(n_vals);
            let vals_inv_trans = inv_transform(vals_inp, &bounds_type, &lower_bounds, &upper_bounds);
            (&target_log_kernel)(&vals_inv_trans,Some(&mut grad_obj));
            let mut jacob_matrix = inv_jacobian_adjust(vals_inp,&bounds_type,&lower_bounds,&upper_bounds);
            if let Some(jacob_matrix_out) = jacob_matrix_out {
                *jacob_matrix_out = jacob_matrix.clone();
            }
            vals_inp + step_size * step_size * jacob_matrix * &precond_matrix * &grad_obj / 2.0
        })
    } else {
        Box::new(|vals_inp, step_size, jacob_matrix_out| {
            let n_vals = vals_inp.nrows();
            let mut grad_obj = DVector::zeros(n_vals);
            (&target_log_kernel)(vals_inp, Some(&mut grad_obj));
            vals_inp + step_size * step_size * precond_matrix.clone() * &grad_obj / 2.0
        })
    };

    let mut first_draw = initial_vals.clone();
    if settings.vals_bound {
        first_draw = transform(&initial_vals, &bounds_type, &lower_bounds, &upper_bounds);
    }

    let mut draws_out = DMatrix::zeros(settings.n_draws, n_vals);
    let mut prev_LP = box_log_kernel(&first_draw, None);
    let mut prop_LP = prev_LP;

    let mut prev_draw = first_draw.clone();
    let mut new_draw  = first_draw.clone();

    let mut n_accept = 0;
    let mut krand = DVector::zeros(n_vals);

    for jj in 0..(settings.n_draws + settings.n_burnin) {
        if settings.vals_bound {
            fill_with_std_normal(&mut krand);
            let mut jacob_matrix = DMatrix::zeros(n_vals, n_vals);
            let mean_vec = mala_mean_fn(&prev_draw, settings.step_size, Some(&mut jacob_matrix));
            new_draw = mean_vec + settings.step_size * Cholesky::new(jacob_matrix).unwrap().l() * &sqrt_precond_matrix * &krand;
        } else {
            new_draw = mala_mean_fn(&prev_draw, settings.step_size, None) + settings.step_size * sqrt_precond_matrix.clone() * &krand;
        }

        prop_LP = box_log_kernel(&new_draw, None);
        if !prop_LP.is_finite() {
            prop_LP = NEG_INFINITY;
        }

        let comp_val = (prop_LP - prev_LP + mala_prop_adjustment(&new_draw, &prev_draw, settings.step_size, settings.vals_bound, &precond_matrix, &mala_mean_fn)).min(0.0);
        let z : f64 = rand::random();

        if z < comp_val.exp() {
            prev_draw = new_draw.clone();
            prev_LP = prop_LP;

            if jj >= settings.n_burnin {
                draws_out.row_mut(jj - settings.n_burnin).tr_copy_from(&new_draw);
                n_accept+=1;
            }
        } else {
            if jj >= settings.n_burnin {
                draws_out.row_mut(jj - settings.n_burnin).tr_copy_from(&prev_draw);
            }
        }
    }

    if settings.vals_bound {
        for jj in 0..settings.n_draws {
            let invt = inv_transform(&draws_out.row(jj).clone_owned().transpose(), &bounds_type, &lower_bounds, &upper_bounds);
            draws_out.row_mut(jj).tr_copy_from(&invt);
        }
    }

    let accept_rate = n_accept as f64 / settings.n_draws as f64;
    return Ok(MALAOutput {
        draws_out,
        accept_rate
    })
}

fn mala_prop_adjustment(
    prop_vals : &DVector<f64>,
    prev_vals : &DVector<f64>,
    step_size : f64,
    vals_bound : bool,
    precond_mat : &DMatrix<f64>,
    mala_mean_fn : &dyn Fn(&DVector<f64>, f64, Option<&mut DMatrix<f64>>)->DVector<f64>
) -> f64 {
    let step_size_sq = step_size*step_size;
    if vals_bound {
        let mut prop_inv_jacob = DMatrix::zeros(prop_vals.nrows(), prop_vals.ncols());
        let mut prev_inv_jacob = prop_inv_jacob.clone();
        let prop_mean = mala_mean_fn(prop_vals, step_size, Some(&mut prop_inv_jacob));
        let prev_mean = mala_mean_fn(prev_vals, step_size, Some(&mut prev_inv_jacob));
        let prev_lp = crate::prob::multinormal_log_prob(&prev_vals, &prop_mean, &(step_size_sq*prop_inv_jacob.clone()*precond_mat));
        let prop_lp = crate::prob::multinormal_log_prob(&prop_vals, &prev_mean, &(step_size_sq*prop_inv_jacob*precond_mat));
        prev_lp - prop_lp
    } else {
        let prop_mean = mala_mean_fn(prop_vals, step_size, None);
        let prev_mean = mala_mean_fn(prev_vals, step_size, None);
        let prev_lp = crate::prob::multinormal_log_prob(&prev_vals, &prop_mean, &(step_size_sq*precond_mat));
        let prop_lp = crate::prob::multinormal_log_prob(&prop_vals, &prev_mean, &(step_size_sq*precond_mat));
        prev_lp - prop_lp
    }
}

