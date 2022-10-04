/* Credit goes to mcmclib (Keith O'hara), which released the original C++ implementation via Apache 2.0 */

use super::*;

pub struct HMCSettings {
    pub n_draws : usize,
    pub n_burnin : usize,
    pub step_size : f64,
    pub n_leap_steps : usize,
    pub lower_bounds : DVector<f64>,
    pub upper_bounds : DVector<f64>,
    pub precond_mat : Option<DMatrix<f64>>,
    pub vals_bound : bool
}

pub struct HMCOutput {
    pub draws_out : DMatrix<f64>,
    pub accept_rate : f64
}

// The target_log_kernel here carries the parameter vector and gradient vector.
fn hmc_int(
    initial_vals : &DVector<f64>,
    target_log_kernel : impl Fn(&DVector<f64>, Option<&mut DVector<f64>>)->f64,
    settings : &HMCSettings
) -> Result<HMCOutput, ()> {
    let n_vals = initial_vals.nrows();
    let precond_matrix = settings.precond_mat.clone().unwrap_or(DMatrix::identity(n_vals, n_vals));
    let inv_precond_matrix = LU::new(precond_matrix.clone()).try_inverse().unwrap();
    let sqrt_precond_matrix = Cholesky::new(precond_matrix.clone()).unwrap().l();
    let lower_bounds = &settings.lower_bounds;
    let upper_bounds = &settings.upper_bounds;
    let bounds_type = determine_bounds_type(settings.vals_bound, n_vals, &lower_bounds, &upper_bounds);
    let box_log_kernel = boxed_log_prob_kernel_with_grad(settings.vals_bound, &bounds_type, &lower_bounds, &upper_bounds, &target_log_kernel);

    // Carries position, momentum, step_size, and jacobian matrix output
    let mntm_update_fn : Box<dyn Fn(&DVector<f64>, &DVector<f64>, f64, Option<&mut DMatrix<f64>>)->DVector<f64>> = if settings.vals_bound {
        Box::new(|pos_inp, mntm_inp, step_size, jacob_matrix_out| {
            let n_vals = pos_inp.nrows();
            let mut grad_obj = DVector::zeros(n_vals);
            let pos_inv_trans = inv_transform(pos_inp, &bounds_type, &lower_bounds, &upper_bounds);
            (&target_log_kernel)(&pos_inv_trans,Some(&mut grad_obj));
            let jacob_matrix = inv_jacobian_adjust(pos_inp,&bounds_type,&lower_bounds,&upper_bounds);
            if let Some(jacob_matrix_out) = jacob_matrix_out {
                *jacob_matrix_out = jacob_matrix.clone();
            }
            mntm_inp + step_size * jacob_matrix * grad_obj / 2.0
        })
    } else {
        Box::new(|pos_inp, mntm_inp, step_size, jacob_matrix_out| {
            let n_vals = pos_inp.nrows();
            let mut grad_obj = DVector::zeros(n_vals);
            (&target_log_kernel)(pos_inp,Some(&mut grad_obj));
            mntm_inp + step_size * grad_obj / 2.0
        })
    };

    let mut first_draw = initial_vals.clone();
    if settings.vals_bound {
        first_draw = transform(&initial_vals, &bounds_type, &lower_bounds, &upper_bounds);
    }

    let mut draws_out = DMatrix::zeros(settings.n_draws, n_vals);
    let mut prev_U = -1.0*box_log_kernel(&first_draw, None);
    let mut prop_U = prev_U;
    let (mut prop_K, mut prev_K) = (0., 0.);
    let mut prev_draw = first_draw.clone();
    let mut new_draw  = first_draw.clone();
    let mut new_mntm = DVector::zeros(n_vals);
    let mut n_accept = 0;
    let mut r = DVector::zeros(n_vals);
    for jj  in 0..(settings.n_draws + settings.n_burnin) {
        fill_with_std_normal(&mut r);
        new_mntm = sqrt_precond_matrix.clone() * &r;
        prev_K = new_mntm.dot(&(inv_precond_matrix.clone() * &new_mntm)) / 2.0;
        new_draw = prev_draw.clone();
        for k in 0..settings.n_leap_steps {
            new_mntm = mntm_update_fn(&new_draw,&new_mntm,settings.step_size,None);
            new_draw += settings.step_size * (inv_precond_matrix.clone() * &new_mntm);
            new_mntm = mntm_update_fn(&new_draw,&new_mntm,settings.step_size,None);
        }
        prop_U = -1.0*box_log_kernel(&new_draw, None);
        if !prop_U.is_finite() {
            prop_U = f64::INFINITY;
        }
        prop_K = new_mntm.dot(&(inv_precond_matrix.clone() * &new_mntm)) / 2.0;

        let comp_val = (-prop_U -prop_K + prev_U + prev_K).min(0.0);
        let z : f64 = rand::random();

        if z < comp_val.exp() {
            prev_draw = new_draw.clone();
            prev_U = prop_U;
            prev_K = prop_K;
            if jj >= settings.n_burnin {
                draws_out.row_mut(jj - settings.n_burnin).tr_copy_from(&new_draw);
                n_accept+=1;
            }
        }
        else {
            if jj >= settings.n_burnin {
                draws_out.row_mut(jj - settings.n_burnin).tr_copy_from(&prev_draw);
            }
        }
    }

    if settings.vals_bound {
        for jj in 0..settings.n_draws {
            let invt = inv_transform(&draws_out.row(jj).transpose(), &bounds_type, &lower_bounds, &upper_bounds);
            draws_out.row_mut(jj).tr_copy_from(&invt);
        }
    }
    let accept_rate = n_accept as f64 / settings.n_draws as f64;
    Ok(HMCOutput {
        draws_out,
        accept_rate
    })
}
