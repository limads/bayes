use super::*;

pub struct RMHMCOutput {
    pub draws_out : DMatrix<f64>,
    pub accept_rate : f64
}

pub struct RMHMCSettings {
    pub n_draws : usize,
    pub n_burnin : usize,
    pub step_size : f64,
    pub leap_steps : usize,
    pub fp_steps : usize,
    pub vals_bound : bool,
    pub lower_bounds : DVector<f64>,
    pub upper_bounds : DVector<f64>
}

type MomentUpdateBx = Box<dyn Fn(&DVector<f64>, &DVector<f64>, f64, &DMatrix<f64>, &Matrices<f64>, Option<&mut DMatrix<f64>>)->DVector<f64>>;

fn rmhmc_int<'a>(
    initial_vals : &DVector<f64>,
    target_log_kernel : impl Fn(&DVector<f64>, Option<&mut DVector<f64>>)->f64 + Clone + 'static,
    tensor_fn : impl Fn(&DVector<f64>, Option<&mut Matrices<f64>>)->DMatrix<f64> + 'static,
    settings : &RMHMCSettings
) -> Result<RMHMCOutput, ()> {
    let n_vals = initial_vals.nrows();
    let lower_bounds = &settings.lower_bounds;
    let upper_bounds = &settings.upper_bounds;
    let bounds_type = determine_bounds_type(settings.vals_bound, n_vals, &lower_bounds, &upper_bounds);
    let box_log_kernel = boxed_log_prob_kernel_with_grad(settings.vals_bound, &bounds_type, &lower_bounds, &upper_bounds, &target_log_kernel);

    let mntm_update_fn : MomentUpdateBx = if settings.vals_bound {
        Box::new({
            let lower_bounds = lower_bounds.clone();
            let bounds_type = bounds_type.clone();
            let upper_bounds = upper_bounds.clone();

            // TODO confirm this clone won't mess with the closure state across calls.
            let target_log_kernel = target_log_kernel.clone();

            move |pos_inp, mntm_inp, step_size, inv_tensor_mat, tensor_deriv, jacob_matrix_out| {
                let n_vals = pos_inp.nrows();
                let mut grad_obj = DVector::zeros(n_vals);
                let pos_inv_trans = inv_transform(pos_inp, &bounds_type, &lower_bounds, &upper_bounds);
                target_log_kernel(&pos_inv_trans,Some(&mut grad_obj));

                for i in 0..n_vals {
                    let tmp_mat = inv_tensor_mat * tensor_deriv.slice(i);
                    grad_obj[i] = -1.0*grad_obj[i] + 0.5*( tmp_mat.trace() - (mntm_inp.transpose() * tmp_mat * inv_tensor_mat * mntm_inp )[0] );
                }

                let mut jacob_matrix = inv_jacobian_adjust(pos_inp,&bounds_type,&lower_bounds,&upper_bounds);

                if let Some(jacob_matrix_out) = jacob_matrix_out {
                    *jacob_matrix_out = jacob_matrix.clone();
                }
                step_size * jacob_matrix * grad_obj / 2.0
            }
        })
    } else {
        Box::new({

            // TODO confirm this clone won't mess with the closure state across calls.
            let target_log_kernel = target_log_kernel.clone();

            move |pos_inp, mntm_inp, step_size, inv_tensor_mat, tensor_deriv, jacob_matrix_out| {
                let n_vals = pos_inp.nrows();
                let mut grad_obj = DVector::zeros(n_vals);
                target_log_kernel(pos_inp,Some(&mut grad_obj));
                for i in 0..n_vals {
                    let tmp_mat = inv_tensor_mat * tensor_deriv.slice(i);
                    grad_obj[i] = -1.0*grad_obj[i] + 0.5*( tmp_mat.trace() - (mntm_inp.transpose() * tmp_mat * inv_tensor_mat * mntm_inp)[0] );
                }

                // This isn't used in the original impl
                let _mntm_out = step_size * grad_obj.clone() / 2.0;
                step_size * grad_obj / 2.0
            }
        })
    };

    let box_tensor_fn : Box<dyn Fn(&DVector<f64>, Option<&mut Matrices<f64>>)->DMatrix<f64>> = if settings.vals_bound {
        Box::new({
            let bounds_type = bounds_type.clone();
            let lower_bounds = lower_bounds.clone();
            let upper_bounds = upper_bounds.clone();
            move |vals_inp, tensor_deriv_out| {
                let vals_inv_trans = inv_transform(vals_inp, &bounds_type, &lower_bounds, &upper_bounds);
                tensor_fn(&vals_inv_trans, tensor_deriv_out)
            }
        })
    } else {
        Box::new(|vals_inp, tensor_deriv_out| {
            tensor_fn(vals_inp, tensor_deriv_out)
        })
    };

    let mut first_draw = initial_vals.clone();
    if settings.vals_bound {
        first_draw = transform(&initial_vals, &bounds_type, &lower_bounds, &upper_bounds);
    }

    let mut draws_out = DMatrix::zeros(settings.n_draws, n_vals);

    let mut prev_draw = first_draw.clone();
    let mut new_draw  = first_draw.clone();

    let mut new_mntm = DVector::zeros(n_vals);
    fill_with_std_normal(&mut new_mntm);

    // TODO define dimensionality
    let mut new_deriv_cube = Matrices::zeros(1,1,1);
    let mut new_tensor = box_tensor_fn(&new_draw,Some(&mut new_deriv_cube));

    let mut prev_tensor = new_tensor.clone();
    let mut inv_new_tensor = LU::new(new_tensor.clone()).try_inverse().unwrap();
    let mut inv_prev_tensor = inv_new_tensor.clone();

    let mut prev_deriv_cube = new_deriv_cube.clone();

    let cons_term = 0.5*(n_vals as f64)*( 2.0 * std::f64::consts::PI ).ln();

    let mut prev_U = cons_term - box_log_kernel(&first_draw, None) + LU::new(new_tensor.clone()).determinant().ln();
    let mut prop_U = prev_U;

    let (mut prop_K, mut prev_K) = (0., 0.);

    let mut n_accept = 0;
    let mut comp_val = 0.0;

    let mut r = DVector::zeros(n_vals);
    for jj in 0..(settings.n_draws + settings.n_burnin) {
        fill_with_std_normal(&mut r);
        new_mntm = Cholesky::new(prev_tensor.clone()).unwrap().l() * &r;
        prev_K = new_mntm.dot(&(inv_prev_tensor.clone()* &new_mntm )) / 2.0;
        new_draw = prev_draw.clone();
        for k in 0..settings.leap_steps {
            let mut prop_mntm = new_mntm.clone();
            for kk in 0..settings.fp_steps {
                prop_mntm = new_mntm.clone() + mntm_update_fn(&new_draw,&prop_mntm,settings.step_size,&inv_prev_tensor,&prev_deriv_cube,None);
            }

            new_mntm = prop_mntm.clone();
            let mut prop_draw = new_draw.clone();

            for kk in 0..settings.fp_steps {
                inv_new_tensor = LU::new(box_tensor_fn(&prop_draw,None)).try_inverse().unwrap();
                prop_draw = new_draw.clone() + 0.5 * settings.step_size * ( inv_prev_tensor.clone() + &inv_new_tensor ) * &new_mntm;
            }

            new_draw = prop_draw.clone();
            new_tensor = box_tensor_fn(&new_draw,Some(&mut new_deriv_cube));
            inv_new_tensor = LU::new(new_tensor.clone()).try_inverse().unwrap();
            new_mntm += mntm_update_fn(&new_draw,&new_mntm,settings.step_size,&inv_new_tensor,&new_deriv_cube,None);
        }

        prop_U = cons_term - box_log_kernel(&new_draw, None) + 0.5*(new_tensor.clone().determinant().ln());

        if !prop_U.is_finite() {
            prop_U = INFINITY;
        }

        prop_K = new_mntm.dot(&(inv_new_tensor.clone() * &new_mntm )) / 2.0;

        comp_val = (-1.0*(prop_U + prop_K) + (prev_U + prev_K)).min(0.0);
        let z_rand : f64 = rand::random();
        if z_rand < comp_val.exp() {
            prev_draw = new_draw.clone();

            prev_U = prop_U;
            prev_K = prop_K;

            prev_tensor     = new_tensor.clone();
            inv_prev_tensor = inv_new_tensor.clone();
            prev_deriv_cube = new_deriv_cube.clone();

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
            let invt = inv_transform(&draws_out.row(jj).transpose(), &bounds_type, &lower_bounds, &upper_bounds);
            draws_out.row_mut(jj).tr_copy_from(&invt);
        }
    }


    let accept_rate = n_accept as f64 / settings.n_draws as f64;

    return Ok(RMHMCOutput {
        accept_rate,
        draws_out
    })
}
