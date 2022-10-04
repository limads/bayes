/* Credit goes to mcmclib (Keith O'hara), which released the original C++ implementation via Apache 2.0 */

use super::*;

pub struct AEESOutput {
    pub draws_out : DMatrix<f64>
}

pub struct AEESSettings {
    pub n_draws : usize,
    pub n_initial_draws : usize,
    pub n_burnin : usize,
    pub prob_par : f64,
    pub vals_bound : bool,
    pub temper_vec : DVector<f64>,
    pub lower_bounds : DVector<f64>,
    pub upper_bounds : DVector<f64>,
    pub n_rings : usize,
    pub par_scale : f64,
    pub cov_mat : Option<DMatrix<f64>>
}

fn aees_int(
    initial_vals : &DVector<f64>,
    target_log_kernel : impl Fn(&DVector<f64>)->f64,
    settings : AEESSettings
) -> Result<AEESOutput, ()> {
    let n_vals = initial_vals.nrows();
    let K = settings.temper_vec.nrows() + 1;
    let mut temper_vec = DVector::zeros(K);
    temper_vec.index_mut((0..(K-1), ..)).copy_from(&settings.temper_vec);
    temper_vec[K-1] = 1.0;
    temper_vec.as_mut_slice().sort_by(|a, b| b.total_cmp(&a) );
    let total_draws = settings.n_draws + K*(settings.n_initial_draws + settings.n_burnin);
    let cov_mcmc : DMatrix<f64> = settings.cov_mat.clone().unwrap_or(DMatrix::identity(n_vals, n_vals));
    assert!(cov_mcmc.nrows() == cov_mcmc.ncols() && cov_mcmc.nrows() == initial_vals.len());
    let sqrt_cov_mcmc = settings.par_scale * Cholesky::new(cov_mcmc).unwrap().l();
    let lower_bounds = &settings.lower_bounds;
    let upper_bounds = &settings.upper_bounds;
    let bounds_type = determine_bounds_type(settings.vals_bound, n_vals, lower_bounds, upper_bounds);
    let box_log_kernel = boxed_log_prob_kernel(settings.vals_bound, &bounds_type, &lower_bounds, &upper_bounds, target_log_kernel);
    let mut first_draw = initial_vals.clone();
    if settings.vals_bound {
        first_draw = transform(initial_vals, &bounds_type, &lower_bounds, &upper_bounds);
    }
    let mut X_out = Matrices::zeros(n_vals, K, total_draws);
    let mut X_new = DMatrix::zeros(n_vals, K);
    X_new.column_mut(0).copy_from(&first_draw);
    let mut ring_vals = DMatrix::zeros(K, settings.n_rings-1);
    let mut kernel_vals = DMatrix::zeros(K,total_draws);
    let mut kernel_vals_prev = DMatrix::zeros(2,K);
    let mut kernel_vals_new = DMatrix::zeros(2,K);
    for n in 0..total_draws {
        let X_prev = X_new.clone();
        kernel_vals_prev = kernel_vals_new.clone();
        let mut val_out_hot = 0.0;
        X_new.column_mut(0)
            .copy_from(&single_step_mh(X_prev.column(0), temper_vec[0], &sqrt_cov_mcmc, &box_log_kernel, Some(&mut val_out_hot)));
        kernel_vals_new.column_mut(0).fill(val_out_hot);
        for j in 1..K {
            if n > j*(settings.n_initial_draws + settings.n_burnin) {
                let mut val_out_j = 0.0;
                let z_eps : f64 = rand::random();
                if z_eps > settings.prob_par {
                    X_new.column_mut(j)
                        .copy_from(&single_step_mh(X_prev.column(j), temper_vec[j], &sqrt_cov_mcmc, &box_log_kernel, Some(&mut val_out_j)));
                    kernel_vals_new[(0,j)] = val_out_j / temper_vec[j-1];
                    kernel_vals_new[(1,j)] = val_out_j / temper_vec[j];
                } else {
                    let draws_j_begin_ind = (j-1)*(settings.n_initial_draws + settings.n_burnin);

                    let ring_ind_spacing = ((n - draws_j_begin_ind + 1) as f64 / settings.n_rings as f64).floor() as usize;

                    if ring_ind_spacing == 0 {
                        X_new.column_mut(j).copy_from(&X_prev.column(j));
                        kernel_vals_new.column_mut(j).copy_from(&kernel_vals_prev.column(j));
                    } else {

                        let mut past_kernel_vals : Vec<(usize, f64)> = kernel_vals
                            .row(j-1).index((.., draws_j_begin_ind..=n))
                            .iter()
                            .copied()
                            .enumerate()
                            .collect();
                        past_kernel_vals.sort_by(|a, b| a.1.total_cmp(&b.1) );

                        for i in 0..(settings.n_rings-1) {
                            let ring_i_ind = (i+1)*ring_ind_spacing;
                            ring_vals[(j-1,i)] = (past_kernel_vals[ring_i_ind].1 + past_kernel_vals[ring_i_ind-1].1) / 2.0;
                        }

                        let mut which_ring = 0;
                        while which_ring < (settings.n_rings-1) && kernel_vals[(j,n-1)] > ring_vals[(j-1,which_ring)] {
                            which_ring+=1;
                        }

                        let z_tmp : f64 = rand::random();
                        let ind_mix = past_kernel_vals[ring_ind_spacing*which_ring + (z_tmp * ring_ind_spacing as f64).floor() as usize].0;

                        X_new.column_mut(j).copy_from(&X_out.slice(ind_mix).column(j-1));

                        val_out_j = box_log_kernel(&X_new.column(j).clone_owned());

                        kernel_vals_new[(0,j)] = val_out_j / temper_vec[j-1];
                        kernel_vals_new[(1,j)] = val_out_j / temper_vec[j];

                        let comp_val_1 = kernel_vals_new[(1,j)] - kernel_vals_prev[(1,j)];
                        let comp_val_2 = kernel_vals_prev[(0,j)] - kernel_vals_new[(0,j)];

                        let comp_val = (comp_val_1 + comp_val_2).min(0.0);
                        let z : f64 = rand::random();

                        if z > comp_val.exp() {
                            X_new.column_mut(j).copy_from(&X_prev.column(j));
                            kernel_vals_new.column_mut(j).copy_from(&kernel_vals_prev.column(j));
                        }
                    }
                }
                kernel_vals[(j,n)] = box_log_kernel(&X_new.column(j).clone_owned());
            }
        }
        X_out.slice_mut(n).copy_from(&X_new);
    }

    //

    let mut draws_out = DMatrix::zeros(total_draws,n_vals);

    for i in 0..total_draws {
        let mut tmp_vec = X_out.slice(i).column(K-1).clone_owned();
        if settings.vals_bound {
            tmp_vec = inv_transform(&tmp_vec, &bounds_type, &lower_bounds, &upper_bounds);
        }
        draws_out.row_mut(i).tr_copy_from(&tmp_vec);
    }
    draws_out = draws_out.remove_rows(0, K*(settings.n_initial_draws + settings.n_burnin)-1);
    return Ok(AEESOutput { draws_out });
}

fn single_step_mh(
    X_prev : DVectorSlice<f64>,
    temper_val : f64,
    sqrt_cov_mcmc : &DMatrix<f64>,
    target_log_kernel : &dyn Fn(&DVector<f64>)->f64,
    val_out : Option<&mut f64>
) -> DVector<f64> {
    let n_vals = X_prev.nrows();
    let mut r = DVector::zeros(n_vals);
    fill_with_std_normal(&mut r);
    let X_new = X_prev.clone_owned() + temper_val.sqrt() * sqrt_cov_mcmc.clone() * &r;
    let val_new  = target_log_kernel(&X_new);
    let val_prev = target_log_kernel(&X_prev.clone_owned());
    let comp_val = ((val_new - val_prev) / temper_val).min(0.0);
    let z : f64 = rand::random();
    if z < comp_val.exp() {
        if let Some(val_out) = val_out {
            *val_out = val_new;
        }
        X_new
    } else {
        if let Some(val_out) = val_out {
            *val_out = val_prev;
        }
        X_prev.clone_owned()
    }
}