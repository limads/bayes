/* Credit goes to mcmclib (Keith O'hara), which released the original C++ implementation via Apache 2.0 */

use super::*;
use rand::distributions::{Distribution, Uniform};

pub struct DEOutput {
    // Output here, unlike rwmh/aees, has three channels: n_pop x n_vals * n_gen
    // where different samples run across the last n_gen dimension ("wide" matrix).
    pub draws_out : Matrices<f64>,
    pub accept_rate : f64
}

pub struct DESettings {
    pub n_pop : usize,
    pub n_gen : usize,
    pub n_burnin : usize,
    pub jumps : bool,
    pub par_b : f64,
    pub par_gamma : f64,
    pub par_gamma_jump : f64,
    pub initial_lb : Option<DVector<f64>>,
    pub initial_ub : Option<DVector<f64>>,
    pub lower_bounds : DVector<f64>,
    pub upper_bounds : DVector<f64>,
    pub vals_bound : bool
}

fn de_int(
    initial_vals : &DVector<f64>,
    target_log_kernel : impl Fn(&DVector<f64>)->f64,
    settings : &DESettings
) -> Result<DEOutput, ()> {
    let n_vals = initial_vals.nrows();

    // const double par_gamma = settings.de_par_gamma;
    let par_gamma = 2.38 / (2.0*n_vals as f64).sqrt();

    let par_initial_lb = settings.initial_lb.clone().unwrap_or(initial_vals.add_scalar(-0.5));
    let par_initial_ub = settings.initial_ub.clone().unwrap_or(initial_vals.add_scalar(0.5));

    let lower_bounds = &settings.lower_bounds;
    let upper_bounds = &settings.upper_bounds;

    let bounds_type = determine_bounds_type(settings.vals_bound, n_vals, lower_bounds, upper_bounds);
    let box_log_kernel = boxed_log_prob_kernel(settings.vals_bound, &bounds_type, &lower_bounds, &upper_bounds, target_log_kernel);
    //
    let mut target_vals = DVector::zeros(settings.n_pop);
    let mut X = DMatrix::zeros(settings.n_pop,n_vals);
    let mut r = RowDVector::zeros(n_vals);
    for i in 0..settings.n_pop {
        fill_with_std_normal(&mut r);
        X.row_mut(i).copy_from(&(par_initial_lb.transpose() + (par_initial_ub.transpose() - par_initial_lb.transpose()).component_mul(&r)));
        let mut prop_kernel_val = box_log_kernel(&X.row(i).clone_owned().transpose());
        if !prop_kernel_val.is_finite() {
            prop_kernel_val = NEG_INFINITY;
        }
        target_vals[i] = prop_kernel_val;
    }

    //
    let mut draws_out = Matrices::zeros(settings.n_pop,n_vals,settings.n_gen);

    let mut n_accept = 0;
    let mut par_gamma_run = par_gamma;

    let unif = Uniform::from(0..(settings.n_pop-1));
    let mut prop_rand = RowDVector::zeros(n_vals);
    let mut rng = rand::thread_rng();

    for j in 0..(settings.n_gen+settings.n_burnin) {
        let temperature_j = de_cooling_schedule(j as i32,settings.n_gen as i32);

        if settings.jumps && ((j+1) % 10 == 0) {
            par_gamma_run = settings.par_gamma_jump;
        }

        for i in 0..settings.n_pop {
            let (mut R_1, mut R_2) = (0, 0);
            loop {
                R_1 = unif.sample(&mut rng);
                if R_1 == i {
                    break;
                }
            }
            loop {
                R_2 = unif.sample(&mut rng);
                if R_2 == i || R_2 == R_1 {
                    break;
                }
            }

            fill_with_uniform(&mut prop_rand);
            prop_rand = (prop_rand*(2.0*settings.par_b)).add_scalar(-1.0*settings.par_b);

            let X_prop = X.row(i).clone_owned() + par_gamma_run * ( X.row(R_1) - X.row(R_2) ) + &prop_rand;

            let mut prop_kernel_val = box_log_kernel(&X_prop.transpose());

            if !prop_kernel_val.is_finite() {
                prop_kernel_val = NEG_INFINITY;
            }

            let comp_val = prop_kernel_val - target_vals[i];
            let z : f64 = rand::random();

            if comp_val > temperature_j * z.ln() {
                X.row_mut(i).copy_from(&X_prop);
                target_vals[i] = prop_kernel_val;
                if j >= settings.n_burnin {
                    n_accept+=1;
                }
            }
        }

        if j >= settings.n_burnin {
            draws_out.slice_mut(j-settings.n_burnin).copy_from(&X);
        }

        if settings.jumps && ((j+1) % 10 == 0) {
            par_gamma_run = par_gamma;
        }
    }

    if settings.vals_bound {
        for ii in 0..settings.n_gen {
            for jj in 0..settings.n_pop {
                let invt = inv_transform(&draws_out.slice(ii).row(jj).transpose(), &bounds_type, &lower_bounds, &upper_bounds);
                draws_out.slice_mut(ii)
                    .row_mut(jj)
                    .tr_copy_from(&invt);
            }
        }
    }

    let accept_rate = n_accept as f64 / (settings.n_pop*settings.n_gen) as f64;
    Ok(DEOutput {
        accept_rate,
        draws_out
    })
}

fn de_cooling_schedule(_s : i32, _N_gen : i32) -> f64 {
    1.0
}
