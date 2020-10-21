use crate::gsl::multimin::*;
use nalgebra::*;
use std::ffi::c_void;
use std::mem;
use crate::gsl::utils::*;
use super::*;

unsafe extern "C" fn gradient<T : Sized + Clone>(
    p : *const gsl_vector,
    extra : *mut c_void,
    g : *mut gsl_vector
) {
    let extra_ptr : *mut MinProblem<T> = mem::transmute(extra);
    let extra = &mut *extra_ptr;
    let user_p : DVector<f64> = (*p).clone().into();
    if let Some(ref mut grad_fn) = &mut extra.grad_update {
        let user_grad = (grad_fn)(&user_p, &mut extra.user_data);
        let updated_grad : gsl_vector = user_grad.into();
        *g = updated_grad;
    } else {
        println!("Could not retrieve grad function");
    }
}

unsafe extern "C" fn obj_grad<T : Sized + Clone>(
    x : *const gsl_vector,
    extra : *mut c_void,
    f : *mut f64,
    g : *mut gsl_vector
) {
    *f = objective::<T>(x, extra);
    gradient::<T>(x, extra, g);
}

/// Returns [params]; [gradient]; min
unsafe fn fetch_data_from_grad_minimizer<'a>(
    minimizer : &'a *mut gsl_multimin_fdfminimizer
) -> (DVectorSlice<'a, f64>, DVectorSlice<'a, f64>, f64) {
    let x = gsl_multimin_fdfminimizer_x(*minimizer);
    let user_x = DVectorSlice::<f64>::slice_from_gsl(&*x);
    let g = gsl_multimin_fdfminimizer_gradient(*minimizer);
    let user_g = DVectorSlice::<f64>::slice_from_gsl(&*g);
    let obj = gsl_multimin_fdfminimizer_minimum(*minimizer);
    (user_x, user_g, obj)
}

/// Solves a minimization problem,
/// returing, if successful, the optimized parameter, the first derivative
/// at the minimum, the minimum scalar value and the user_data struct
/// back to the user. If the matrices/vectors traj, grad and eval are supplied,
/// writes the iteration states into them.
pub fn minimize_with_grad<T : Sized + Clone>(
    params : DVector<f64>,
    user_data : T,
    fn_update : Box<dyn FnMut(&DVector<f64>, &mut T)->f64>,
    grad_update : Box<dyn FnMut(&DVector<f64>, &mut T)->DVector<f64>>,
    mut opt_traj : Option<&mut DMatrix<f64>>,
    mut opt_grad : Option<&mut DMatrix<f64>>,
    mut opt_eval : Option<&mut DVector<f64>>,
    max_iter : usize,
) -> Result<(DVector<f64>, DVector<f64>, f64, usize, T), String> {

    // Verify if the memory arrays have adequate size.
    if let (Some(traj), Some(grad), Some(eval)) = (&opt_traj, &opt_grad, &opt_eval) {
        assert!(traj.ncols() == max_iter);
        assert!(grad.ncols() == max_iter);
        assert!(eval.nrows() == max_iter);
    }

    // Initialize the minimizer state.
    let n = params.nrows();
    let mut probl = MinProblem {
        user_data,
        user_params : params.clone(),
        user_grad : DVector::<f64>::from_element(n, 0.1),
        fn_update,
        grad_update : Some(grad_update)
    };
    unsafe {
        let minimizer = gsl_multimin_fdfminimizer_alloc(
            gsl_multimin_fdfminimizer_vector_bfgs2,
            n
        );
        let mut fdf = gsl_multimin_function_fdf {
            f : Some(objective::<T>),
            df : Some(gradient::<T>),
            fdf : Some(obj_grad::<T>),
            n : n,
            params : ((&mut probl) as *mut MinProblem<T>) as *mut ::std::os::raw::c_void
        };
        let params_gsl : gsl_vector = params.into();
        let mut steps : Vec<OptimizationStep> = Vec::new();
        let init_status = gsl_multimin_fdfminimizer_set(
            minimizer,
            &mut fdf as *mut _,
            &params_gsl,
            1.0,     // First step size
            0.1      // Tolerance (0.1 recommended by gsl doc)
        );
        match GSLStatus::from_code(init_status) {
            GSLStatus::Success => { },
            status => return Err(format!("Error initializing minimizer: {:?}", status))
        };
        // Iterate over minimization steps. Return early if minimum is found.
        for i in 0..max_iter {
            let res = gsl_multimin_fdfminimizer_iterate(minimizer);

            println!("Minimum: {}", gsl_multimin_fdfminimizer_minimum(minimizer));

            match GSLStatus::from_code(res) {
                GSLStatus::EnoProg => {
                    if i == 0 {
                        return Err(format!("Unable to improve on estimate at first iteration"));
                    } else {
                        let (user_x, user_g, obj) = fetch_data_from_grad_minimizer(&minimizer);
                        /*// TODO read actual state value from params_gsl.
                        let user_x = DVectorSlice::slice_from_gsl(&params_gsl).clone_owned();
                        let user_g = probl.user_grad.clone();
                        let mut data = probl.user_data.clone();
                        let obj = (probl.fn_update)(&user_x, &mut data);*/
                        println!("Converged in {} iterations", i);
                        println!("x = {}; g = {}; y = {}", user_x, user_g, obj);
                        // gsl_multimin_fdfminimizer_free(minimizer);
                        return Ok((DVector::from(user_x), DVector::from(user_g), obj, i, probl.user_data));
                    }
                },
                _ => { }
            }
            let (user_x, user_g, obj) = fetch_data_from_grad_minimizer(&minimizer);

            if let Some(traj) = &mut opt_traj {
                traj.column_mut(i).copy_from(&user_x);
            }
            if let Some(eval) = &mut opt_eval {
                eval[i] = obj;
            }
            if let Some(grad) = &mut opt_grad {
                grad.column_mut(i).copy_from(&user_g);
            }

            let param_norm = user_x.norm();
            let jac_norm = user_g.norm();
            steps.push(OptimizationStep {
                obj,
                jac_norm,
                param_norm
            });

            // Stop when gradient magnitude is within 0.1% of parameter vector magnitude
            let g = gsl_multimin_fdfminimizer_gradient(minimizer);
            let res = gsl_multimin_test_gradient(g, 0.001 * param_norm);
            match GSLStatus::from_code(res) {
                GSLStatus::Continue => { },
                GSLStatus::Success => {
                    println!("Converged in {} iterations", i);
                    // gsl_multimin_fdfminimizer_free(minimizer);
                    return Ok((DVector::from(user_x), DVector::from(user_g), obj, i, probl.user_data));
                },
                status => {
                    println!("Steps: {:?}", steps);
                    // gsl_multimin_fdfminimizer_free(minimizer);
                    return Err(format!("Error (Iteration {}): {:?}", i, status));
                }
            }
        }
        println!("Steps: {:?}", steps);
        //gsl_multimin_fdfminimizer_free(minimizer);
        Err(String::from("Maximum number of iterations reached"))
    }
}

