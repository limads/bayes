use crate::gsl::multimin::*;
use nalgebra::*;
use crate::gsl::vector_double::*;
use crate::gsl::matrix_double::*;
use std::ffi::c_void;
use std::mem;
use crate::gsl::utils::*;
use super::*;

unsafe extern "C" fn gradient<T : Sized>(
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

unsafe extern "C" fn obj_grad<T : Sized>(
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
/// back to the user.
pub fn minimize_with_grad<T : Sized>(
    params : DVector<f64>,
    user_data : T,
    fn_update : Box<dyn FnMut(&DVector<f64>, &mut T)->f64>,
    grad_update : Box<dyn FnMut(&DVector<f64>, &mut T)->DVector<f64>>,
    max_iter : usize
) -> Result<(DVector<f64>, DVector<f64>, f64, T), String> {
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
        let mut fdf : gsl_multimin_function_fdf = mem::uninitialized();
        fdf.f = Some(objective::<T>);
        fdf.df = Some(gradient::<T>);
        fdf.fdf = Some(obj_grad::<T>);
        fdf.n = n;
        fdf.params = ((&mut probl) as *mut MinProblem<T>) as *mut ::std::os::raw::c_void;
        let params_gsl : gsl_vector = params.into();
        let mut steps : Vec<OptimizationStep> = Vec::new();
        gsl_multimin_fdfminimizer_set(
            minimizer,
            &mut fdf as *mut _,
            &params_gsl,
            1.0,   // First step size
            0.1    // Tolerance
        );
        for i in 0..max_iter {
            let res = gsl_multimin_fdfminimizer_iterate(minimizer);
            match GSLStatus::from_code(res) {
                GSLStatus::EnoProg => {
                    if i > 1 {
                        let (user_x, user_g, obj) = fetch_data_from_grad_minimizer(&minimizer);
                        println!("Converged in {} iterations", i);
                        return Ok((DVector::from(user_x), DVector::from(user_g), obj, probl.user_data));
                    } else {
                        return Err(format!("Unable to improve estimate (Iteration {})", i));
                    }
                },
                _ => { }
            }
            let (user_x, user_g, obj) = fetch_data_from_grad_minimizer(&minimizer);
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
                    return Ok((DVector::from(user_x), DVector::from(user_g), obj, probl.user_data));
                },
                status => {
                    println!("Steps: {:?}", steps);
                    return Err(format!("Error (Iteration {}): {:?}", i, status));
                }
            }
        }
        println!("Steps: {:?}", steps);
        Err(String::from("Maximum number of iterations reached"))
    }
}

