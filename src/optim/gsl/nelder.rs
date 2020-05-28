use crate::gsl::multimin::*;
use nalgebra::*;
use crate::gsl::vector_double::*;
use crate::gsl::matrix_double::*;
use std::ffi::c_void;
use std::mem;
use crate::gsl::utils::*;
use super::*;

unsafe fn fetch_data_from_minimizer(
    minimizer : & *mut gsl_multimin_fminimizer
) -> (DVector<f64>, f64) {
    let x = gsl_multimin_fminimizer_x(*minimizer);
    let user_x = DVectorSlice::<f64>::slice_from_gsl(&*x);
    let end_param : DVector<f64> = user_x.into();
    let minimum = gsl_multimin_fminimizer_minimum(*minimizer);
    (end_param, minimum)
}

pub fn minimize<T : Sized>(
    params : DVector<f64>,
    user_data : T,
    fn_update : Box<dyn FnMut(&DVector<f64>, &mut T)->f64>,
    max_iter : usize
) -> Result<(DVector<f64>, f64), String> {
    let n = params.nrows();
    let mut probl = MinProblem {
        user_data,
        user_params : params.clone(),
        user_grad : DVector::<f64>::from_element(n, 0.1),
        fn_update,
        grad_update : None
    };
    unsafe {
        let minimizer = gsl_multimin_fminimizer_alloc(
            gsl_multimin_fminimizer_nmsimplex,
            n
        );
        let mut mf : gsl_multimin_function = mem::uninitialized();
        mf.f = Some(objective::<T>);
        mf.n = n;
        mf.params = ((&mut probl) as *mut MinProblem<T>) as *mut ::std::os::raw::c_void;
        let params_gsl : gsl_vector = params.clone().into();
        let step_sz_gsl : gsl_vector = params.into();
        gsl_multimin_fminimizer_set(
            minimizer,
            &mut mf as *mut _,
            &params_gsl,
            &step_sz_gsl
        );
        for i in 0..max_iter {
            let res = gsl_multimin_fminimizer_iterate(minimizer);
            match GSLStatus::from_code(res) {
                GSLStatus::EnoProg => {
                    if i > 1 {
                        println!("Converged in {} iterations", i);
                        let (end_param, min) = fetch_data_from_minimizer(&minimizer);
                        return Ok((end_param, min))
                    } else {
                        return Err(format!("Unable to improve estimate (Iteration {})", i));
                    }
                },
                _ => { }
            }
            let size = gsl_multimin_fminimizer_size(minimizer);
            match GSLStatus::from_code(gsl_multimin_test_size(0.001, size)) {
                GSLStatus::Continue => { },
                GSLStatus::Success => {
                    println!("Converged in {} iterations", i);
                    return Ok(fetch_data_from_minimizer(&minimizer));
                },
                status => {
                    return Err(format!("Error (Iteration {}): {:?}", i, status));
                }
            }
        }
    }
    Err(String::from("Maximum number of iterations reached"))
}


