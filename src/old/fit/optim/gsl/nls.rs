use crate::foreign::gsl::vector_double::*;
use crate::foreign::gsl::matrix_double::*;
use std::mem::{self};
// use std::ptr;
use std::boxed::Box;
use nalgebra::*;
use crate::foreign::gsl::multifit_nlinear::*;
// use crate::gsl::utils::*;

//use crate::gsl::errno::gsl_error;
//extern "C" {
// Copied from gsl_matrix_double.h
//    pub fn gsl_matrix_alloc(n1: usize, n2: usize) -> *mut gsl_matrix;
//}

/*pub fn gsl_error(err : i32) -> &'static str {
    match err {

    }
}*/

const GSL_SUCCESS : i32 = 0;

const GSL_CONTINUE : i32 = -2;

/// Perhaps use the same fit(.) function, but passing None
/// to the Jacobian parameter dispatches to a linear least squares
/// problem, passing Some(j) dispatches to a non-linear least squares
/// problem.

/// C function which maps vector x of p parameters (updated at each step) constant across samples
/// into n random outcomes, potentially affected by a generic user-defined
/// arbitrary parameter structure params.
unsafe extern "C" fn f<T : Sized>(
    x: *const gsl_vector,
    params: *mut ::std::os::raw::c_void,
    f: *mut gsl_vector
) -> ::std::os::raw::c_int {
    let extra_ptr : *mut NLSProblem<T> = mem::transmute(params);
    let extra = &*extra_ptr;
    let user_x : DVector<f64> = (*x).clone().into();
    let user_f = (extra.fn_update)(&user_x, &extra.user_params);
    let updated_f : gsl_vector = user_f.into();
    *f = updated_f;
    //println!("f = {}", *((*f).data));
    0
}

/// C function which maps vector x of p parameters (updated at each step) constant across samples
/// and potentially by a generic user-defined structure params into a n x p
/// jacobian df that drives parameter updates.
unsafe extern "C" fn df<T : Sized>(
    x: *const gsl_vector,
    params: *mut ::std::os::raw::c_void,
    df: *mut gsl_matrix
) -> ::std::os::raw::c_int {
    let extra_ptr : *mut NLSProblem<T> = mem::transmute(params);
    let extra = &*extra_ptr;
    let user_x : DVector<f64> = (*x).clone().into();
    let user_jacobian = (extra.jacobian_update)(&user_x, &extra.user_params);
    let updated_j : gsl_matrix = user_jacobian.into();
    *df = updated_j;
    //println!("df = {}", *((*df).data));
    0
}

#[repr(C)]
pub struct NLSProblem<T : Sized> {

    // User-informed extra parameters, constant across calls
    user_params : T,

    user_beta : DVector<f64>,

    user_jacobian : DMatrix<f64>,

    /// Return evaluation of all N samples based on beta values given in
    /// the first argument and arbitrary data given on the second argument.
    fn_update : Box<dyn Fn(&DVector<f64>, &T)->DVector<f64>>,

    /// Return evaluation of N samples x P parameters Jacobian
    /// given beta values in the first argument and arbitrary data given on the
    /// second argument.
    jacobian_update  : Box<dyn Fn(&DVector<f64>, &T)->DMatrix<f64>>
}

pub struct NLSResult {
    pub beta : DVector<f64>,
    pub cov : DMatrix<f64>,
    pub resid : DVector<f64>,
    pub niter : usize
}

unsafe fn get_nls_result(
    nl_ws : *mut gsl_multifit_nlinear_workspace,
    p : usize,
    niter : usize
) -> Option<NLSResult> {
    let resid_gsl : *mut gsl_vector = gsl_multifit_nlinear_residual(
        nl_ws
    );
    let jac_gsl = gsl_multifit_nlinear_jac(nl_ws);
    let cov_gsl = gsl_matrix_alloc(p, p);
    let covar_ans = gsl_multifit_nlinear_covar(
        jac_gsl,
        0.001,
        cov_gsl
    );
    if covar_ans != 0 {
        println!("Error recovering covariance matrix: {}", covar_ans);
        return None;
    }
    let beta_gsl = gsl_multifit_nlinear_position(nl_ws);
    let cov : DMatrix<f64> = (*cov_gsl).clone().into();
    let beta : DVector<f64> = (*beta_gsl).clone().into();
    let resid : DVector<f64> = (*resid_gsl).clone().into();
    Some(NLSResult{
        beta,
        cov,
        resid,
        niter
    })
}

/// Call NLS routine, generic over the type of the extra parameter T.
/// fn_update must return a vector of predicted function values given a vector
/// of parameters; jacobian_update must return a new Jacobian matrix given a
/// vector of parameters. It is through this jacobian return function that
/// the errors of measured values must inform about function updates.
pub fn minimize_nls<T : Sized>(
    beta_init : DVector<f64>,
    extra : T,
    n : usize,
    fn_update : Box<dyn Fn(&DVector<f64>, &T)->DVector<f64>>,
    jacobian_update : Box<dyn Fn(&DVector<f64>, &T)->DMatrix<f64>>,
    weights : Option<DVector<f64>>,
    max_iter : usize
) -> Option<NLSResult> {
    let p = beta_init.nrows();
    let init_jacobian = DMatrix::<f64>::from_element(n, p, 0.01);
    let weights = weights.unwrap_or(DVector::<f64>::from_element(n, 1.));
    let mut nls_problem = NLSProblem{
        user_params : extra,
        user_beta : beta_init.clone(),
        user_jacobian : init_jacobian,
        fn_update : fn_update,
        jacobian_update : jacobian_update
    };
    unsafe {
        let ntype = gsl_multifit_nlinear_trust;
        let mut nls_params = gsl_multifit_nlinear_default_parameters();
        // Number of non-linear functions
        let nl_ws = gsl_multifit_nlinear_alloc(
            ntype,
            &mut nls_params as *mut gsl_multifit_nlinear_parameters,
            n,
            p
        );
        //TODO make sure GSL is not relying on aliased use of user x and library x.
        let x : gsl_vector = nls_problem.user_beta.clone().into();
        let mut fdf = gsl_multifit_nlinear_fdf{
            f : Some(f::<T>),
            df : Some(df::<T>),
            fvv : None, // or Some(ptr::null)
            n : n,
            p : p,
            params : ((&mut nls_problem) as *mut NLSProblem<T>) as *mut ::std::os::raw::c_void,
            nevaldf : 0,
            nevalf : 0,
            nevalfvv : 0
        };
        let weights_gsl : gsl_vector = weights.into();
        let ans = gsl_multifit_nlinear_winit(
            &x as *const gsl_vector,
            &weights_gsl as *const gsl_vector,
            &mut fdf as *mut gsl_multifit_nlinear_fdf,
            nl_ws
        );
        if ans != 0 {
            println!("Error initializing NLS workspace: {}", ans);
            return None;
        }
        let _test_ans : i32 = GSL_CONTINUE;
        // This is the parameter tolerance
        let xtol = (10.0).powf(-1.0*(p as f64));
        let gtol = 0.0001; //GSL_DBL_EPSILON.powf(1.0 / 3.0);
        let ftol = 0.00001;
        let mut niter : usize = 1;
        let mut info : i32 = -1;
        while niter <= max_iter {
            let iter_ans = gsl_multifit_nlinear_iterate(nl_ws);
            //println!("Iter ans = {}", iter_ans);
            if iter_ans != GSL_SUCCESS {
                println!("Error at NLS iterate call (Iteration {}): GSL Error {}", niter, iter_ans);
                if iter_ans == 2 {
                    println!("Tried to iterate with NaN values");
                }
                if iter_ans == 27 {
                    println!("Iteration is not making progress towards solution");
                }

                /*let reason = CString::new("").expect("CString::new failed");
                let file = CString::new("").expect("CString::new failed");
                let line = CString::new("").expect("CString::new failed");
                gsl_error(
                    reason.into_raw(),
                    file.into_raw(),
                    line.into_raw(),
                    gsl_errno: ::std::os::raw::c_int,
                );*/
                return None;
            }
            let test_ans = gsl_multifit_nlinear_test(
                xtol,
                gtol,
                ftol,
                &mut info as *mut i32,
                nl_ws
            );
            //if test_ans != GSL_SUCCESS {
            //    println!("Testing for test convergence failed: {}", test_ans);
            //    return None;
            //}
            //println!("test_ans = {}", test_ans);
            match test_ans {
                GSL_CONTINUE => { niter += 1; }
                GSL_SUCCESS => {
                    return get_nls_result(nl_ws, p, niter);
                },
                _ => {
                    println!("Error at NLS convergence check (Iteration {}). GSL Result {}; GSL Info {}", niter, test_ans, info);
                    return None;
                }
            }
        }
        println!("Error : Maximum number of iterations reached ({})", niter);
        None
    }
}


