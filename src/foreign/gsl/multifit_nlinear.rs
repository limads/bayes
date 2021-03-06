use crate::foreign::gsl::vector_double::*;
use crate::foreign::gsl::matrix_double::*;
use crate::foreign::gsl::block_double::*;

/* automatically generated by rust-bindgen */

pub const gsl_multifit_nlinear_fdtype_GSL_MULTIFIT_NLINEAR_FWDIFF: gsl_multifit_nlinear_fdtype = 0;
pub const gsl_multifit_nlinear_fdtype_GSL_MULTIFIT_NLINEAR_CTRDIFF: gsl_multifit_nlinear_fdtype = 1;
pub type gsl_multifit_nlinear_fdtype = u32;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_multifit_nlinear_fdf {
    pub f: ::std::option::Option<
        unsafe extern "C" fn(x: *const gsl_vector,
                             params: *mut ::std::os::raw::c_void,
                             f: *mut gsl_vector)
                             -> ::std::os::raw::c_int,
    >,
    pub df: ::std::option::Option<
        unsafe extern "C" fn(x: *const gsl_vector,
                             params: *mut ::std::os::raw::c_void,
                             df: *mut gsl_matrix)
                             -> ::std::os::raw::c_int,
    >,
    pub fvv: ::std::option::Option<
        unsafe extern "C" fn(x: *const gsl_vector,
                             v: *const gsl_vector,
                             params: *mut ::std::os::raw::c_void,
                             fvv: *mut gsl_vector)
                             -> ::std::os::raw::c_int,
    >,
    pub n: usize,
    pub p: usize,
    pub params: *mut ::std::os::raw::c_void,
    pub nevalf: usize,
    pub nevaldf: usize,
    pub nevalfvv: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_multifit_nlinear_trs {
    pub name: *const ::std::os::raw::c_char,
    pub alloc: ::std::option::Option<
        unsafe extern "C" fn(params: *const ::std::os::raw::c_void,
                             n: usize,
                             p: usize)
                             -> *mut ::std::os::raw::c_void,
    >,
    pub init:
        ::std::option::Option<
            unsafe extern "C" fn(vtrust_state: *const ::std::os::raw::c_void,
                                 vstate: *mut ::std::os::raw::c_void)
                                 -> ::std::os::raw::c_int,
        >,
    pub preloop:
        ::std::option::Option<
            unsafe extern "C" fn(vtrust_state: *const ::std::os::raw::c_void,
                                 vstate: *mut ::std::os::raw::c_void)
                                 -> ::std::os::raw::c_int,
        >,
    pub step:
        ::std::option::Option<
            unsafe extern "C" fn(vtrust_state: *const ::std::os::raw::c_void,
                                 delta: f64,
                                 dx: *mut gsl_vector,
                                 vstate: *mut ::std::os::raw::c_void)
                                 -> ::std::os::raw::c_int,
        >,
    pub preduction:
        ::std::option::Option<
            unsafe extern "C" fn(vtrust_state: *const ::std::os::raw::c_void,
                                 dx: *const gsl_vector,
                                 pred: *mut f64,
                                 vstate: *mut ::std::os::raw::c_void)
                                 -> ::std::os::raw::c_int,
        >,
    pub free: ::std::option::Option<unsafe extern "C" fn(vstate: *mut ::std::os::raw::c_void)>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_multifit_nlinear_scale {
    pub name: *const ::std::os::raw::c_char,
    pub init: ::std::option::Option<
        unsafe extern "C" fn(J: *const gsl_matrix,
                             diag: *mut gsl_vector)
                             -> ::std::os::raw::c_int,
    >,
    pub update: ::std::option::Option<
        unsafe extern "C" fn(J: *const gsl_matrix,
                             diag: *mut gsl_vector)
                             -> ::std::os::raw::c_int,
    >,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_multifit_nlinear_solver {
    pub name: *const ::std::os::raw::c_char,
    pub alloc: ::std::option::Option<
        unsafe extern "C" fn(n: usize, p: usize)
                             -> *mut ::std::os::raw::c_void,
    >,
    pub init:
        ::std::option::Option<
            unsafe extern "C" fn(vtrust_state: *const ::std::os::raw::c_void,
                                 vstate: *mut ::std::os::raw::c_void)
                                 -> ::std::os::raw::c_int,
        >,
    pub presolve:
        ::std::option::Option<
            unsafe extern "C" fn(mu: f64,
                                 vtrust_state: *const ::std::os::raw::c_void,
                                 vstate: *mut ::std::os::raw::c_void)
                                 -> ::std::os::raw::c_int,
        >,
    pub solve:
        ::std::option::Option<
            unsafe extern "C" fn(f: *const gsl_vector,
                                 x: *mut gsl_vector,
                                 vtrust_state: *const ::std::os::raw::c_void,
                                 vstate: *mut ::std::os::raw::c_void)
                                 -> ::std::os::raw::c_int,
        >,
    pub rcond: ::std::option::Option<
        unsafe extern "C" fn(rcond: *mut f64,
                             vstate: *mut ::std::os::raw::c_void)
                             -> ::std::os::raw::c_int,
    >,
    pub free: ::std::option::Option<unsafe extern "C" fn(vstate: *mut ::std::os::raw::c_void)>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_multifit_nlinear_parameters {
    pub trs: *const gsl_multifit_nlinear_trs,
    pub scale: *const gsl_multifit_nlinear_scale,
    pub solver: *const gsl_multifit_nlinear_solver,
    pub fdtype: gsl_multifit_nlinear_fdtype,
    pub factor_up: f64,
    pub factor_down: f64,
    pub avmax: f64,
    pub h_df: f64,
    pub h_fvv: f64,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_multifit_nlinear_type {
    pub name: *const ::std::os::raw::c_char,
    pub alloc:
        ::std::option::Option<
            unsafe extern "C" fn(params: *const gsl_multifit_nlinear_parameters,
                                 n: usize,
                                 p: usize)
                                 -> *mut ::std::os::raw::c_void,
        >,
    pub init: ::std::option::Option<
        unsafe extern "C" fn(state: *mut ::std::os::raw::c_void,
                             wts: *const gsl_vector,
                             fdf: *mut gsl_multifit_nlinear_fdf,
                             x: *const gsl_vector,
                             f: *mut gsl_vector,
                             J: *mut gsl_matrix,
                             g: *mut gsl_vector)
                             -> ::std::os::raw::c_int,
    >,
    pub iterate: ::std::option::Option<
        unsafe extern "C" fn(state: *mut ::std::os::raw::c_void,
                             wts: *const gsl_vector,
                             fdf: *mut gsl_multifit_nlinear_fdf,
                             x: *mut gsl_vector,
                             f: *mut gsl_vector,
                             J: *mut gsl_matrix,
                             g: *mut gsl_vector,
                             dx: *mut gsl_vector)
                             -> ::std::os::raw::c_int,
    >,
    pub rcond: ::std::option::Option<
        unsafe extern "C" fn(rcond: *mut f64,
                             state: *mut ::std::os::raw::c_void)
                             -> ::std::os::raw::c_int,
    >,
    pub avratio:
        ::std::option::Option<unsafe extern "C" fn(state: *mut ::std::os::raw::c_void) -> f64>,
    pub free: ::std::option::Option<unsafe extern "C" fn(state: *mut ::std::os::raw::c_void)>,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_multifit_nlinear_trust_state {
    pub x: *const gsl_vector,
    pub f: *const gsl_vector,
    pub g: *const gsl_vector,
    pub J: *const gsl_matrix,
    pub diag: *const gsl_vector,
    pub sqrt_wts: *const gsl_vector,
    pub mu: *const f64,
    pub params: *const gsl_multifit_nlinear_parameters,
    pub solver_state: *mut ::std::os::raw::c_void,
    pub fdf: *mut gsl_multifit_nlinear_fdf,
    pub avratio: *mut f64,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_multifit_nlinear_workspace {
    pub type_: *const gsl_multifit_nlinear_type,
    pub fdf: *mut gsl_multifit_nlinear_fdf,
    pub x: *mut gsl_vector,
    pub f: *mut gsl_vector,
    pub dx: *mut gsl_vector,
    pub g: *mut gsl_vector,
    pub J: *mut gsl_matrix,
    pub sqrt_wts_work: *mut gsl_vector,
    pub sqrt_wts: *mut gsl_vector,
    pub niter: usize,
    pub params: gsl_multifit_nlinear_parameters,
    pub state: *mut ::std::os::raw::c_void,
}
extern "C" {
    pub fn gsl_multifit_nlinear_alloc(
        T: *const gsl_multifit_nlinear_type,
        params: *const gsl_multifit_nlinear_parameters,
        n: usize,
        p: usize,
    ) -> *mut gsl_multifit_nlinear_workspace;
}
extern "C" {
    pub fn gsl_multifit_nlinear_free(w: *mut gsl_multifit_nlinear_workspace);
}
extern "C" {
    pub fn gsl_multifit_nlinear_default_parameters() -> gsl_multifit_nlinear_parameters;
}
extern "C" {
    pub fn gsl_multifit_nlinear_init(
        x: *const gsl_vector,
        fdf: *mut gsl_multifit_nlinear_fdf,
        w: *mut gsl_multifit_nlinear_workspace,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_winit(
        x: *const gsl_vector,
        wts: *const gsl_vector,
        fdf: *mut gsl_multifit_nlinear_fdf,
        w: *mut gsl_multifit_nlinear_workspace,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_iterate(
        w: *mut gsl_multifit_nlinear_workspace,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_avratio(w: *const gsl_multifit_nlinear_workspace) -> f64;
}

extern "C" {
    pub fn gsl_multifit_nlinear_jac(w: *const gsl_multifit_nlinear_workspace) -> *mut gsl_matrix;
}
extern "C" {
    pub fn gsl_multifit_nlinear_name(
        w: *const gsl_multifit_nlinear_workspace,
    ) -> *const ::std::os::raw::c_char;
}
extern "C" {
    pub fn gsl_multifit_nlinear_position(
        w: *const gsl_multifit_nlinear_workspace,
    ) -> *mut gsl_vector;
}
extern "C" {
    pub fn gsl_multifit_nlinear_residual(
        w: *const gsl_multifit_nlinear_workspace,
    ) -> *mut gsl_vector;
}
extern "C" {
    pub fn gsl_multifit_nlinear_niter(w: *const gsl_multifit_nlinear_workspace) -> usize;
}
extern "C" {
    pub fn gsl_multifit_nlinear_rcond(
        rcond: *mut f64,
        w: *const gsl_multifit_nlinear_workspace,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_trs_name(
        w: *const gsl_multifit_nlinear_workspace,
    ) -> *const ::std::os::raw::c_char;
}
extern "C" {
    pub fn gsl_multifit_nlinear_eval_f(
        fdf: *mut gsl_multifit_nlinear_fdf,
        x: *const gsl_vector,
        swts: *const gsl_vector,
        y: *mut gsl_vector,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_eval_df(
        x: *const gsl_vector,
        f: *const gsl_vector,
        swts: *const gsl_vector,
        h: f64,
        fdtype: gsl_multifit_nlinear_fdtype,
        fdf: *mut gsl_multifit_nlinear_fdf,
        df: *mut gsl_matrix,
        work: *mut gsl_vector,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_eval_fvv(
        h: f64,
        x: *const gsl_vector,
        v: *const gsl_vector,
        f: *const gsl_vector,
        J: *const gsl_matrix,
        swts: *const gsl_vector,
        fdf: *mut gsl_multifit_nlinear_fdf,
        yvv: *mut gsl_vector,
        work: *mut gsl_vector,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_covar(
        J: *const gsl_matrix,
        epsrel: f64,
        covar: *mut gsl_matrix,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_test(
        xtol: f64,
        gtol: f64,
        ftol: f64,
        info: *mut ::std::os::raw::c_int,
        w: *const gsl_multifit_nlinear_workspace,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_df(
        h: f64,
        fdtype: gsl_multifit_nlinear_fdtype,
        x: *const gsl_vector,
        wts: *const gsl_vector,
        fdf: *mut gsl_multifit_nlinear_fdf,
        f: *const gsl_vector,
        J: *mut gsl_matrix,
        work: *mut gsl_vector,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_multifit_nlinear_fdfvv(
        h: f64,
        x: *const gsl_vector,
        v: *const gsl_vector,
        f: *const gsl_vector,
        J: *const gsl_matrix,
        swts: *const gsl_vector,
        fdf: *mut gsl_multifit_nlinear_fdf,
        fvv: *mut gsl_vector,
        work: *mut gsl_vector,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_trust: *const gsl_multifit_nlinear_type;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_trs_lm: *const gsl_multifit_nlinear_trs;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_trs_lmaccel: *const gsl_multifit_nlinear_trs;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_trs_dogleg: *const gsl_multifit_nlinear_trs;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_trs_ddogleg: *const gsl_multifit_nlinear_trs;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_trs_subspace2D: *const gsl_multifit_nlinear_trs;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_scale_levenberg: *const gsl_multifit_nlinear_scale;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_scale_marquardt: *const gsl_multifit_nlinear_scale;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_scale_more: *const gsl_multifit_nlinear_scale;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_solver_cholesky: *const gsl_multifit_nlinear_solver;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_solver_qr: *const gsl_multifit_nlinear_solver;
}
extern "C" {
    pub static mut gsl_multifit_nlinear_solver_svd: *const gsl_multifit_nlinear_solver;
}
