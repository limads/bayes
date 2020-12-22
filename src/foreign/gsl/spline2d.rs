/* automatically generated by rust-bindgen */

#[repr(C)]
pub struct gsl_interp_accel {
    pub cache: usize,
    pub miss_count: usize,
    pub hit_count: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_interp_type {
    pub name: *const ::std::os::raw::c_char,
    pub min_size: ::std::os::raw::c_uint,
    pub alloc:
        ::std::option::Option<unsafe extern "C" fn(size: usize) -> *mut ::std::os::raw::c_void>,
    pub init: ::std::option::Option<
        unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                             xa: *const f64,
                             ya: *const f64,
                             size: usize)
                             -> ::std::os::raw::c_int,
    >,
    pub eval: ::std::option::Option<
        unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                             xa: *const f64,
                             ya: *const f64,
                             size: usize,
                             x: f64,
                             arg2: *mut gsl_interp_accel,
                             y: *mut f64)
                             -> ::std::os::raw::c_int,
    >,
    pub eval_deriv:
        ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                 xa: *const f64,
                                 ya: *const f64,
                                 size: usize,
                                 x: f64,
                                 arg2: *mut gsl_interp_accel,
                                 y_p: *mut f64)
                                 -> ::std::os::raw::c_int,
        >,
    pub eval_deriv2:
        ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                 xa: *const f64,
                                 ya: *const f64,
                                 size: usize,
                                 x: f64,
                                 arg2: *mut gsl_interp_accel,
                                 y_pp: *mut f64)
                                 -> ::std::os::raw::c_int,
        >,
    pub eval_integ:
        ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                 xa: *const f64,
                                 ya: *const f64,
                                 size: usize,
                                 arg2: *mut gsl_interp_accel,
                                 a: f64,
                                 b: f64,
                                 result: *mut f64)
                                 -> ::std::os::raw::c_int,
        >,
    pub free: ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void)>,
}
#[repr(C)]
pub struct gsl_interp {
    pub type_: *const gsl_interp_type,
    pub xmin: f64,
    pub xmax: f64,
    pub size: usize,
    pub state: *mut ::std::os::raw::c_void,
}
extern "C" {
    pub static mut gsl_interp_linear: *const gsl_interp_type;
}
extern "C" {
    pub static mut gsl_interp_polynomial: *const gsl_interp_type;
}
extern "C" {
    pub static mut gsl_interp_cspline: *const gsl_interp_type;
}
extern "C" {
    pub static mut gsl_interp_cspline_periodic: *const gsl_interp_type;
}
extern "C" {
    pub static mut gsl_interp_akima: *const gsl_interp_type;
}
extern "C" {
    pub static mut gsl_interp_akima_periodic: *const gsl_interp_type;
}
extern "C" {
    pub static mut gsl_interp_steffen: *const gsl_interp_type;
}
extern "C" {
    pub fn gsl_interp_accel_alloc() -> *mut gsl_interp_accel;
}
extern "C" {
    pub fn gsl_interp_accel_reset(a: *mut gsl_interp_accel) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp_accel_free(a: *mut gsl_interp_accel);
}
extern "C" {
    pub fn gsl_interp_alloc(T: *const gsl_interp_type, n: usize) -> *mut gsl_interp;
}
extern "C" {
    pub fn gsl_interp_init(
        obj: *mut gsl_interp,
        xa: *const f64,
        ya: *const f64,
        size: usize,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp_name(interp: *const gsl_interp) -> *const ::std::os::raw::c_char;
}
extern "C" {
    pub fn gsl_interp_min_size(interp: *const gsl_interp) -> ::std::os::raw::c_uint;
}
extern "C" {
    pub fn gsl_interp_type_min_size(T: *const gsl_interp_type) -> ::std::os::raw::c_uint;
}
extern "C" {
    pub fn gsl_interp_eval_e(
        obj: *const gsl_interp,
        xa: *const f64,
        ya: *const f64,
        x: f64,
        a: *mut gsl_interp_accel,
        y: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp_eval(
        obj: *const gsl_interp,
        xa: *const f64,
        ya: *const f64,
        x: f64,
        a: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp_eval_deriv_e(
        obj: *const gsl_interp,
        xa: *const f64,
        ya: *const f64,
        x: f64,
        a: *mut gsl_interp_accel,
        d: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp_eval_deriv(
        obj: *const gsl_interp,
        xa: *const f64,
        ya: *const f64,
        x: f64,
        a: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp_eval_deriv2_e(
        obj: *const gsl_interp,
        xa: *const f64,
        ya: *const f64,
        x: f64,
        a: *mut gsl_interp_accel,
        d2: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp_eval_deriv2(
        obj: *const gsl_interp,
        xa: *const f64,
        ya: *const f64,
        x: f64,
        a: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp_eval_integ_e(
        obj: *const gsl_interp,
        xa: *const f64,
        ya: *const f64,
        a: f64,
        b: f64,
        acc: *mut gsl_interp_accel,
        result: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp_eval_integ(
        obj: *const gsl_interp,
        xa: *const f64,
        ya: *const f64,
        a: f64,
        b: f64,
        acc: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp_free(interp: *mut gsl_interp);
}
extern "C" {
    pub fn gsl_interp_bsearch(
        x_array: *const f64,
        x: f64,
        index_lo: usize,
        index_hi: usize,
    ) -> usize;
}
extern "C" {
    pub fn gsl_interp_accel_find(
        a: *mut gsl_interp_accel,
        x_array: *const f64,
        size: usize,
        x: f64,
    ) -> usize;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gsl_interp2d_type {
    pub name: *const ::std::os::raw::c_char,
    pub min_size: ::std::os::raw::c_uint,
    pub alloc: ::std::option::Option<
        unsafe extern "C" fn(xsize: usize, ysize: usize)
                             -> *mut ::std::os::raw::c_void,
    >,
    pub init: ::std::option::Option<
        unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void,
                             xa: *const f64,
                             ya: *const f64,
                             za: *const f64,
                             xsize: usize,
                             ysize: usize)
                             -> ::std::os::raw::c_int,
    >,
    pub eval: ::std::option::Option<
        unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                             xa: *const f64,
                             ya: *const f64,
                             za: *const f64,
                             xsize: usize,
                             ysize: usize,
                             x: f64,
                             y: f64,
                             arg2: *mut gsl_interp_accel,
                             arg3: *mut gsl_interp_accel,
                             z: *mut f64)
                             -> ::std::os::raw::c_int,
    >,
    pub eval_deriv_x:
        ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                 xa: *const f64,
                                 ya: *const f64,
                                 za: *const f64,
                                 xsize: usize,
                                 ysize: usize,
                                 x: f64,
                                 y: f64,
                                 arg2: *mut gsl_interp_accel,
                                 arg3: *mut gsl_interp_accel,
                                 z_p: *mut f64)
                                 -> ::std::os::raw::c_int,
        >,
    pub eval_deriv_y:
        ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                 xa: *const f64,
                                 ya: *const f64,
                                 za: *const f64,
                                 xsize: usize,
                                 ysize: usize,
                                 x: f64,
                                 y: f64,
                                 arg2: *mut gsl_interp_accel,
                                 arg3: *mut gsl_interp_accel,
                                 z_p: *mut f64)
                                 -> ::std::os::raw::c_int,
        >,
    pub eval_deriv_xx:
        ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                 xa: *const f64,
                                 ya: *const f64,
                                 za: *const f64,
                                 xsize: usize,
                                 ysize: usize,
                                 x: f64,
                                 y: f64,
                                 arg2: *mut gsl_interp_accel,
                                 arg3: *mut gsl_interp_accel,
                                 z_pp: *mut f64)
                                 -> ::std::os::raw::c_int,
        >,
    pub eval_deriv_xy:
        ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                 xa: *const f64,
                                 ya: *const f64,
                                 za: *const f64,
                                 xsize: usize,
                                 ysize: usize,
                                 x: f64,
                                 y: f64,
                                 arg2: *mut gsl_interp_accel,
                                 arg3: *mut gsl_interp_accel,
                                 z_pp: *mut f64)
                                 -> ::std::os::raw::c_int,
        >,
    pub eval_deriv_yy:
        ::std::option::Option<
            unsafe extern "C" fn(arg1: *const ::std::os::raw::c_void,
                                 xa: *const f64,
                                 ya: *const f64,
                                 za: *const f64,
                                 xsize: usize,
                                 ysize: usize,
                                 x: f64,
                                 y: f64,
                                 arg2: *mut gsl_interp_accel,
                                 arg3: *mut gsl_interp_accel,
                                 z_pp: *mut f64)
                                 -> ::std::os::raw::c_int,
        >,
    pub free: ::std::option::Option<unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void)>,
}
#[repr(C)]
pub struct gsl_interp2d {
    pub type_: *const gsl_interp2d_type,
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub xsize: usize,
    pub ysize: usize,
    pub state: *mut ::std::os::raw::c_void,
}
extern "C" {
    pub static mut gsl_interp2d_bilinear: *const gsl_interp2d_type;
}
extern "C" {
    pub static mut gsl_interp2d_bicubic: *const gsl_interp2d_type;
}
extern "C" {
    pub fn gsl_interp2d_alloc(
        T: *const gsl_interp2d_type,
        xsize: usize,
        ysize: usize,
    ) -> *mut gsl_interp2d;
}
extern "C" {
    pub fn gsl_interp2d_name(interp: *const gsl_interp2d) -> *const ::std::os::raw::c_char;
}
extern "C" {
    pub fn gsl_interp2d_min_size(interp: *const gsl_interp2d) -> usize;
}
extern "C" {
    pub fn gsl_interp2d_type_min_size(T: *const gsl_interp2d_type) -> usize;
}
extern "C" {
    pub fn gsl_interp2d_set(
        interp: *const gsl_interp2d,
        zarr: *mut f64,
        i: usize,
        j: usize,
        z: f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp2d_get(
        interp: *const gsl_interp2d,
        zarr: *const f64,
        i: usize,
        j: usize,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp2d_idx(interp: *const gsl_interp2d, i: usize, j: usize) -> usize;
}
extern "C" {
    pub fn gsl_interp2d_init(
        interp: *mut gsl_interp2d,
        xa: *const f64,
        ya: *const f64,
        za: *const f64,
        xsize: usize,
        ysize: usize,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp2d_free(interp: *mut gsl_interp2d);
}
extern "C" {
    pub fn gsl_interp2d_eval(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp2d_eval_extrap(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp2d_eval_e(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp2d_eval_e_extrap(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_x(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_x_e(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_y(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_y_e(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_xx(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_xx_e(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_yy(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_yy_e(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_xy(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_interp2d_eval_deriv_xy_e(
        interp: *const gsl_interp2d,
        xarr: *const f64,
        yarr: *const f64,
        zarr: *const f64,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
#[repr(C)]
pub struct gsl_spline2d {
    pub interp_object: gsl_interp2d,
    pub xarr: *mut f64,
    pub yarr: *mut f64,
    pub zarr: *mut f64,
}
extern "C" {
    pub fn gsl_spline2d_alloc(
        T: *const gsl_interp2d_type,
        xsize: usize,
        ysize: usize,
    ) -> *mut gsl_spline2d;
}
extern "C" {
    pub fn gsl_spline2d_init(
        interp: *mut gsl_spline2d,
        xa: *const f64,
        ya: *const f64,
        za: *const f64,
        xsize: usize,
        ysize: usize,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_spline2d_free(interp: *mut gsl_spline2d);
}
extern "C" {
    pub fn gsl_spline2d_eval(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_spline2d_eval_e(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_x(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_x_e(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_y(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_y_e(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_xx(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_xx_e(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_yy(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_yy_e(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_xy(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
    ) -> f64;
}
extern "C" {
    pub fn gsl_spline2d_eval_deriv_xy_e(
        interp: *const gsl_spline2d,
        x: f64,
        y: f64,
        xa: *mut gsl_interp_accel,
        ya: *mut gsl_interp_accel,
        z: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_spline2d_min_size(interp: *const gsl_spline2d) -> usize;
}
extern "C" {
    pub fn gsl_spline2d_name(interp: *const gsl_spline2d) -> *const ::std::os::raw::c_char;
}
extern "C" {
    pub fn gsl_spline2d_set(
        interp: *const gsl_spline2d,
        zarr: *mut f64,
        i: usize,
        j: usize,
        z: f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn gsl_spline2d_get(
        interp: *const gsl_spline2d,
        zarr: *const f64,
        i: usize,
        j: usize,
    ) -> f64;
}