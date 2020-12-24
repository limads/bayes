use crate::foreign::gsl::spline;
use crate::foreign::gsl::spline2d;
use nalgebra::*;
use nalgebra::base::storage::Storage;

pub enum Modality {
    Linear,
    Polynomial,
    CSpline
}

pub struct Interpolation {
    acc : *mut spline::gsl_interp_accel,
    modality : Modality,
    interp_type : *const spline::gsl_interp_type
}

impl Interpolation {

    pub fn new(modality : Modality) -> Self {
        unsafe {
            let acc = spline::gsl_interp_accel_alloc();
            let interp_type = match modality {
                Modality::Linear => spline::gsl_interp_linear,
                Modality::Polynomial => spline::gsl_interp_cspline,
                Modality::CSpline => spline::gsl_interp_polynomial
            };
            Self{ acc, modality, interp_type }
        }
    }

    pub fn interpolate(&mut self, x : &DVector<f64>, y : &DVector<f64>, n : usize) -> (DVector<f64>, DVector<f64>) {
        unsafe {
            let spline = spline::gsl_spline_alloc(self.interp_type, x.nrows());
            spline::gsl_spline_init (spline, &x[0] as *const f64, &y[0] as *const f64, x.nrows());
            let (x_compl, out) = interpolate(spline, self.acc, x.data.as_slice(), y.data.as_slice(), n).unwrap();
            spline::gsl_spline_free(spline);
            (DVector::from_vec(x_compl), DVector::from_vec(out))
        }
    }

    /*fn interpolate(&mut self, y : CsVector) -> DVector<f64> {
        let domain : Vec<f64> = (0..y.nrows()).map(|ix| ix as f64).collect();
    }*/

}

impl Drop for Interpolation {

    fn drop(&mut self) {
        unsafe {
            spline::gsl_interp_accel_free(self.acc);
        }
    }

}

fn define_step(domain : &[f64], n : usize) -> f64 {
    let dom_start = domain[0];
    let dom_end = domain[domain.len()-1];
    (dom_end - dom_start).abs() / n as f64
}

fn interpolate(
    spline : *mut spline::gsl_spline,
    acc : *mut spline::gsl_interp_accel,
    x : &[f64],
    y : &[f64],
    n : usize
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    if x.len() != y.len() {
        return Err("Incompatible lengths");
    }
    if x.len() < 2 {
        return Err("Invalid length");
    }
    let step = define_step(x, n);
    let mut x_compl = Vec::new();
    let mut interp = Vec::new();
    unsafe {
        for i in 0..n {
            x_compl.push(x[0] + step * (i as f64));
            interp.push(spline::gsl_spline_eval(spline, x_compl[i], acc));
        }
        Ok((x_compl, interp))
    }
}

/*fn interpolate_auto(y : &[f64], n : usize) -> Result<Vec<f64>, &'static str> {
    let mut x = Vec::new();
    for i in 0..y.len() {
        x.push(i as f64)
    };
    interpolate(&x[..], y, n)
}*/

fn define_homogeneous_domain(from : f64, to : f64, n : usize) -> Vec<f64> {
    let step = (to - from) / n as f64;
    let mut domain = Vec::new();
    for i in 0..(n+1) {
        domain.push(from + step*(i as f64));
    }
    domain
}

/// z must be a row-major matrix with dimensions (y.len() by x.len())
fn interpolate2d(
    x : &[f64],
    y : &[f64],
    z : &[f64],
    nx : usize,
    ny : usize,
    x_domain : (f64, f64),
    y_domain : (f64, f64)
) -> Result<Vec<Vec<f64>>, &'static str> {
    if x.len() != y.len() {
        return Err("Incompatible (x,y) domain lengths");
    }
    if x.len() < 2 {
        return Err("(x, y) domain lengths too small");
    }
    if z.len() != x.len() * y.len() {
        return Err("Invalid height dimension");
    }
    unsafe {
        let spline2d_type = spline2d::gsl_interp2d_bilinear;
        let spline = spline2d::gsl_spline2d_alloc(spline2d_type, x.len(), y.len());
        let xacc = spline2d::gsl_interp_accel_alloc();
        let yacc = spline2d::gsl_interp_accel_alloc();
        spline2d::gsl_spline2d_init(spline, &x[0] as *const f64, &y[0] as *const f64, &z[0] as *const f64, x.len(), y.len());
        let xvals = define_homogeneous_domain(x_domain.0, x_domain.1, nx);
        let yvals = define_homogeneous_domain(y_domain.0, y_domain.1, ny);
        let mut ans = Vec::new();
        for yv in yvals.iter() {
            let mut row = Vec::new();
            for xv in xvals.iter() {
                row.push(spline2d::gsl_spline2d_eval(spline, *xv, *yv, xacc, yacc));
            }
            ans.push(row);
        }
        Ok(ans)
    }
}

