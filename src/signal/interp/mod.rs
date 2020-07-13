use crate::gsl::spline;
use crate::gsl::spline2d;
//use nagebra::sparse::{CsMatrix, CsVector};
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

    pub fn new(modality : Modality, n : usize) -> Self {
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

    fn interpolate(&mut self, x : &DVector<f64>, y : &DVector<f64>, n : usize) -> DVector<f64> {
        unsafe {
            let spline = spline::gsl_spline_alloc(self.interp_type, x.nrows());
            spline::gsl_spline_init (spline, &x[0] as *const f64, &y[0] as *const f64, x.nrows());
            let out = interpolate(spline, self.acc, x.data.as_slice(), y.data.as_slice(), n).unwrap();
            spline::gsl_spline_free(spline);
            DVector::from_vec(out)
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
) -> Result<Vec<f64>, &'static str> {
    if x.len() != y.len() {
        return Err("Incompatible lengths");
    }
    if x.len() < 2 {
        return Err("Invalid length");
    }
    unsafe {
        let step = define_step(x, n);
        let mut interp = Vec::new();
        for i in 0..n {
            interp.push(spline::gsl_spline_eval(spline, x[0] + step * (i as f64), acc));
        }
        Ok(interp)
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

pub struct Interpolation2D {
    _spline2d_type : *const spline2d::gsl_interp2d_type,
    spline : *mut spline2d::gsl_spline2d,
    xacc : *mut spline2d::gsl_interp_accel,
    yacc : *mut spline2d::gsl_interp_accel,
    x_dom : (f64, f64),
    y_dom : (f64, f64),
//    x_buffer : Vec<f64>,
//    y_buffer : Vec<f64>,
    _z_buf : Vec<f64>
}

impl Interpolation2D {

    pub fn new(
        x_dom : (f64, f64),
        y_dom : (f64, f64),
        x_density : usize,
        y_density : usize,
        x : &[f64],
        y : &[f64],
        z : &[f64]
    ) -> Result<Self, &'static str> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err("Incompatible (x,y,z) domain lengths");
        }
        if x.len() < 2 {
            return Err("(x, y, z) domain lengths too small");
        }
        // We might want to interpolate values right at the edge, so we add a security measure.
        let x_dom = (x_dom.0 - 1., x_dom.1 + 1.);
        let y_dom = (y_dom.0 - 1., y_dom.1 + 1.);
        println!("z data: {:?}", z);
        unsafe {
            let spline2d_type = spline2d::gsl_interp2d_bilinear;
            let spline = spline2d::gsl_spline2d_alloc(spline2d_type, x_density + 1, y_density + 1);
            let xacc = spline2d::gsl_interp_accel_alloc();
            let yacc = spline2d::gsl_interp_accel_alloc();
            let x_dom_ext = (x_dom.1 - x_dom.0).abs();
            let y_dom_ext = (y_dom.1 - y_dom.0).abs();
            // println!("{:?}", y_dom);
            let x_step = x_dom_ext / x_density as f64;
            let y_step = y_dom_ext / y_density as f64;
            let x_buf : Vec<_>= (0..(x_density+1)).map(|i| x_dom.0+(i as f64)*x_step ).collect();
            let y_buf : Vec<_>= (0..(y_density+1)).map(|i| y_dom.0+(i as f64)*y_step ).collect();
            println!("X Buffer : {:?}", x_buf);
            println!("Y Buffer : {:?}", y_buf);
            //println!("{:?}", (x_step, y_step));
            let mut z_buf = vec![0.0; (x_density + 1) * (y_density + 1)];
            for ((x, y), z) in x.iter().zip(y.iter()).zip(z.iter()) {
                //println!("Coords : {}, {}", x, y);
                let x_pos = ( ((*x /*- x_dom.0*/ ) / x_step).floor() as i32 ) as usize;
                let y_pos = ( ((*y /*- y_dom.0*/ ) / y_step).floor() as i32 ) as usize;
                println!("{} x {}", x_pos, y_pos);
                spline2d::gsl_spline2d_set(spline, &mut z_buf[0] as *mut f64, x_pos, y_pos, *z);
            }
            spline2d::gsl_spline2d_init(
                spline,
                &x_buf[0] as *const f64,
                &y_buf[0] as *const f64,
                &z_buf[0] as *const f64,
                x_density + 1,
                y_density + 1
            );
            println!("interp initialized");

            Ok(Self {
                _spline2d_type : spline2d_type,
                spline,
                xacc,
                yacc,
                _z_buf : z_buf,
                x_dom,
                y_dom
            })
        }
    }

    pub fn interpolate_point(&self, x : f64, y : f64) -> f64 {
        println!("x : {}, (Domain {:?})", x, self.x_dom);
        println!("y : {}, (Domain {:?})", y, self.y_dom);
        if x > self.x_dom.0 && x < self.x_dom.1 {
            if y > self.y_dom.0 && y < self.y_dom.1 {
                unsafe {
                    spline2d::gsl_spline2d_eval(self.spline, x, y, self.xacc, self.yacc)
                }
            } else {
                panic!("Tried to plot y point {} but y scale is limited to {}-{}", y, self.y_dom.0, self.y_dom.1);
            }
        } else {
            panic!("Tried to plot x point {} but x scale is limited to {}-{}", x, self.x_dom.0, self.x_dom.1);
        }

    }

}


