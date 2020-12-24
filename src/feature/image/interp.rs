use crate::foreign::gsl::spline;
use crate::foreign::gsl::spline2d;
use nalgebra::*;
use nalgebra::base::storage::Storage;

pub struct Interpolation2D {
    _spline2d_type : *const spline2d::gsl_interp2d_type,
    spline : *mut spline2d::gsl_spline2d,
    xacc : *mut spline2d::gsl_interp_accel,
    yacc : *mut spline2d::gsl_interp_accel,
    x_dom : (f64, f64),
    y_dom : (f64, f64),
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
        // println!("z data: {:?}", z);
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
            // println!("X Buffer : {:?}", x_buf);
            // println!("Y Buffer : {:?}", y_buf);
            //println!("{:?}", (x_step, y_step));
            let mut z_buf = vec![0.0; (x_density + 1) * (y_density + 1)];
            for ((x, y), z) in x.iter().zip(y.iter()).zip(z.iter()) {
                // println!("Coords : {}, {}", x, y);
                let x_pos = ( ((*x /*- x_dom.0*/ ) / x_step).floor() as i32 ) as usize;
                let y_pos = ( ((*y /*- y_dom.0*/ ) / y_step).floor() as i32 ) as usize;
                // println!("{} x {}", x_pos, y_pos);
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
            // println!("interp initialized");
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
        // println!("x : {}, (Domain {:?})", x, self.x_dom);
        // println!("y : {}, (Domain {:?})", y, self.y_dom);
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

