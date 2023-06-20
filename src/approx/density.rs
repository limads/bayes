use std::f64::consts::PI;

/* Each variant carries its bandwith parameter h
https://scikit-learn.org/stable/modules/density.html */
#[derive(Clone, Copy)]
pub enum Kernel {

    Gaussian(f64),

    TopHat(f64),

    Epanechnikov(f64),

    Exponential(f64),

    Linear(f64),

    Cosine(f64)

}

/* The KernelCall encapsulates the currying of the bandwidth parameter
from the kernel enum. The currying happens on Kernel::callback. The
second parameter is a function of h that is constant across data
points. */
type KernelCall = Box<dyn Fn(f64)->f64 + Send + Sync + 'static>;

fn gauss_kernel(diff : f64, scale : f64) -> f64 {
    ((-1.0) * (diff.powf(2.) / scale)).exp()
}

fn tophat_kernel(diff : f64, h : f64) -> f64 {
    if diff.abs() <= h { 1.0 } else { 0.0 }
}

fn epanechnikov_kernel(diff : f64, scale : f64) -> f64 {
    1.0 - (diff.powf(2.) / scale)
}

fn exponential_kernel(diff : f64, scale : f64) -> f64 {
    ((-1.0) * (diff.abs() / scale)).exp()
}

fn linear_kernel(diff : f64, h : f64) -> f64 {
    let diff_abs = diff.abs();
    if diff_abs < h { 1.0 - diff_abs / h } else { 0.0 }
}

fn cosine_kernel(diff : f64, h : f64) -> f64 {
    let diff_abs = diff.abs();
    if diff_abs < h { ((diff_abs * PI) / (2.0*h)).cos() } else { 0.0 }
}

impl Kernel {

    pub fn bandwidth(&self) -> f64 {
        match self {
            Kernel::Gaussian(b) => *b,
            Kernel::TopHat(h) => *h,
            Kernel::Epanechnikov(h) => *h,
            Kernel::Exponential(h) => *h,
            Kernel::Linear(h) => *h,
            Kernel::Cosine(h) => *h
        }
    }

    fn callback(&self) -> KernelCall {
        match *self {
            Kernel::Gaussian(h) => {
                Box::new(move |diff : f64| gauss_kernel(diff, 2.0 * h.powf(2.)) )
            },
            Kernel::TopHat(h) => {
                Box::new(move |diff : f64| tophat_kernel(diff, h) )
            },
            Kernel::Epanechnikov(h) => {
                Box::new(move |diff : f64| epanechnikov_kernel(diff, h.powf(2.)) )
            },
            Kernel::Exponential(h) => {
                Box::new(move |diff : f64| exponential_kernel(diff, h.powf(2.)) )
            },
            Kernel::Linear(h) => {
                Box::new(move |diff : f64| linear_kernel(diff, h) )
            },
            Kernel::Cosine(h) => {
                Box::new(move |diff : f64| cosine_kernel(diff, h) )
            }
        }
    }

}

/// Kernel density estimate of empirical distributions
// https://en.wikipedia.org/wiki/Kernel_density_estimation
pub struct Density {
    pts : Vec<f64>,
    cb : KernelCall,
    bandwidth : f64
}

impl Density {

    pub fn estimate(pts : &[f64], kernel : Kernel) -> Self {
        Self {
            pts : pts.to_vec(),
            bandwidth : kernel.bandwidth(),
            cb : kernel.callback()
        }
    }

    pub fn evaluate(&self, x : f64) -> f64 {
        let mut d = 0.0;
        for pt in &self.pts {
           d += (self.cb)(x - pt);
        }
        d /= (self.pts.len() as f64 /* * self.bandwidth */ );
        d
    }

}


