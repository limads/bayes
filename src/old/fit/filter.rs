use crate::prob::*;
use nalgebra::*;
use crate::fit::linear::OLS;

/// The Weiner filter is basically a least-squares problem, where the k past
/// samples are taken to be predictors for the next sample. The disadvantage
/// is that we must re-calculate the OLS estimate for each new circular
/// change in the input vector.
pub struct Weiner(OLS);

/// Start with the MSE \hat MSE = arg min_\theta E[(y - \tilde y)^2].
/// Suppose E[\theta] = 0 (prior over \theta is zero) and E[x] = 0
/// (expected value of observation is zero in principle, i.e. x is an error
/// between a noisy and "true" signal). If \theta = A^T X where A is an linear
/// estimator that reduces the MSE, then MSE(A) = E[||\theta - A^T x||_2^2]
/// Leads to the estimate \hat A = \Sigma_{xx}^{-1} \Sigma_{x\theta}
/// and \hat \theta_{LMS} = \Sigma_{x\theta} \Sigma_{xx}^{-1} X (Weiner filter).
/// (derived from Wiener-Hopf Equation).
/// The LMS adaptively updates filter weights (b) by following
/// the steepest-descent direction of the mean squared error
/// between a desired and actual measurement, wrt. the parameter
/// vector. In the long-run, the LMS converges to the Weiner filter.
/// While the algorithm supplies the steepest-descent direction for
/// a new sample, the user must tune the learning rate by informing
/// a constant number in [0.0..1.0].
pub struct LMS {

    // Coefficient vector
    b : DVector<f64>,

    // Actual past output value of dimension 1x1
    y_old : f64,

    // Predictor values of current sample of dimension kx1
    scaled_x_new : DVector<f64>,

    // Learning rate
    rate : f64
}

impl LMS {

    pub fn init(k : usize, rate : f64) -> Self {
        let b = DVector::from_element(k, 1e-6);
        Self { b, y_old : 0.0, scaled_x_new : DVector::zeros(k), rate }
    }

    pub fn step(&mut self, x_new : &[f64], y_new : &[f64]) {
        let x_new = DVectorSlice::from(x_new);
        let err = self.y_old - self.b.dot(&x_new);
        self.scaled_x_new.copy_from(&x_new);
        self.scaled_x_new.scale_mut(err * self.rate);

        // b_new = b_old * rate*err*x_new
        self.b += &self.scaled_x_new;

        self.y_old = self.b.dot(&x_new);

    }

}

pub struct RLS {

    // Coefficient vector
    b : DVector<f64>,

    // P matrix
    p_mat : DMatrix<f64>,

    y_old : f64,

    // Learning rate
    rate : f64,

    z : DVector<f64>,

    scaled_err : DVector<f64>,

    kalman : DVector<f64>
}

impl RLS {

    /// lambda should be a value close to 1.0 (e.g. 0.98, 0.99).
    pub fn init(k : usize, lambda : f64, rate : f64) -> Self {
        let mut p_mat = DMatrix::zeros(k, k);
        p_mat.fill_diagonal(1. / lambda);
        let b = DVector::from_element(k, 1e-6);
        let y_old = 0.0;
        let kalman = DVector::zeros(k);
        let scaled_err = DVector::zeros(k);
        let z = DVector::zeros(k);
        Self { b, p_mat, y_old, rate, scaled_err, kalman, z }
    }

    pub fn step(&mut self, x_new : &[f64], y_new : f64) {
        let x_new = DVectorSlice::from(x_new);

        // Calculate Kalman gain from inverse-covariance and new predictors
        self.z = &self.p_mat * &x_new;
        self.kalman = self.z.clone();
        self.kalman.scale_mut(1. / (self.rate + x_new.dot(&self.z)));

        // Update coefficients
        self.scaled_err = self.kalman.clone();
        let err = self.y_old - self.b.dot(&x_new);
        self.scaled_err.scale_mut(err);
        self.b += &self.scaled_err;

        // Update p matrix
        let mut kal_prod_zt = self.kalman.clone() * self.z.transpose();
        kal_prod_zt.scale_mut(1. / self.rate);
        self.p_mat.scale_mut(1. / self.rate);
        self.p_mat -= &kal_prod_zt;

        self.y_old = self.b.dot(&x_new);
    }
}
