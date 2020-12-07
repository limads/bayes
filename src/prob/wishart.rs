use nalgebra::*;
use super::*;
// use std::fmt::{self, Debug};
use serde::{Serialize, Deserialize};
use super::vonmises::*;
use std::fmt::{self, Display};

/// A structural representation of a correlation matrix, with entries over [-1,1], resulting
/// from a function of a scalar correlation rho and the (i,j) offset. Conditional on a realization of this
/// correlation scalar, the diagonal of an empirical covariance matrix is a sufficient statistic
/// that can be used to estimate the realized matrix log-probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Correlation {

    /// Single common-scale parameter, estimated by tr(S) / p.
    Isotropic,

    /// Independent parameters, estimated by S[i,i]. p(s) = prod p(S[i,i]).
    /// Off-diagonal correlation elements are simply ignored when calculating the
    /// log-probability, since they are assumed independent. The vector holds a
    /// [tau0, log(tau0) ... tau_p, log(tau_p) sequence].
    Diagonal,

    /// Covariance matrix for with independent diagonal terms and with
    /// off-diagonal terms represented by (si)(sj)(r), for r = cos(theta), where theta is a draw
    /// from an independent Von-Mises distribution. The sample correlation coefficients
    /// r = sij^2 / (sqrt(si) sqrt(sj)) are used as an estimate for cos(theta).
    /// All diagonal elements are conditionally independent given a draw from r, which
    /// allow their estimation. Diagonal elements follow the same representation as the diagonal
    /// variant. The f64 field represent the realized rho.
    Homogeneous(f64),

    /// Follow a structure similar to homoneneous, but with a penalty term
    /// to the r coefficient determined by how far apart the (i,j) pair is. This penalty
    /// is determined by (i - j) / k and applies up to a constant fixed
    /// interval k; the covariance is zero beyond this limit. All diagonal elements
    /// are mutually independent given a draw from r. Diagonal elements follow the same
    /// representation as the diagonal variant. The f64 field represent the realized rho;
    /// the usize field the autocorrelation order.
    Autoregressive(f64, usize),

    /// The banded structure is a autoregressive structure that repeats
    /// itself over row-blocks of the matrix with fixed number of rows.
    /// This structure is well-suited
    /// to represent spatial correlation when the multivariate mean represent
    /// row-wise or column-wise measurements over a 2D surface. The f64 field represent
    /// the realized rho; the first field the autocorrelation order; the third field
    /// the number of rows of each correlated band.
    Banded(f64, usize, usize)

}

/// An inverse-Wishart distribution, when sampled, generate precision matrices.
/// While Wisharts are formally defined for all positive-definite
/// matrices, this structure represent only a subset of structured covariance matrices,
/// the rational of which is documented in the Correlation enum.
/// The sufficient statistic for a Wishart, conditional on a realization of a single correlation
/// coefficient and a defined covariance structure, is the diagonal of the precision matrix (or its
/// trace in the case of isotropic covariance). For any realized covariance S, S = D R D, where
/// D is a diagonal of variance elements (assumed mutually independent); and R is the correlation structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wishart {

    diag : DVector<f64>,

    /// Prior for the diagonal of the precision matrix. Will hold a single
    /// gamma for a isotropic covariance; or p gammas for a generic
    /// covariance.
    diag_factor : Vec<Gamma>,

    /// Autocorrelation structure. The precision/covariance can
    /// be recovered as a function from this structure and the
    /// rotation factor.
    corr : Correlation,

    corr_mat : DMatrix<f64>,

    /// Set at parameter update;
    gamma_suf : DMatrix<f64>,

    /// Rotation factor rho = cos(theta), argument to the correlation function.
    rot_fact : Option<VonMises>,

    traj : Option<Trajectory>,

    approx : Option<Box<MultiNormal>>

}

/*impl Debug for Wishart {

    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.debug_tuple("").field(&self.diag).finish()
    }

}*/

impl Wishart {

    pub fn rebuild_corr(&mut self, rho : f64) {
        match self.corr {
            Correlation::Homogeneous(ref mut old_rho) => {
                *old_rho = rho;
            },
            Correlation::Autoregressive(ref mut old_rho, _order) => {
                *old_rho = rho;
            },
            Correlation::Banded(ref mut old_rho, _order, _nrow) => {
                *old_rho = rho;
            }
            _ => {
                self.corr_mat.set_diagonal(&DVector::from_element(self.diag.nrows(), 1.));
            }
        }
    }

    /// Extract diagonal of an realized precision matrix, using current correlation
    /// structure.
    pub fn extract_diag(&self, _prec : DMatrixSlice<'_, f64>) -> DVector<f64> {
        unimplemented!()
    }

    /*pub fn write_cov(&self, p : &mut DMatrix<f64>) {
        let s = self.sample();
        p.set_diagonal(&s.column(0));
        for i in 0..self.diag.nrows() {
            for j in 0..self.diag.nrows() {
                if i != j {
                    p[(i, j)] = self.cov_struct.apply(p[(i, j)], i, j);
                }
            }
        }
    }

    pub fn sample_cov(&self) -> DMatrix<f64> {
        let mut cov = DMatrix::from_element(self.diag.nrows(), self.diag.nrows(), 0.0);
        self.write_cov(&mut cov);
        cov
    }

    pub fn sample_prec(&self) -> DMatrix<f64> {
        let mut prec = DMatrix::from_element(self.diag.nrows(), self.diag.nrows(), 0.0);
        self.write_prec(&mut prec);
        prec
    }

    /// Sample from self and use the results to write a covariance matrix.
    pub fn write_prec(&self, p : &mut DMatrix<f64>) {
        // write_cov into p
        // invert p
        // write p
        unimplemented!()
    }

    pub fn new_isotropic(sigma : f64, n : usize) -> Self {
        unimplemented!()
    }

    /// A multivariate random variable that generates
    /// a set of independent variances fixed at the sigma
    /// values.
    pub fn new_diagonal(diag : &[f64]) -> Self {
        unimplemented!()
    }

    /// Generate a new full covariance matrix by an arbitrary function from the
    /// corresponding diagonal element at the row and the distance from this element.

    pub fn new_correlated<F>(diag : &[f64], f : F) -> Self
        where F : Fn(f64, usize, usize)->f64
    {
        unimplemented!()
    }*/

}

impl Distribution for Wishart {

    fn set_parameter(&mut self, _p : DVectorSlice<'_, f64>, _natural : bool) {
        // Use p to calculate a 2-column matrix of sufficient statistics for the gamma
        // priors to each entry of the diagonal of the covariance. Received p in the
        // natural scale is log(precision); or precision in the original scale.
        // If there is a rotation factor; eval log_prob(.) of this factor w.r.t.
        // the last element of p.
        // let n_params = self.mean().nrows() + 1;
        unimplemented!()
    }

    fn view_parameter(&self, _natural : bool) -> &DVector<f64> {
        unimplemented!()
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        // Return diagonal entry current values.
        unimplemented!()
    }

    fn mode(&self) -> DVector<f64> {
        unimplemented!()
    }

    fn var(&self) -> DVector<f64> {
        unimplemented!()
    }

    /// Does not really use y because the value against which
    /// log_prob(.) is evaluated is a constant set at the same
    /// time the multivariate normal is updated (this is gamma_suf).
    fn log_prob(&self, _y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>) -> f64 {
        let rot_lp = if let Some(ref r) = self.rot_fact {
            match self.corr {
                Correlation::Homogeneous(ref rho) | Correlation::Autoregressive(ref rho, _) |
                Correlation::Banded(ref rho, _, _) => {
                    let t = DMatrix::from_element(1,1,*rho);
                    r.suf_log_prob(t.slice((0,0), (1,1)))
                },
                _ => panic!("Invalid rotation factor for given autocorrelation")
            }
        } else {
            0.0
        };
        let mut lp = 0.0;
        for (i, g) in self.diag_factor.iter().enumerate() {
            lp += g.suf_log_prob(self.gamma_suf.slice((i,0), (1, self.gamma_suf.ncols())));
        }
        lp + rot_lp
    }

    fn sample_into(&self, _dst : DMatrixSliceMut<'_,f64>) {
        unimplemented!()
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        None
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        None
    }

}

impl Posterior for Wishart {

    fn dyn_factors_mut(&mut self) -> (Option<&mut dyn Posterior>, Option<&mut dyn Posterior>) {
        match &mut self.rot_fact {
            Some(ref mut v) => {
                (Some(v as &mut dyn Posterior), None)
            },
            _ => (None, None)
        }
    }

    fn approximation_mut(&mut self) -> Option<&mut MultiNormal> {
        self.approx.as_mut().map(|apprx| apprx.as_mut())
    }

    fn approximation(&self) -> Option<&MultiNormal> {
        self.approx.as_ref().map(|apprx| apprx.as_ref())
    }

    fn trajectory(&self) -> Option<&Trajectory> {
        self.traj.as_ref()
    }

    fn trajectory_mut(&mut self) -> Option<&mut Trajectory> {
        self.traj.as_mut()
    }
    
    fn start_trajectory(&mut self, size : usize) {
        self.traj = Some(Trajectory::new(size, self.view_parameter(true).nrows()));
    }
    
    /// Finish the trajectory before its predicted end.
    fn finish_trajectory(&mut self) {
        self.traj.as_mut().unwrap().closed = true;
    }

}

impl ExponentialFamily<U1> for Wishart
    where
        Self : Distribution
{

    fn base_measure(_y : DMatrixSlice<'_, f64>) -> DVector<f64>
        //where S : Storage<f64, Dynamic, U1>
    {
        unimplemented!()
    }

    fn sufficient_stat(_y : DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        unimplemented!()
    }

    fn suf_log_prob(&self, _t : DMatrixSlice<'_, f64>) -> f64 {
        unimplemented!()
    }

    fn update_log_partition<'a>(&'a mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn log_partition<'a>(&'a self) -> &'a DVector<f64> {
        unimplemented!()
    }

    /*fn update_grad(&mut self, _eta : DVectorSlice<'_, f64>) {
        unimplemented!()
    }

    fn grad(&self) -> &DVector<f64> {
        unimplemented!()
    }*/

    fn link_inverse<S>(_eta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        unimplemented!()
    }

    fn link<S>(_theta : &Matrix<f64, Dynamic, U1, S>) -> DVector<f64>
        where S : Storage<f64, Dynamic, U1>
    {
        unimplemented!()
    }

}

impl Conditional<VonMises> for Wishart {

    fn condition(self, _vm : VonMises) -> Self {
        unimplemented!()
    }

    fn view_factor(&self) -> Option<&VonMises> {
        unimplemented!()
    }

    fn take_factor(self) -> Option<VonMises> {
        unimplemented!()
    }

    fn factor_mut(&mut self) -> Option<&mut VonMises> {
        unimplemented!()
    }

}

impl Display for Wishart {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wish({})", self.diag.nrows())
    }

}

