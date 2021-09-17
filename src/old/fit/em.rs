use nalgebra::*;
use crate::prob::*;
use crate::fit::Estimator;
use crate::sample::Sample;

// use super::*;
// use crate::optim::*;
// use std::ops::AddAssign;

/// (Work in progress) The expectation maximization algorithm is a general-purpose inference
/// algorithm that generates a posterior gaussian approximation for each
/// distribution composing a model. At each step of the algorihtm, the full
/// model log-probability is evaluated by changing the parameter vector of a single node
/// in a probabilistic graph, while keeping all the others at their last estimated local
/// maxima.
///
/// The algorithm is guaranteed to converge to a global maxima when the log-posterior
/// is a quadratic function. The gaussian approximation can be used directly if only the
/// posterior mode is required (i.e. prediction; decision under a quadratic loss) or can
/// serve as a basis to build an efficient proposal distribution for the Metropolis-Hastings posterior
/// sampler, which can be used to build a full posterior. This algorithm is useful any time you need
/// to estimate latent variables via maximum likelihood: The mean of the gaussian approximation to
/// the latent variables can be taken as their maximum likelihood.
///
/// # References
/// Dempster, A. P., Laird, N. M., & Rubin, D. B.
/// ([1977](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1977.tb01600.x)).
/// Maximum Likelihood from Incomplete Data
/// Via the EM Algorithm. Journal of the Royal Statistical Society: Series B (Methodological),
/// 39(1), 1–22. doi: 10.1111/j.2517-6161.1977.tb01600.x
pub struct ExpectMax {

}

/*impl<D> Estimator<D> for ExpectMax
    where D : Distribution
{

    fn predict<'a>(&'a self, cond : Option<&'a Sample<'a>>) -> Box<dyn Sample<'a>> {
        unimplemented!()
    }

    fn fit<'a>(&'a mut self) -> Result<&'a D, &'static str> {
        /*self.visit_factors(|f : &dyn mut Posterior| {
            *(f.approximation_mut()) = Some(MultiNormal::from_approximation(eta_traj, convx_traj));
        });
        for i in 0..max_iter {
            let ans = DMatrix::from_
        }*/
        unimplemented!()
    }

}*/

/// Update responsibility matrix from current state of distributions.
/// - cat : Probabilities
/// - mn : category-conditional likelihoods
/// - resp : resposibilities (category-conditional log-likelihoods for each category realization at each column).
fn expectation_step(resp : &mut DMatrix<f64>, cat : &Categorical, liks : &[MultiNormal]) {

    let k = resp.ncols();

    // Arrange categorical theta vector into a row
    let cat_prob_row = cat.view_parameter(false).clone_owned().transpose();

    // Arrange final normal probabilities into a n-sized column
    for i in 0..k {
        let mut norm_prob_col = liks[k].cond_log_prob().unwrap();
        norm_prob_col.scale_mut(cat_prob_row[k]);
        resp.column_mut(k).copy_from(&norm_prob_col);
    }

    // Calculate a column vector of marginal responsibilites over classes
    // with the same dimensionality of the data points (n x 1)
    let marg_resp_classes = resp.row_sum();

    // Normalize responsibility matrix by the marginal responsibilities over classes
    for i in 0..k {
        resp.column_mut(i).component_div_mut(&marg_resp_classes);
    }

}

/// - y : Data matrix
/// - cat : Categorical factors
/// - mn : Multinormals
/// - resp : Responsibility matrix
fn maximization_step(cat : &mut Categorical, liks : &mut [MultiNormal], y : &DMatrix<f64>, resp : &DMatrix<f64>) {

    let k = resp.ncols();
    let n = y.nrows();

    // Row of marginal responsibilities (over observations), one for each class
    let marg_resp_obs = resp.column_sum();

    let mut resp_weighted_y = y.clone();
    for i in 0..k {
        resp_weighted_y.copy_from(&y);
        resp_weighted_y.row_iter_mut()
            .enumerate()
            .for_each(|(ix, mut row)| row.scale_mut(resp[(ix, k)]) );
        let next_mu_row = resp_weighted_y.row_sum().component_div(&marg_resp_obs);

        let mut next_sigma = DMatrix::zeros(k, k);
        for (ix, row) in y.row_iter().enumerate() {
            let err = (row.clone_owned() - &next_mu_row);
            let mut err_prod = err.clone() * err.transpose();
            err_prod.scale_mut(resp[(ix, k)]);
            next_sigma += err_prod;
        }
        next_sigma.unscale_mut(marg_resp_obs[k]);
        liks[k].set_parameter((&next_mu_row.transpose()).into(), true);
        liks[k].set_cov(next_sigma);
    }

    let mut probs = marg_resp_obs.transpose();
    probs.unscale_mut(n as f64);
    cat.set_parameter((&probs).into(), false);

}

fn expectation_maximization(cat : &mut Categorical, liks : &mut [MultiNormal], y : &DMatrix<f64>) {

    let k = cat.view_parameter(true).nrows();
    let n = y.ncols();

    // Matrix of the log-likelihood where each row represent a multinormal allocation;
    // and each column represent a data point. The reponsibility is the log-likelihood
    // of the data point when the selected category is the kth one.
    let mut resp = DMatrix::zeros(n, k);

    // mu_diff holds the average absolute error of previous iteration against current iteration, over
    // all multinormals; prob_diff the absolute error of the categorical probabilities.
    let mut mu_diff = DVector::zeros(k);
    let mut prob_diff = DVector::zeros(k);

    let prob_tol = 1e-5;
    let mu_tol = 1e-5;
    let max_iter = 1000;
    let mut n_iter = 0;

    while mu_diff.norm() > mu_tol && prob_diff.norm() > prob_tol && n_iter <= max_iter {

        let mut mu_prev : Vec<_> = liks.iter()
            .map(|mn| mn.view_parameter(true)
            .clone_owned() ).collect();

        let mut prob_prev = cat.view_parameter(false).clone_owned();

        expectation_step(&mut resp, &cat, &liks);
        maximization_step(cat, liks, &y, &resp);

        prob_diff = prob_prev - cat.view_parameter(false);
        mu_diff = DVector::zeros(k);
        for (old_mu, new_mu) in mu_prev.drain(0..).zip(liks.iter().map(|mn| mn.view_parameter(true) )) {
            mu_diff += (1. / k as f64) * (old_mu - new_mu);
        }

        n_iter += 1;
    }

}

// Univariate implementation
/*let marg_responsibilities = norms.iter()
    .zip(probs.iter())
    .map(|(norm, prob)| norm[i].log_prob() * prob )
    .sum();

// Update responsibility matrix.
for (i, r) in responsibilities.iter_mut().enumerate() {
    *r = (norm.log_prob() *) probs[i] / marg_responsibilities;
}

// Maximization step
for i in 1..k {
    // Calculate average of kth normal by weighting data points by their responsibilities
    let summed_res = responsibilities.column(i).sum();
    let mu_c = data.iter().zip(responsibilities.column(i))
        .map(|d, r| d*r )
        .sum() / summed_res;
    let sigma_c = data.iter().zip(responsibilities.column(i))
        .map(|d, r| (d - mu_c).powf(2.) )
        .sum() / summed_res;
    norm[i].set_parameter(mu_c);
    norm[i].set_var(sigma_c);
    probs[i] = responsibilities.map(|resp| resp / n );
}*/
