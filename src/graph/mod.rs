use petgraph::graph::DiGraph;
use petgraph::graph::NodeIndex;
use nalgebra::*;
use std::fmt::{self, Display};
use std::error::Error;
use std::collections::HashMap;
use super::distribution::*;
use std::convert::From;
mod transformation;
use transformation::*;

/// A probabilistic model, expressed as a directed graph. A node in the graph
/// holds either a distribution or a constant; An edge from a distribution
/// or a constant node to a target distribution means that the target is
/// conditionally independent from the rest of the model.
/// Graphs are useful for building general-purpose sampling
/// algorithms for estimation of marginals (situations when the mode is known
/// but the distribution is not) or full posteriors (for complex models for which
/// no global mode can be assumed). A graph dissociates the
/// declaration of the model from the estimating algorithm, which brings two
/// advantages: (1) It allows the comparison of the output from different estimators
/// for the same problem (extremely useful to determine the adequacy of
/// analytical approximations and/or the convergence of samplers); (2) It allows
/// model definitions in a declarative fashion (e.g. JSON), so models
/// can be saved, have their structure tweaked and then re-loaded,
/// without touching source code. The graph can either be generated automatically
/// from a model with structure known at compile time (such as a conjugate linear
/// or a generalized linear model); or it can be created by the user and fed into
/// a general-purpose estimator (such as the GibbsSampler or MetropolisHastings).
/// In contrast to auto-generated graphs, for which a posterior solution is fairly guaranteed
/// (except for data collinearily and numerical issues), several algorithm-specific
/// conditions need to hold so that the parameters of an user-defined model
/// are successfully estimated. A model is only valid if for all likelihood nodes
/// there is a complete prior specification (single node for location or two
/// nodes for location-scale priors). In such representation, a conditional independence
/// relationship is maintained by pairs of nodes with a node between them: if there is
/// a node behind the first node in the chain, we have two possible conditional independence
/// relationships: A-C|B and B-D|C. Such "even" joint probabilities (at 2,4..6 chain elements)
/// can have their log-probabiblities factored as sums.
/// A graph is a factorization of a joint probability distribution, and as such it implements
/// the Distribution trait, parametrized by an n-dimensional parameter vector. The meaning
/// of this vector is model-specific; and is determined by the order with which the functions
/// add_prior(.) and add_likelihood(.) are called; and what is the dimensionality of the arguments
/// to those functions. A complex joint distribution can be decomposed into graphs in different
/// ways; This structure represents just one of those ways, which is aimed to be the most
/// convenient for some approximating or sampling algorithm. Classical statistical models
/// (generalized linear models; expectation maximization; hierarchical mixed models) will naturally
/// follow a nice representation with single-parent elements (single prior). While a generic conditional
/// probability factorization of k variables require k(k-1)(k-2)...(1) terms, the "loc" relationship
/// between two composed distributions (where the output of one distribution is the direct input
/// to another) establishes a conditional independence between them, requiring only 2 terms: k(k-1),
/// where k-1 is calculated recursively back over the model. The model graph is passed in the
/// forward direction for sampling; and in the backward direction for calculation of log-probabilities.
/// Links essentially mean a conditional expectation. Sampling at intermediate and likelihood nodes
/// are always conditional on parameters from previous nodes; and so are the log-likelihood calculations.
/// The expected value of the previous node is the sampling argument for the next mode.
pub struct Model {

    graph : DiGraph<ModelNode, Transf>,

    /// Constant (target) nodes.
    data_ix : Vec<NodeIndex>,

    /// Likelihood nodes (have constant data target nodes)
    lik_ix : Vec<NodeIndex>,

    /// Parameter prior/posterior nodes (have distribution target nodes).
    /// The order here defines a natural order for the parameter vector
    /// at the distribution implementation.
    prior_ix : Vec<NodeIndex>,

    /// Constant nodes, both external (predictors) and internal
    /// (transformations, intermediate computations).
    const_ix : Vec<NodeIndex>,

    /// Hyperpriors (functions with fixed priors, without incoming nodes).
    /// During sampling, those are the nodes that are sampled first,
    /// and have their samples propagated to the next level. Those elements
    /// are special in the sense that they are non-normalizable and they
    /// cannot be evaluated conditional on any other element (but can serve
    /// as conditioning elements). This is the same as saying that their
    /// marginal distribution equals to any conditional distribution we might
    /// want to build.
    hyper_ix : <Vec<NodeIndex>>

}

#[derive(Debug)]
pub struct ModelError {
    msg : String
}

impl ModelError{

    pub fn new(msg : &'static str) ->Self {
        Self { msg : String::from(msg) }
    }

}

impl Display for ModelError {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl Error for ModelError{ }

pub enum CovModality {

    // MultiNormal is represented by p + 1 parameters, where the last
    // element is sigma such that sigma*I is the covariance.
    Isotropic,

    // Multinormal is represented by 2p parameters, where
    // parameters p+1..2p are the diagonal elements of the covariance
    // matrix.
    Uncorrelated,

    // Follows same size as uncorrelated; but covariance row is
    // determined by an arbitrary function of the distance its distance from the
    // variance diagonal.
    Mapped(Box<dyn Fn(f64, usize)->f64),

    // Parameter vectors has p + p^2 parameters, where parameters p+1..p^2
    // holds the covariance entries (column-wise).
    Heterogenous

}

/// Likelihood model nodes do not have an index in the prior/posterior
/// parameter vector; prior/posterior nodes have Some(ix), which is used
/// to build the parameter vector. Such indices depend on the prior
/// insertion order: the current index is the previous prior index +
/// previous prior dimensionality. A model node represents not only the
/// alternating logic between different probability distributions;
/// but completely encodes how the internal distribution parameters
/// are accomodated in the final graph parameter vector. The vector
/// representation is essential for approximation and sampling,
/// since those algorithms ammount to the comparison between the graph
/// distribution and some proposal or approximating distribution.
/// For sampling, bounded proportion parameters and non-negative scale
/// parameters should be transformed to their linear counterparts
/// when getting or setting them.
pub enum ModelNode {
    Constant{ name : String, value : DMatrix<f64> },

    /// Whether informed parameter values are on theta space (0-1) or eta space (-inf to inf).
    /// The latter representation is useful for sampling algorihtms. The internal Bernuolli
    /// always hold only the constrained version; the indicator only encodes how the information
    /// lives in the output parameter vector (and thus can be used to decide if some transformation
    /// is required before any internal updates).
    Bernoulli{ ix : Option<usize>, name : String, distr : Bernoulli, constrained : bool },
    Poisson{ ix : Option<usize>, name : String, distr : Poisson },
    Exponential{ ix : Option<usize>, name : String, distr : Exponential },
    Normal{ ix : Option<usize>, name : String, distr : Normal },

    /// The constrained option follows hte same logic as described by the documentation of Bernoulli.
    Categorical{ ix : Option<usize>, name : String , distr: Categorical, constrained : bool },

    MultiNormal{ ix : Option<usize>, name : String, distr : MultiNormal, cov : CovModality }
}

impl From<(usize, String, DMatrix<f64>)> for ModelNode {

    fn from(t : (usize, String, DMatrix<f64>)) -> Self {
        ModelNode::Constant{ index : t.0, name : t.1, value : t.2 }
    }

}

impl From<(usize, String, Bernoulli)> for ModelNode {

    fn from(t : (usize, String, Bernoulli)) -> Self {
        ModelNode::Bernoulli{name : t.1, distr : t.2}
    }

}

impl From<(usize, String, Poisson)> for ModelNode {

    fn from(t : (usize, String, Poisson)) -> Self {
        ModelNode::Poisson{name : t.1, distr : t.2}
    }

}

impl From<(usize, String, Exponential)> for ModelNode {

    fn from(t : (usize, String, Exponential)) -> Self {
        ModelNode::Exponential{name : t.1, distr : t.2}
    }

}

impl From<(usize, String, Normal)> for ModelNode {

    fn from(t : (usize, String, Normal)) -> Self {
        ModelNode::Normal{name : t.1, distr : t.2}
    }

}

impl From<(usize, String, Categorical)> for ModelNode {

    fn from(t : (usize, String, Categorical)) -> Self {
        ModelNode::Categorical{name : t.1, distr : t.2}
    }

}

impl From<(usize, String, MultiNormal)> for ModelNode {

    fn from(t : (usize, String, MultiNormal)) -> Self {
        ModelNode::MultiNormal{name : t.1, distr : t.2}
    }

}

impl ModelNode {

    pub fn name<'a>(&'a self) -> &'a str {
        match self {
            ModelNode::Constant{name, ..} => &name[..],
            ModelNode::Bernoulli{name, ..} => &name[..],
            ModelNode::Poisson{name, ..} => &name[..],
            ModelNode::Exponential{name, ..} => &name[..],
            ModelNode::Normal{name, ..} => &name[..],
            ModelNode::Categorical{name, ..} => &name[..],
            ModelNode::MultiNormal{name, ..} => &name[..],
        }
    }

    pub fn retrieve_data(&self) -> DMatrix<f64> {
        match self {
            ModelNode::Constant{ value, ..} => value.clone(),
            ModelNode::Bernoulli{ distr, ..} => DMatrix::<f64>::from_element(1,1,distr.parameter() ),
            ModelNode::Poisson{ distr, ..} => DMatrix::<f64>::from_element(1,1,distr.parameter() ),
            ModelNode::Exponential{ distr, ..} => DMatrix::<f64>::from_element(1,1,distr.parameter() ),
            ModelNode::Normal{ distr, ..} => DMatrix::<f64>::from_element(1,1,distr.parameter().0 ),
            ModelNode::Categorical{ distr, ..} => {
                let theta = distr.parameter();
                DMatrix::<f64>::from_iterator(theta.nrows(), 1, theta.iter().map(|t| *t))
            },
            ModelNode::MultiNormal{ distr, .. } => {
                let (mu,_) = distr.parameter();
                DMatrix::<f64>::from_iterator(mu.nrows(), 1, mu.iter().map(|t| *t))
            }
        }
    }

    pub fn update_data(&mut self, data : DMatrix<f64>, extra : Option<DMatrix<f64>>) {
        match self {
            ModelNode::Constant{ ref mut value, ..} => {
                value.copy_from(&data);
            },
            ModelNode::Bernoulli{ ref mut distr, ..} => {
                distr.update(data[(0,0)]);
            },
            ModelNode::Poisson{ ref mut distr, ..} => {
                distr.update(data[(0,0)]);
            },
            ModelNode::Exponential{ ref mut distr, ..} => {
                distr.update(data[(0,0)]);
            },
            ModelNode::Normal{ ref mut distr, ..} => {
                let sigma2 = extra.map(|e| e[(0,0)]).unwrap_or(distr.parameter().1);
                distr.update((data[(0,0)], sigma2));
            },
            ModelNode::Categorical{ ref mut distr, ..} => {
                let theta : DVector<f64> = data.column(0).into();
                distr.update(theta);
            },
            ModelNode::MultiNormal{ ref mut distr, .. } => {
                let sigma = extra.unwrap_or(distr.parameter().1);
                let mu : DVector<f64> = data.column(0).into();
                distr.update((mu, sigma));
            }
        }
    }

    pub fn is_distribution(&self) -> bool {
        match self {
            ModelNode::Constant{ .. } => false,
            _ => true
        }
    }

    /*pub fn is_likelihood(&self) -> bool {
        if self.is_distribution() {
            match self {

            }
        } else {
            false
        }
    }*/

}

struct SubModel<'a> {
    full : &'a Model,
    top_ix : usize
}

/// A graph of compositions of analytical probability distributions has a log-probability that is
/// a simple function of the distributions' log-probability.
/// Its parameter vector is defined by a natural ordering dependent on the order
/// of prior insertion (with multivariate distribution locations defining
/// fixed vector segments; and covariances flattened over columns). Such formalism
/// is relevant for building conditional distributions during sampling, which often
/// requires the comparison between a complex user-defined distribution and a simpler
/// proposal distribution. Such conditional distributions are slices over the full graph.
/// The density of this implementor is given by the density of the last element in the graph,
/// conditionally generated by walking from all its root hyperparameters. This suggests a simple recursive
/// rule for generating the distribution's log-probability: If x is a final distribution, return its log-probability.
/// If not, x will be conditionally independent from its roots, so its log-probability factors as the
/// sum of its current sample and the sample of the previous distributions.
/*impl<'a> SubModel<'a> {

    /// If there is a data vector/matrix, evaluate the cond_log_prob of the
    /// current distribution. If not, use the current parameter and evaluate log_prob
    /// of the scalar parameter alone.


}*/

/// Models are built top-to-bottom: First add the likelihoods (distributions
/// that are conditioned on data and for which markovianity is sustained relative
/// to the rest of the graph); then add the priors:
/// First call model.add_likelihood("y", Normal::new());
/// Then call model.add_prior("mu", Normal::new(), Some("y", Transf::Location));
/// Then call model.add_prior("sigma", Gamma::new(), Some("y", Transf::Scale));
impl Model {

    fn insert_node_if_unique<'a>(
        &'a mut self,
        node : ModelNode,
        target : Option<(Transf, NodeIndex)>
    ) -> Result<NodeIndex, ModelError> {
        for node_ix in self.graph.node_indices() {
            if self.graph[node_ix].name() == node.name() {
                return Err(ModelError::new("Name already used"));
            }
        }
        let ix = self.graph.add_node(node);
        if let Some(targ) = target {
            self.graph.add_edge(ix, targ.1, targ.0);
        }
        Ok(ix)
    }

    /*// The returned distributions are restricted to accept the same
    // dimensionality for the data.
    fn likelihood_iter<'a>(&'a mut self) -> impl Iterator<Item=&'a mut dyn Distribution<N,D>> {
        self.lik_ix.iter_mut()
    }

    fn parameter_iter<'a>(&'a mut self) -> impl Iterator<Item=&'a mut dyn Distribution<N,D>> {

    }*/

    fn validate_distr_transf(t : &Transf) -> Result<(), ModelError> {
        match t {
            Transf::Loc | Transf::Scale => Ok(()),
            _ => Err(ModelError::new(
                "Admissable transformations to other distributions are Loc|Scale"
            ))
        }
    }

    /// Add a prior distribution, that can be connected to a target prior or likelihood.
    /// If no target is informed, this distribution is an hyperprior, and its underlying parameters
    /// will not change as the algorithm runs.
    pub fn add_prior<D, P, E>(
        &mut self,
        name : &str,
        d : D,
        target : &str,
        t : Transf
    ) -> Result<(), impl Error>
        where
            D : Distribution<P, E>,
            ModelNode : From<(String, D)>
    {
        Self::validate_distr_transf(&t)?;
        let prior_ix = self.prior_ix.clone();
        let lik_ix = self.lik_ix.clone();
        for targ_ix in prior_ix.iter().chain(lik_ix.iter()) {
            if self.graph[*targ_ix].name() == target {
                let distr_tuple = (name.to_string(), d);
                let prior_node : ModelNode = distr_tuple.into();
                match self.insert_node_if_unique(prior_node, Some((t, *targ_ix))) {
                    Ok(ix) => {
                        self.prior_ix.push(ix);
                        if let Some(i) = self.hyper_ix.iter().position(|n| n == targ_ix ) {
                            self.hyper_ix.remove(i);
                            self.hyper_ix.insert(ix, i);
                        }
                        return Ok(());
                    },
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }
        Err(ModelError::new("Prior target not found in prior or likelihood nodes"))
    }

    /// Add a leaf distribution, which will be conditioned on data. Unlike priors,
    /// this distribution has no target. Its name should match the HashMap that is passed
    /// as data to the InferenceAlgorithm implementor.
    pub fn add_likelihood<D, P, E>(
        &mut self,
        name : &str,
        d : D,
        target : &str
    ) -> Result<(), impl Error>
        where
            D : Distribution<P, E>,
            ModelNode : From<(String, D)>
    {
        let data_ix = self.data_ix.clone();
        for data_ix in data_ix.iter() {
            if self.graph[*data_ix].name() == target {
                let distr_tuple = (name.to_string(), d);
                let lik_node : ModelNode = distr_tuple.into();
                match self.insert_node_if_unique(lik_node, Some((Transf::Gen, *data_ix))) {
                    Ok(ix) => {
                        self.lik_ix.push(ix);
                        return Ok(())
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }
        Err(ModelError::new("Likelihood target not found in set of data nodes"))
    }

    /// A predictor is a constant node that have
    /// another constant node or a distribution as its
    /// target. Its values are assumed known and are not
    /// modelled as a generative process. A predictor
    /// that maps to a single likelihood is avalid model: you
    /// can observe its mode for each predictor sample, and
    /// you can sample from its resulting distribution,
    /// although there is no inference problem for it to solve.
    /// One can also add constants to express intermediate non-probabilistic
    /// computations.
    pub fn add_constant(
        &mut self,
        name : &str,
        data : DMatrix<f64>,
        target : &str,
        t : Transf
    ) -> Result<(), impl Error> {
        let prior_ix = self.prior_ix.clone();
        let lik_ix = self.lik_ix.clone();
        let const_ix = self.const_ix.clone();
        for targ_ix in prior_ix.iter().chain(lik_ix.iter()).chain(const_ix.iter()) {
            if self.graph[*targ_ix].name() == target {
                if self.graph[*targ_ix].is_distribution() {
                    Self::validate_distr_transf(&t)?;
                }
                let node = ModelNode::from((name.to_string(), data));
                match self.insert_node_if_unique(node, Some((t, *targ_ix))) {
                    Ok(ix) => {
                        self.const_ix.push(ix);
                        return Ok(());
                    },
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }
        Err(ModelError::new("Constant target not found in set of data/likelihood/prior nodes"))
    }

    /// Adds an external data node. Such nodes are constant nodes
    /// living at the leafs of the graph, and will inform
    /// the likelihood parameters via conditioning, which will
    /// in turn inform their priors. Those nodes, by definition,
    /// are terminal. Start building the graph from there;
    /// likelihood functions are then set relative to them.
    pub fn add_observation(
        &mut self,
        name : &str,
        data : DMatrix<f64>
    ) -> Result<(), impl Error> {
        let node = ModelNode::from((name.to_string(), data));
        match self.insert_node_if_unique(node, None) {
            Ok(ix) => {
                self.data_ix.push(ix);
                Ok(())
            },
            Err(e) => Err(e)
        }
    }

    fn joint_log_prob(&self) -> f64 {
        let mut log_prob = 0.0;
        for lik in self.lik_ix.iter() {
            log_prob += self.partial_log_prob(lik);
        }
        log_prob
    }

    /// Calculates log-probability from the informed node index downstream
    /// in the graph.
    fn partial_log_prob(&self, ix : NodeIndex) -> f64 {
        let mut log_prob = 0.0;
        for e in self.graph.edges_directed(ix, Direction::Incoming) {
            log_prob += self.partial_log_prob(e.target());
        }
        log_prob += self.graph[ix].log_prob();
        log_prob
    }

    pub fn partial_update(&mut self, ix : usize, param : &[f64]) -> Result<(), &'static str> {

    }

    pub fn update(&mut self, params : DVector<f64>) -> Result<(), &'static str> {
        let mut vec_ix = 0;
        for ix in self.data.prior_ix.iter() {
            match self.graph[ix] {
                ModelNode::Bernoulli{.. distr} => {
                    distr.update(params[vec_ix]);
                    vec_ix += 1;
                },
                ModelNode::Poisson{ .., distr } => {
                    distr.update(params[vec_ix]);
                    vec_ix += 1;
                },
                ModelNode::Exponential{ .., distr } => {
                    distr.update(params[vec_ix]);
                    vec_ix += 1;
                },
                ModelNode::Normal{ .., distr } => {
                    distr.update(params[vec_ix], params[vec_ix] + 1);
                    vec_ix += 2;
                },
                ModelNode::Categorical{ .., distr } => {
                    let n_params = distr.n_params();
                    let theta = DVector::from_iterator(params.iter().skip(vec_ix).take(n_params));
                    distr.update(theta);
                    vec_ix += n_params;
                },
                ModelNode::MultiNormal{ .., distr } => {
                    let dim = distr.dim();
                    let mu = DVector::from_iterator(params.iter().skip(vec_ix).take(dim));
                    vec_ix += dim;
                    let sigma_sz = dim.pow(2);
                    let sigma = DMatrix::from_iterator(vec_ix, sigma_sz, params.iter().skip(vec_ix).take(sigma_sz));
                    vec_ix += sigma_sz;
                }
            }
        }
        if vec_ix != params.len() {
            Err("Wrong number of parameters");
        } else {
            Ok(())
        }
    }

    /// Generate a new sample from the statistical model,
    /// starting at hyperparameters and sampling up to
    /// the last data blocks.
    pub fn sample(&self) -> HashMap<String, DMatrix<f64>> {
        unimplemented!()
    }

    /*pub fn data(&self) -> HashMap<String, DMatrix<f64>> {
    }

    pub fn likelihood(&self) -> HashMap<String, Distribution> {

    }

    pub fn priors(&self) -> HashMap<String, Distribution> {
        for prior_ix in self.prior_ix {

        }
    }*/

    /// Given informed predictor variables (which are, by definition,
    /// constant nodes which have distributions or constant targets),
    /// return the collection of all implied observed data (constant
    /// nodes without any targets).
    pub fn predict(
        &self,
        pred : HashMap<String, DMatrix<f64>>
    ) -> HashMap<String, DMatrix<f64>> {
        unimplemented!()
    }

    /// Fix the probability of previous nodes; and return
    /// a simplified graph with the ix element as the final
    /// outcome.
    pub fn condition(&self, ix : usize) -> SubModel {

    }
}



