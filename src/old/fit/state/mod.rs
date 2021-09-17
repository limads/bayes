use crate::prob::MultiNormal;
use nalgebra::*;

/// The Kalman filter is applicable whenever we have a dynamical parameter estimation
/// with state update that can be seted as a linear operator (a matrix multiplication).
pub struct Kalman {

}

impl Kalman {

    pub fn with_transition(&mut self, op : DMatrix<f64>) -> &mut Self {
        self
    }
}

/// The particle filter is applicable to dynamical parameter estimation with
/// a state update that is an arbitrary (possibly non-linear) function.
pub struct Particle {

}

impl Particle {

    pub fn with_transition(&mut self, op : impl Fn(&mut [f64])) -> &mut Self {
        self
    }
}

/*pub enum Covariance {
    Constant,
    Linear,
    Exponential
}

pub struct GaussProcess {
    mn : MultiNormal,
    cov_fn : Covariance
}

impl GaussProcess {

    pub fn new(cov : Covariance) -> Self {
        unimplemented!()
    }
}*/

/*
/// A Chain is a directed cyclic graph of categorical distributions.
/// It is the discrete analog of the RandomWalk structure.
/// Categoricals encode state transition probabilities (which inherit all the
/// properties of categoricals, such as conditioning on MultiNormal factors,
/// useful to model the influence of external factors at the transition probabilities).
///
/// Transition probabilities are required only to be conditionally independent,
/// but they might be affected by factor-specific external variables.
struct Chain {

    /// A state is simply a categorical holding transition probabilities.
    /// Since categoricals can be made a function of a multinormal (multinomial regression),
    /// the transition probabilities can be modelled as functions of external features.
    states : Vec<Categorical>,

    /// The target state to which each transition refers to is the entry at the dst
    /// vector. Each entry at the inner vector is an index of the states vector
    /// (including the current one). Transition targets are not required to be of
    /// the same size, and empty inner vectors mean final states. Transitions might
    /// refer to any categorical in the states vector, including the current state.
    dst : Vec<Vec<usize>>,

    /// The limit field determines the maximum transition size. Without this field,
    /// recursive chains would repeat forever.
    limit : usize,

    curr_state : usize
}

pub enum Transition {

    /// Explore all transition possibilities
    Any,

    /// Only transition to the highest probability
    Highest,

    /// Accept only probabilities that have minimum value
    Minimum(f64),

    /// Accept only the n-best probabilities for any given transition
    Best(usize)

}

impl Chain {

    /// Return an exhaustive list of all possible trajectories and
    /// their respective joint probabilities, ordered from the most likely trajectory to
    /// the least likely. Using trajectories.first() yields the MAP estimate for the markov process.
    /// Trajectores start at the informed state and end until either a final node is found
    /// or the state transition limit is reached.
    fn trajectories(&self, from : usize, rule : Transition) -> Vec<(Vec<usize>, f64)>;

    /// Generate a random trajectory
    fn sample(&self, n : usize, rule : Transition) -> Vec<<Vec<usize>>;

}

/// Use the curr_state method to walk into some state. Might yield mutable references so
/// the categoricals may be updated with external data.
impl Iterator for Chain {

}

impl Extend for Chain {

    /// Receives an iterator over the tuple (Categorical, Vec<usize>)
    fn extend<T>(&mut self, iter: T) {

    }
}

/// HiddenMarkov wraps a Markov chain for which only the realizations
/// of corresponding continuous distributions are seen (the observed variables
/// are mixture distributions). A index realization
/// i means that the observation is a draw by indexing the obs vector at
/// index i. Using observed continuous states conditional on discrete states
/// naturally accomodate translation/scale variants expected in sound/image
/// recognition problems.
struct HiddenMarkov {
    chain : MarkovChain
    obs : Vec<MultiNormal>
}

*/

/*

/// Undirected probabilistic graph. A markov field is defined by the property:
/// An element in the graph is independent of the rest of the graph conditional
/// on its immediate neighbors. For computational tractability, the neighborhood
/// is usually kept at a very small dimensionality (2 or 4). 
pub struct Field {

}

/// Continuous stochastic process. A stochastic process is defined by the markov property:
/// The future is independent of the past conditional on the present. While image::Image and signal::Signal
/// are the actual realizations, think of a process as a concise mathematical description of the set of all possible 
/// realizations. A Process is to a Signal or an Image what a Distribution is to a structured sample.
/// Processes usually imply
/// a dynamic transition model which allow for inference over parameters changing over time. 
/// A random walk is a process with the identity function as the transition. A Gaussian process
/// is a process for which the "transitions" are specified by a function of pair-wise distances.
pub struct Process<T> {
    distr : D,
    transition : DMatrix<f64>
}

pub type GaussProcess = Process<MultiNormal>;

pub type Kalman = Process<MultiNormal>;

*/

/*struct RLSFilter {

    // Attenuation factor λ^{n - i} (i = 0..p)
    lambda : f64,

    n : usize,

    // Model signal
    model_sig : DVector<f64>,

    state : RLSState
}

pub struct RLSParams {
    model_sig : DVector<f64>,
    attenuation_factor : f64
}

impl<'a> AdaptiveFilter<'a, RLSState, RLSParams> for RLSFilter {

    /// Starts a new RLS filter implementation, with an optional
    /// attenuation factor (0,1]
    fn init(
        pt : DVector<f64>,
        state : Option<S>,
        params : Option<I>
    ) -> Self {
        match params {
            Some(params) => {
                if lambda > 0 && lambda <= 1.0 {
                    self.lambda = lambda;
                    Ok(())
                } else {
                    return Err("Invalid attenuation factor. Should be within (0, 1]")
                }
            },
            None => { Err("No model signal informed.") }
        }
    }

    fn update(
        &'a mut self,
        pt : DVector<f64>
    ) -> &'a mut DVector<f64> {
        // Update gain
        let gain_scale = 1.0 / (self.lambda + pt.transpose() * self.state.precision * pt);
        self.state.gain = self.state.precision * pt * gain_scale;
        let err = self.model_sig - pt * self.weights;

        // Update weights
        self.weights += self.state.gain.scale(err);

        // Update precision for next iteration
        self.state.precision = 1.0 / self.lambda * self.state.precision -
            self.state.gain * pt.transpose() * 1.0 / self.lambda * self.state.precision;
    }

    fn state(&'a self) -> RLSState {

    }
}*/
