use crate::prob::MultiNormal;

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
