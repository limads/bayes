use nalgebra::DVectorSlice;
use petgraph::graph::{DiGraph, NodeIndex};
use std::iter::{FromIterator, IntoIterator};
use petgraph::data::Build;
use petgraph::visit::Data;
use petgraph::visit::GraphBase;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use crate::fit::Estimator;
use crate::approx::Walk;

/*pub enum MFactor {

}

pub enum UFactor {

}*/

pub struct SubGraph {

    graph : Graph,

    // Edges between child nodes of subgraph and last factor at parent graph.
    edges : Vec<f64>

}

pub enum Factor {
    Bern(Bernoulli),
    Poiss(Poisson),
    Norm(Normal),
    Beta(Beta),
    Gamma(Gamma),
    JointBern(Joint<Bernoulli>),
    JointPoiss(Joint<Poisson>),
    JointNorm(Joint<Normal>),
    Walk(Walk)
}

/*impl<D> Factor<D> {

    // pub fn split_factors(mut self) ->

    pub fn normal_parent(&self) -> Option<&Normal> {
        match self {
            Factor::UParent(UFactor::Norm(ref n)) => Some(n),
            _ => None,
        }
    }

    pub fn normal_parent_mut(&mut self) -> Option<&mut Normal> {
        match self {
            Factor::UParent(UFactor::Norm(ref mut n)) => Some(n),
            _ => None,
        }
    }

    //pub fn as_mut<D>(&mut self) -> Option<&mut D> {
    //}

}*/

/* A handle to a distribution that has been added to a probabilistic graph.
Graph is indexable by this handle, which is guaranteed to return a
borrowed distribution (mutable or immutable) of the type given by its parameter */
#[derive(Debug, Clone, Copy)]
pub struct Node<D> {
    ix : NodeIndex,
    d : PhantomData<D>
}

impl<D> Node<D> {

    /* Transform this node into the equivalent Walk node, to index a posterior graph
    generated from the graph this original index refers to.
    Note indexing the original graph or any graph other than the one generated
    via Graph::posterior for the equivalent position with this value leads to a panic
    due to type mismatch */
    pub fn walk(&self) -> NodeIndex<crate::approx::Walk> {
        unimplemented!()
    }

    // Returns the raw index, used to index elements as a runtime-defined Factor.
    // Indexing a graph by Node<Factor> is always valid.
    pub fn factor(&self) -> Node<Factor> {
        Node::<Factor> { ix : self.ix, d : PhantomData }
    }

}

// TODO define this implementation (and IndexMut) for all distributions.
impl Index<Node<Normal>> for Graph {

    type Output=Normal;

    fn index(&self, ix : Node<Normal>) -> &Normal {
        unimplemented!()
    }

}

/* impl Conditional<Parent> for Children establish a directed relationship
between parent and child. Each parent-child relationship establish a topological
sorting of the graph that is useful for several sampling and optimization algorithms
like Gibbs sampling */
pub trait Conditional<P> {

}

/* Accumulator over a DAG used for log-probability calculation */
pub struct LogProb {

}

/* Accumulator over a DAG used for ancestral sampling */
pub struct Sample {

}

// Can also be called Joint<D>
// #[derive(Debug, Clone)]
pub struct Graph(DiGraph<Factor, Option<f64>, u32>);

impl Graph {

    /* Walks over this graph, in topological order. This means parents always come
    before children. This is useful both for sampling and optimization, since
    by changing the state of a node at the current iteration guarantees the changes will
    affect all children at the same iteration. Keep, e.g. a RWMHState for each children;
    and call step(.) on each children state and cond_log_prob(.) inside the closure to evolve it
    is a simple way to proceed with Gibbs sampling. */
    pub fn visit(&mut self, f : impl Fn(&mut Self, Node<Factor>)) {
        while let Some(node) = petgraph::visit::Topo::new(&self.0).next(&self.0) {
            f(self, Node::<Factor> { ix : node, d : PhantomData } );
        }
    }

    pub fn children_factors<'a>(&'a self, ix : Node<Factor>) -> impl Iterator<Item=&'a Factor> + 'a {
        let mut neighs = self.0.neighbors_directed(ix.ix, petgraph::Direction::Outgoing).detach();
        std::iter::from_fn(move || {
            neighs.next_node(&self.0).map(|node| &self.0[node] )
        })
    }

    pub fn parent_factors<'a>(&'a self, ix : Node<Factor>) -> impl Iterator<Item=&'a Factor> + 'a {
        let mut neighs = self.0.neighbors_directed(ix.ix, petgraph::Direction::Incoming).detach();
        std::iter::from_fn(move || {
            neighs.next_node(&self.0).map(|node| &self.0[node] )
        })
    }

    /*pub fn parent_factors_mut<'a>(&'a mut self, ix : Node<Factor>) -> impl Iterator<Item=&'a mut Factor> + 'a {
        let mut neighs = self.0.neighbors_directed(ix.ix, petgraph::Direction::Incoming).detach();
        std::iter::from_fn(move || {
            neighs.next_node(&self.0).map(|node| &mut self.0[node] )
        })
    }*/

    /* Calculates the log-probability of a node state, given the current state
    of its parents (if any). This is the log-probability evaluated against the
    parameter state + linear combination of parent vectors and weights. */
    pub fn conditional_log_probability<D>(&self, ix : Node<D>, x : &[f64]) -> f64 {
        0.0
    }

    /* Calculates the log-probability of a node state, irrespective of the
    state of its parents */
    pub fn log_probability<D>(&self, ix : Node<D>) -> f64 {
        0.0
    }

    pub fn sample<D>(&self, ix : Node<D>) -> Sample {
        unimplemented!()
    }

    /* Generates a sample up to the desired node by accumulating
    a value via ancestral sampling */
    pub fn conditional_sample<D>(&self, ix : Node<D>) -> Sample {
        unimplemented!()
    }

    /* If this is a probabilistic graph containing a prior + prior predictive, calling
    this method produce a structurally equivalent graph containing posterior + posterior
    predictive. The prior graph is moved because many inference algorithms operate by just
    modifying prior parameters, which might be heavy (think Kalman filters), so this by-move
    strategy guarantees performance. Moreover, if we were just mutating the prior graph, the
    original indices returned by the add(.) method could be invalidated, so the semantics that
    the indicices are invalid is more clear if we move the graph: The user can intuitively
    understand they might not be valid anymore. For specific algorithms where the index is
    still valid after posterior is called, this can be informed at the documentation.
    The estimator, however, must be passed by mutable borrow, since the user might want
    to examine its final state. If estimation fails, the prior graph is returned as the error variant. */
    pub fn posterior<E : Estimator>(model : Graph, estimator : &mut E) -> Result<Graph, E::Error> {
        unimplemented!()
    }

    pub fn new() -> Self {
        Self(DiGraph::new())
    }

    fn add_many<C>(&mut self, children : impl IntoIterator<Item=C>) -> Vec<Node<C>>
    where
        C : Into<Factor>,
        Vec<Node<C>> : FromIterator<Node<Factor>>
    {
        children.into_iter().map(|child| self.add(child.into()) ).collect()
    }

    fn add<C>(&mut self, child : C) -> Node<C>
    where
        C : Into<Factor>
    {
        unsafe { let ix = std::mem::transmute(self.0.add_node(child.into())); Node { ix, d : PhantomData } }
    }

    fn condition<C, P>(&mut self, child : Node<C>, parent : Node<P>, edges : &[f64])
    where
        C : Conditional<P>
    {
        let edge = match self.0.node_weight_mut(unsafe { std::mem::transmute(parent.ix) }).unwrap() {
            Factor::JointBern(b) => { b.add_edges(edges); None },
            Factor::JointPoiss(p) => { p.add_edges(edges); None},
            Factor::JointNorm(n) => { n.add_edges(edges); None },
            _ => {
                if edges.len() == 1 {
                    Some(edges[0])
                } else {
                    None
                }
            }
        };

        // TODO Remove transmutes. The compiler is not establishing that NodeIndex = u32 no matter the where clauses
        // set at the trait.
        unsafe { self.0.add_edge(std::mem::transmute(child.ix), std::mem::transmute(parent.ix), edge) };
    }

}

// <D> : Build + Data<NodeWeight=Factor<D>, EdgeWeight=f64> + GraphBase<NodeId=u32>

/*pub trait DAGExt {

    fn add<C>(&mut self, child : C) -> NodeIndex;

    fn condition(&mut self, a : NodeIndex<u32>, b : NodeIndex<u32>, edge : f64);

}*/

/*pub trait Condition {

    type Parent;

    fn
}*/

// impl<D> Condition<P> for DAG<D> {
// }

/*impl<D> DAGExt for DAG<D>
where
    Self : Build + Data<NodeWeight=Factor<D>, EdgeWeight=f64> + GraphBase<NodeId=u32>
{

    /*fn condition_new<C>(&mut self, c : C, parent : NodeIndex, edge : f64) -> NodeIndex
    where
        C : Conditional<D>
    {
        self.add_node(Factor::Child())
        self.add_edge()
    }*/



    // fn children() {
    // }
}*/

mod beta;

mod bernoulli;

mod categorical;

mod dirichlet;

mod gamma;

mod joint;

mod normal;

mod poisson;

mod binomial;

pub use binomial::*;

pub use beta::*;

pub use bernoulli::*;

pub use categorical::*;

pub use dirichlet::*;

pub use gamma::*;

pub use joint::*;

pub use normal::*;

pub use poisson::*;

/* Univariate/Multivariate are implemented by both analytical (Exponential) and
empirical distribution approximations (Histogram, Joint<Histogram>, Density, Empirical, Cumulative, Walk, Joint<Walk>). */

pub trait Univariate {

    // fn mean() -> f64;
    // fn variance() -> f64;
    // fn skew() -> f64;
    // fn kurtosis() -> f64;
    // fn stddev() -> f64;

}

pub trait Multivariate {

    // fn mean() -> &DVector<f64>;
    // fn covariance() -> &DMatrix<f64>;
    // fn precision() -> &DVector<f64>;

}

// Implemented only for univariate distributions. May also be called Univariate.
// If Distribution does not have a scale parameter, scale(&self) always return 1.0.
// Exponential implementors can be linked indefinitely by the Conditional trait to
// follow the left-hand path (location) in a factor tree. This means elements are always
// independent conditioned on the left-hand distribution realization; considering the right-hand
// (scale) term a constant.
pub trait Exponential {

    /// Returns the location parameter eta. This is before the link transformation
    /// is applied.
    fn location(&self) -> f64;

    /// If applicable, returns the scale parameter, applied as
    /// (self.predictor() - self.log_partition()) / self.scale()
    fn scale(&self) -> Option<f64>;

    fn link(s : f64) -> f64;

    fn link_inverse(s : f64) -> f64;

    /// Returns the RHS of the subtraction inside the exponential. The log-partition
    /// is a function of the location parameter
    fn log_partition(&self) -> f64;

    // Returns exp(-log_partition). The partition is the normalization factor
    // that premultiplies the exponential: 1/partition.
    // fn partition(&self) -> f64;

    // Returns the unscaled log probability.
    fn log_probability(&self, y : f64) -> f64 {
        self.location() * y - self.log_partition()
    }

    // Calculates the conditional log-probability of this distribution given a parent state
    // and the node weights for each parent. If the distribution has multiple parents, the
    // parent_state vector concatenate their realizations into the same slice. parent_state and
    // weights must always have the same dimensions.
    fn conditional_log_probability(&self, y : f64, parent_state : &[f64], weights : &[f64]) -> f64 {
        0.0
    }

    // fn base_measure(&self) -> f64;

    // fn log_probability(&self, y : f64) -> f64 {
    //    self.unscaled_log_probability()
    // }

    // Returns the deviate data - self.location()
    // fn error(&self, data : f64) -> f64 {
    //    data - Self::link(self.location())
    // }

    // For likelihood implementors, returns a vector-value function of the data with
    // same dimensionality as self.location() (the sufficient statistic term).
    // fn statistic(&self) -> &[f64] {
    //    unimplemented!()
    // }

    // Returns the dot-product between Self::natural(self.location()) and self.statistic().
    // fn predictor(&self) -> f64 {
    //    unimplemented!()
    // }

}

pub trait Likelihood
where
    Self : Univariate + Sized
{

    fn likelihood(data : &[f64]) -> Joint<Self>;

    // fn as_child(self) -> Factor<Self>;

}

pub trait Prior {

    fn prior(param : &[f64]) -> Self;

    // fn as_parent<L>(self) -> Factor<L>;

}

/*pub trait Posterior {

    type Error;

    fn posterior(&mut self) -> Result<(), Box<dyn Error>>;

    fn posterior_with<E>(&mut self, e : &mut E) -> Result<(), E::Error>
    where
        E : Estimator;

}*/

/*
Construct conditional probability DAGs. Dags are built backward: Start with hyperpriors, then
condition priors on hyperpriors, then condition likelihood on priors. The reason for that is that
DAG<T> is parametric over the distribution type of the leaf nodes.
*/
/*pub trait Conditional<P>
where
    Self : Sized
{

    // fn conditional() -> DAG<P>;

    // fn condition(self, parent : P, edges : &[&[f64]]) -> (DAG<Self>, NodeIndex, NodeIndex);

    // fn split(dag : &DAG<Self>) -> (&Self, &P);

    // fn split_mut(dag : &mut DAG<Self>) -> (&mut Self, &mut P);

}*/

/// Implement Joint<Normal, Association=f64> for Normal AND Joint<Categorical, Association=Contingency> for Categorical.
/// Also Joint<[Normal]> for Normal to build convergent graph structure.
pub trait Marginal
where
    Self::Univariate : Univariate
{

    type Univariate;

    // Builds a joint distribution from self (assumed to be one of the marginals after joining) and other, assumed
    // to be another marginal after joining. Joint is parametrized by the marginal correlations between the variables.
    // This returns Join<T>, with self being accessable by marginal(n) and all others being accessible by marginal (n+k)
    // where k is the slice of marginals.
    fn joint<I, const N : usize>(distrs : I) -> Joint<Self::Univariate>
    where
        Self : Sized,
        I : IntoIterator<Item=Self::Univariate>;

    fn len(&self) -> usize;

    // Access a marginal distribution from this joint distribution.
    fn marginal(&self, ix : usize) -> Self::Univariate;

    // fn set_marginals(&mut self, )

}

/*fn condition_empty<A, B>(a : A, b : B, edges : &[f64]) -> (DAG<A>, NodeIndex, NodeIndex)
where
    A : Conditional<B>,
    B : Prior
{
    assert!(edges.is_empty());
    let mut dag = DiGraph::new();
    let lik = dag.add_node(Factor::Child(a));
    let prior = dag.add_node(b.as_parent());
    dag.add_edge(lik, prior, 1.);
    unsafe { (dag, std::mem::transmute(lik), std::mem::transmute(prior)) }
}*/

// #[test]
// fn conjugate_inference() {
//    let graph = Bernoulli::conditional(&[0.2], Beta::prior(&[0.1, 0.2]));
// }

#[test]
fn linear_regression() {
    let mut graph = Graph::new();
    /*let prior = graph.add(Normal::joint([Normal::prior(&[0., 1.]), Normal::prior(&[0.1, 0.2])]));
    let ys = graph.add_many(Normal::likelihood(&[1., 2., 3., 4., 4., 2., 1., 1.]));
    for y in ys.drain(..) {
        graph.condition(y, prior, &[0., 1.]);
    }*/
}

/*pub trait Marginal<U>
where
    U : Univariate
{

    pub fn marginals_mut<'a>(&'a mut self) -> Vec<&'a mut U>
}*/


