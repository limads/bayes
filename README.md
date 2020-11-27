[![Crates.io](https://img.shields.io/crates/v/bayes?style=flat-square)](https://crates.io/crates/bayes)

# About

This is a **work-in-progress** crate that will offer composable abstractions to build probabilistic models and inference algorithms. Two reference algorithms will be implemented in the short term: the `optim::ExpectMax` (general-purpose posterior mode-finding via expectation maximization) and `sim::Metropolis` (Metropolis-Hastings posterior sampler). Adaptive estimation from conjugate pairs will also be provided (see examples for the `distr::Normal`, `distr::Poisson` and `distr::Binomial` structs). 

Most of the functionality is being implemented using the linear algebra abstractions from the [nalgebra](https://crates.io/crates/nalgebra) crate. Certain optimization, sampling and basis expansion algorithms are provided via [GNU GSL](https://www.gnu.org/software/gsl/doc/html/intro.html) (Required system dependency) and [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html), (Optional system dependency; by switching the cargo feature `features=["mkl"]`).

# Usage

## Model building

The building blocks of probabilistic models are the `Distribution` implementors living under the `distr` module:

- `distr::Bernoulli` for univariate binary outcomes;

- `distr::Beta` for proportion priors;

- `distr::Poisson` for univariate count outcomes;

- `distr::Gamma` for generic inverse-scale or rate priors;

- `distr::Normal` for univariate continuous outcomes and location priors;

- `distr::MultiNormal` for multivariate continous outcomes, location priors and random natural parameters;

- `distr::NormalMixture` for univariate or multivariate continous outcomes marginalized over a discrete factor;

- `distr::Wishart` for multivariate structured inverse-scale priors;

- `distr::Categorical` for multivariate mutually-exclusive discrete outcomes;

- `distr::Dirichlet` for categorical priors.

- `distr::VonMises` for circular continuous outcomes and directional priors.

Creating a single distribution object allow just sampling and calculating summary statistics from its currently-set parameter vector. You will be able to build more complex probabilistic models by conditioning any `Distribution` implementor on another valid target distribution:

```rust
let b = Bernoulli::new(100, None).condition(Beta::new(1,1));
```

This conditioning operation is defined for implementors of `Conditional<Factor>`. This trait is implemented for:

- All conjugate pairs: (Beta-Bernoulli; Normal-Normal, etc);

- Distributions conditioned on a random natural parameter factor (classical generalized linear models: Poisson-MultiNormal; Bernoulli-MultiNormal; Categorical-MultiNormal);

- Distributions that are conditionally-independent over a scale factor (Normal; MultiNormal; VonMisses);

- A mixture and its discrete categorical factor.

More complex probabilistic graphs can in principle be built as long as the neighboring elements have valid `Conditional<Factor>` implementations; although their usability for any given problem is determined by the inference algorithm implementation.

Conditioning takes ownership of the conditioning factor, which can be recovered via:

```rust
let factor : Option<Beta> = b.take_factor();
// or
let factor : Option<&Beta> = b.view_factor();
```

To recover factors from probabilistic graphs with more than one level, you will also be able to use:

```rust
let factor : Option<&Beta> = b.find_factor();
```

Which will search the graph and return the first match. Graph iteration is done from the unique top-level element to all its roots; then from left-hand-side to right-hand side. Location or direction factors are to the left-hand side; conditionally independent scale factors to the right-hand side.

## Adaptive conjugate inference

Certain inference algorithms (usually satisfying a conjugate structure) can be updated sequentially by a cheap parameter update:

```rust
let y = DMatrix::from_column_slice(3, 1, &[0., 1., 0.]);
let bern = Bernoulli::new(3, None).condition(Beta::new(1,1));
let posterior : Result<&Beta,_> = b.fit(&y);
```

## General-purpose inference (planned)

Inference algorithms are determined by any `Estimator<Target>` implementors (which require a `fit(sample)->Result<Distribution,Err>` implementation). All conjugate pairs implement this trait; but any structure that accepts a probabilistic model at construction and maintain it as an internal state can also implement this trait. The returned distribution will be a modified version of the  received probabilistic model: Optimizers can leave the graph in a state that maximizes the log-probability; Samplers can build a non-parametric representation to the posterior whose marginal is held by the corresponding graph node.

## Decision (planned)

Any two distribution implementors can have their relative log-probabilities compared, which is useful for the objectives of selection of two alternative models modified in some selective way (for example, to examine the robustness of an inference procedure to prior specification; or for variable selection). The `decision::BayesFactor` generic structure solves this problem, in a way completely agnostic to model specification. 

If the ultimate objective of an inference is a binary decision, the log-probability difference which maximizes some conditional or marginal error criterion can be found via optimization over a target, known binary outcome vector. The `decision::DecisionBoundray` generic structure solves this problem.

## Basis expansion (planned)

Several non-linear processes (time series; images) can only be modeled meaningfully using some kind of basis expansion. The crate wraps some standard implementation of those algorithms (FFT and Wavelets for now), which work with `nalgebra::DMatrix` and `nalgebra::DVector` structures for convenience.

## Serialization and graphical representation (planned)

A probabilistic model (both a prior and a sampled posterior) will admit a JSON representation, which is a convenient way to tweak, fit and compare separate models using a high-level API (such as the command line), without requiring source changes. Also, a model will have a graphical (.png/.svg) representation, offered by the `petgraph` crate (via `graphviz`).

# Development status

The basic abstractions are already in place and a few implementations for sampling and calculation of log-probabilities are already working, but the crate is still not usable yet. The short-term goals for the Crate are:

1. [X] Basic abstractions (distributions; estimation algorithm traits)

2. [X] GSL/MKL bindings

3. [ ] Conjugate pair AdaptiveEstimator implementations

4. [ ] DecisionBoundary implementation

5. [ ] ExpectMax implementation

6. [ ] Metropolis implementation 

7. [ ] Probabilistic model parsing from JSON

8. [ ] Graph representation for proabilistic models

# System requirements

`libgsl` (Gnu scientific library)

`libmkl` (Intel Math Kernel Library; Optional, via `features=["mkl"]`)

# Usage

You might want to use bayes in different ways depending on you setup:

- Command-line usage: Type `bayes --help` to perform inference at the command line. This is the
easiest way to get started, and is fine to process small volumes of data structured as CSVs.

- Web service: The command bayes serve -h [host] -p [port] will open a hyper server from which clients
can instantiate and query models. This is useful as a component of a microserver architecture, or to 
preserve a model across many call of fit/predict. With a model setup, you can work with any scripting
language (Python/R/Js or even plain SQL by interpolating results from curl) to create new models, modify 
a model, or predict from it.

# License

This crate is licensed under the [LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).


