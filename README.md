# About

This is a **work-in-progress** crate that will offer composable abstractions to build probabilistic models and inference algorihtms. Two reference algorithms will be implemented in the short term: the `optim::ExpectMax` (general-purpose posterior mode-finding via expectation maximization) and `sim::Metropolis` (Metropolis-Hastitings posterior sampler). Adaptive estimation from conjugate pairs will also be provided. Most of the functionality is being implemented using the linear algebra abstractions from the [nalgebra](https://crates.io/crates/nalgebra) crate. Certain optimization and basis expansion algorithms are provided via bindings to GNU GSL and Intel MKL (Optional).

# Usage

## Model building

The building blocks of probabilistic models are the `Distribution` implementors living under the `distr` module:

- `distr::Bernoulli` for univariate binary outcomes;

- `distr::Beta` for binary priors;

- `distr::Poisson` for univariate count outcomes;

- `distr::Gamma` for generic inverse-scale priors;

- `distr::Normal` for univariate continuous outcomes and location priors;

- `distr::MultiNormal` for multivariate continous outcomes, location priors and random natural parameters;

- `distr::NormalMixture` for univariate or multivariate marginalized continous outcomes;

- `distr::Wishart` for multivariate structured inverse-scale priors;

- `distr::Categorical` for multivariate mutually-exclusive discrete outcomes;

- `distr::Dirichlet` for categorical priors.

- `distr::VonMises` for circular continuous outcomes and direction priors.

Probabilistic models are built by conditioning any `Distribution` implementor on another valid target distribution:

```
let b = Bernoulli::new(&[0.5]).condition(Beta::new(1,1));
```

This conditioning operation is defined for implementors of `Conditional<Factor>`. This trait is implemented for:

- All conjugate pairs: (Beta-Bernoulli; Normal-Normal, etc);

- Distributions conditioned on a random natural parameter factor (classical generalized linear models: Poisson-MultiNormal; Bernoulli-MultiNormal; Categorical-MultiNormal);

- Distributions that are conditionally-independent over a scale factor (Normal; MultiNormal; VonMisses);

- A mixture and its discrete categorical draw.

Deep probabilistic graphs can in principle be built as long as the neighboring elements have valid `Conditional<Factor>` implementations; although their usability for any given problem is determined by the inference algorithm implementation.

Conditioning takes ownership of the conditioning factor, which can be recovered via:

```
let factor : Option<Beta> = b.take_factor();
# or
let factor : Option<&Beta> = b.view_factor();
```

For deeper probabilistic graphs, you will also be able to use:

```
let factor : Option<&Beta> = b.find_factor();
```

Which will search the graph and return the first match. Graph iteration is done from the unique top-level element to all its roots; then from left-hand-side to right-hand side. Location or direction factors are to the left-hand side; conditionally independent scale factors to the right-hand side.

## Inference (planned)

Inference algorithms are `Estimator<Target>` implementors. Such algorihtms take a probabilistic graph (represented by the top-level distribution) and returns a modified, but related graph (potentially with prior nodes removed) that holds some kind of posterior distribution representation: The graph state might represent a posterior mode; or marginal representations might be recovered from the `RandomWalk::marginal` method.

```r
let mut metr = Metropolis::new(distr);
let post = metr.fit(y).unwrap();
println!("{}", post.view_factor::<MultiNormal>.unwrap().marginal());
```

Certain inference algorithms (usually satisfying a conjugate structure in shallow probabilistic models) can be updated sequentially by a cheap parameter update:

```r
let y = DMatrix::from_column_slice(3, 1, &[0., 1., 0.]);
let b = Bernoulli::new().condition(beta);
let b : Option<&Beta> = b.fit(&y);
```

## Decision (planned)

Any two distribution implementors can have their relative log-probabilities compared, which is useful for the objectives of selection of two alternative models modified in some selective way (for example, to examine the robustness of an inference procedure to prior specification; or for variable selection). The `decision::BayesFactor` generic structure solves this problem, in a way completely agnostic to model specification. 

If the ultimate objective of an inference is a binary decision, the log-probability difference which maximizes some conditional or marginal error criterion can be found via optimization over a target, known binary outcome vector. The `decision::DecisionBoundray` generic structure solves this problem.

## Basis expansion (planned)

Several non-linear processes (time series; images) can only be modeled meaningfully using some kind of basis expansion. The crate wraps some standard implementation of those algorithms (FFT and Wavelets for now), which work with `nalgebra::DMatrix` and `nalgebra::DVector` structures for convenience.

## Serialization and graphical representation (planned)

A probabilistic model (both a prior and a sampled posterior) will admit a serialized JSON representation, which is a convenient way to tweak, fit and compare separate models, without requiring source changes. Also, a model will have a graphical (.png/.svg) representation, offered by the `petgraph` crate (via `graphviz`). 

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

# License

This crate is licensed under the [LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html).


