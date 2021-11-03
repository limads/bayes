# Generalized linear models

Generalized linear models have one characteristic: The predictive distribution
$p(y|x,w)$ is different from the likelihood $p(z|x,w)$ where: 

Eq. 1.

$$
z = y - E[y|x]$ (\frac{\partial l}{\partial w})
$$

The heteroscedastic normal z  is used as an approximation for the predictive y, 
where the approximation is thought to be close enough near the maximum likelihood 
estimate. This transformation allows one to find the distance to the MLE by recursively 
calculating $E[y|x]$ with the current MLE values for w.

The point is how to represent this transformation in a probabilistic graph. Suppose
we can parametrize any univariate exponential family distribution in its natural
form, for example by conditioning a Bernoulli at its logit instead of its probability,
or conditioning a Poison at is log(rate) instead of its rate. We represent this
possibility by letting exponential-family univariate factors be conditioned on Normal
in addition to letting them be conditioned on their conjugate parent factors. If
an univariate exponential is conditioned on a Normal, the realization is assumed
to represent the parameter in natural form, instead of in its expected-value form.
This agrees nicely with the math, since any exponential family in its natural form
(where the scalar product of the natural parameter with the sufficient statistic gives
the location for the likelihood realization) has likelihood which is identical in its
domain and shape to the normal likelihood (while in the non-natural scale the likelihood
domain might have a distribution-specific shape and domain).

Since in a GLM each observation has a separate expected value, we need to condition
the realization over a different Normal, and jointly condition those normals in a 
MultiNormal that will yield the parameter vector estimate. This part of the model
is identical to the heteroscedastic estimation setup. The variance for each realization
is given by $Eq. 1$.

This normal, in its turn, is conditioned on a set of predictors. The estimated
$\eta$ is accessed by looking at the location parameter of each of those normals;
the conditional expectations $\hat E[y|x]$ is accessed by `Bernoulli::link(eta)`.
fixed (has zero variance) since it is a linear function of the parameters, which already 
capture the variance.

# From conjugate sampling to GLMs

The Bernoulli conditioned on Beta conjugate model, where $E[y|\theta]$ is calculated by first sampling $\theta$, is equivalent to the Bernoulli conditioned on a Normal centered at $\alpha / (\alpha + \beta)$ from which we sample

$$
\eta = ln(\theta / (1. - \theta))
$$

Then calculate

$$
\theta = 1 / (1 + e^{-\eta})
$$

Although this form loses the conjugacy property (where the posterior is just a weighted sum of the prior and likelihood), it will be useful for estimation because we can explore all properties of the Normal, including the extensive ways we can combine it using conditioning to build linear models.

# Fixed Multinormals

Fixed multinormals are built by first calculating a MultiNormal likelihood then "fixing" the values (setting their probability to one), which effectively creates a Normal child node conditioned on all the parent fixed nodes.

```
let a = 1;
println!("Log message");
a
/*impl Fixed for [MultiNormal] {
    pub fn fix(self) -> Box<[Normal]>;
}

impl Fixed for MultiNormal {
    pub fn fix(self) -> Normal;
}

// Builds the undirected joint graph. MultiNormal will not allocate a covariance
// when a single observation is made.
let mn : Box<[MultiNormal]> = [[x1], [x2]].iter().map(MultiNormal::likelihood).collect();

// Builds the directed graph towards new node mu, with sufficient statistic given by the fixed values.
let mut mu : Box<[Normal]> = mn.fix();

// Append prior
mu.condition(MultiNormal::prior([0.1., 0.2]);

// Append likelihood, using impl Condition<[Normal]> for [Bernoulli].
y.condition(mu);

// Were we building a WLS regression, we would set the variances here.
y.for_each(|y| y.set_scale(1.0) );*/
```

# Example

The logistic regression can be represented as:

```rust
/*let bern : Box<[Bernoulli]> = Bernoulli::likelihood(y[i]);
let eta = (0..100)
    .map(|i| MultiNormal::likelihood([bern[i].error(y[i])), x1, x2]) )
    .collect()
    .fix(1..); 
eta.condition(MultiNormal::prior(0.1, 0.3));

// [Bernoulli] satisfies Condition<[Normal]>.
bern.condition(eta);*/
let a = 1;
a
```

Since applying conditioning to two iterators of distributions is a common pattern, 
bayes has the conditional(.) and fixed(.) iterator adaptors. The `conditional(.)` adaptor takes two  iterators over A and B where `A : Condition<B>` and returns the iterator over A, but conditional on B. The fixed calls fix over each element of the iterator.

```rust
/*let n = (0..100)
    .map(|i| MultiNormal::likelihood([bern[i].error(y[i])), x1, x2]) )
    .fixed(1..);
let y : Box<[Bernoulli]> = (0..100).map(|i| Bernoulli::likelihood(y[i]) ).conditional(n).collect();*/
let a = 1;
a
```

Since conditioning over a common factor is also a common pattern, the iterator adaptor
jointly_conditional(.) also accepts a common element, which will condition all elements
over it after collecting the elements into a container. 

Alternatively,

```rust
/*y.iter().map(|y| 
    Bernoulli::likelihood([y])
        .condition(MultiNormal::likelihood([x1..xn]).fix(..) ) 
    ).collect::<Box<[Bernoulli]>>()
    .condition(MultiNormal::prior([0.1, 0.2, 0.3]));*/
let a = 1;
a
```

This match is 

$$
x^2 + 10
$$


