# Heteroscedasticity

Heteroscedasticity is represented in `bayes` by having a collection of multinormals:

```rust
let normals : Box<[Normal]> = (0..10)
    .map(|i| Normal::likelihood([x[i]]).set_variance(v[i]) )
    .collect();
```

Just calling likelihood for a single data variate would leave the mean at the value
and the variance at zero, since this is the variance MLE for a single observation.

We know the MLE/MAP estimates are calculated by a weighted average, where the
weights are the precisions (inverse variance) of each observation. Therefore,
`[Normal]` (the normal slice agnostic to allocation) satisfies conditioning
on `Normal` and `MultiNormal` for calculation of a common scale parameter.
