/// Utilities for dimensionality reduction of structured high-dimensional data.
pub mod dim;

/// Clustering algorithms, useful to classify structured but unlabelled data.
pub mod cluster;

// Structures to represent ordered 1D data (time series) and their frequency/scale transformations.
// pub mod signal;

// Structures to represent ordered 2D data (images) and their frequency/scale transformations.
// pub mod image;

// Text-related features (dictionaries, syntax trees, etc).
// pub mod text;

/*
// If you are working on a problem, and &[T] cannot be considered conditionally-independent
// on the model, after applying a feature-extraction algorithm, Feature<T> such as Cluster<T>
// or Match<T> or Component<T> or Discriminant<T> can be considered
// to be conditionally-independent given the model.
pub trait Feature {

} */
