// Stochastic processes characterized by the Markov property : The future is independent of
// the past conditional on the current state.
// pub mod state;

// Random processes (Process<MultiNormal> and Process<Dirichlet>)
// pub mod process;

/// Utilities for dimensionality reduction of structured high-dimensional data.
pub mod dim;

/// Clustering algorithms, useful to classify structured but unlabelled data.
pub mod cluster;

/// Structures to represent ordered 1D data (time series) and their frequency/scale transformations.
pub mod signal;

/// Structures to represent ordered 2D data (images) and their frequency/scale transformations.
pub mod image;

// Text-related features (dictionaries, syntax trees, etc).
// pub mod text;

