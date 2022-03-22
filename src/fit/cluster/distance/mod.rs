use nalgebra::*;

// Build the "wide" data matrix (observations are columns) from an observation iterator.
pub(crate) fn build_wide_matrix<'a, I, C>(obs : I) -> DMatrix<f64>
where
    I : Iterator<Item=C>,
    C : IntoIterator<Item=&'a f64>
{
    let vecs : Vec<_> = obs
        .map(|item| DVector::from_vec(item.into_iter().cloned().collect::<Vec<f64>>()) )
        .collect();
    DMatrix::from_columns(&vecs[..])
}

/// Builds a generic distance matrix from a arbitrary metric defined for a given structure.
/// complete specifieds whether the lower-triangular part of the matrix should be filled. If
/// false, the lower-triangular part will be set to zeros.
pub(crate) fn generic_distance_matrix<T, F>(obs : &[T], metric : F, complete : bool) -> DMatrix<f64>
where
    F : Fn(&T, &T)->f64
{
    let n = obs.len();
    let mut dist = DMatrix::zeros(n, n);

    // Store measure entry results in upper-triangular portion
    for i in 0..n {
        for j in i..n {
            dist[(i, j)] = metric(&obs[i], &obs[j]);
        }
    }

    if complete {
        // Copy entries to lower-triangular portion
        for j in 0..n {
            for i in 0..j {
                dist[(j, i)] = dist[(i, j)]
            }
        }
    }
    dist
}

/// Builds a square euclidian distance matrix from a sequence
/// of observations, each assumed to have the same dimension. Requires that
/// the data is in "tall" form (observations are rows). Passing a wide
/// matrix saves up one transposition operation. The crate strsim-rs
/// can be used to build custom text metrics. The user can write any
/// Fn(&T, &T)->f64 to serve as a custom object metric.
pub(crate) fn distance_matrix(tall : DMatrix<f64>, opt_wide : Option<DMatrix<f64>>) -> DMatrix<f64> {
    let wide = opt_wide.unwrap_or(tall.transpose());

    // Build the Gram matrix (Matrix of dot products)
    // Reference: https://en.wikipedia.org/wiki/Euclidean_distance_matrix#Relation_to_Gram_matrix
    let gram = tall * wide;

    assert!(gram.nrows() == gram.ncols());

    // Calculate the Euclidian distance matrix from the Gram matrix
    let mut eucl = DMatrix::zeros(gram.nrows(), gram.ncols());
    for i in 0..gram.nrows() {
        for j in 0..gram.ncols() {
            eucl[(i, j)] = gram[(i, i)] - 2. * gram[(i, j)] + gram[(j, j)];
        }
    }

    // All diagonals must be zero (Eucl. dist. matrix is hollow)
    for i in 0..eucl.nrows() {
        assert!(eucl[(i,i)] < 1E-8);
    }

    // All distances must be non-negative
    for i in 0..eucl.nrows() {
        for j in (i + 1)..eucl.ncols() {
            assert!(eucl[(i, j)] >= 0.0);
        }
    }

    // Matrix must be symmetric
    for i in 0..eucl.nrows() {
        for j in 0..eucl.ncols() {
            assert!(eucl[(i, j)] - eucl[(j, i)] < 1E-8);
        }
    }

    eucl
}

