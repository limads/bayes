use nalgebra::*;
use std::iter::FromIterator;

pub mod table;

pub use table::*;

use rand;

/* By leaving the observation type as IntoIterator<f64> we have the possibility
of returning both owned Vec<f64> for structures that must pool data before
returning them; or non-owned &[f64] for structures that should just access
elements in a column-wise fashion. The outer-most vector is just a container
for columns. The columns themselves can be slices (read) or vectors (polled).
Returns an empty vector if the (offset, batch) pair refers to an invalid
position in the data matrix.

row_iter(offset, len) can be a provided method that takes batch(len) and builds
a vector iterator over the rows.
impl Sample<O>
where
    O : IntoIterator<&f64>
{

    pub fn batch(offset : o, n : usize) -> Vec<O>;

    pub fn shape() -> (usize, usize);
}

impl Sample for std::str::Lines()
impl Sample for Stream<Item=Database::Row>
*/

/// Samples are types which hold independent (or at least conditionally independent)
/// observations, and interface directly with the likelihood of probabilistic models.
/// The ability to iterate over those observations via conversion into
/// a dynamically-allocated matrix is the basis for calculating
/// sufficient statistics for arbitrary probability distributions. Splitting its rows and
/// columns is the basis for generic validation and ensemble algorithms.
///
/// Some algorithms are cheaper to execute if individual observations are organized
/// contiguously over memory (such as metric calculations); Others if the variables
/// are organized contiguously over memory (such as log-probability calculations). Sample
/// implementors are agnostic to which scheme is used. Elements can even have a sparse
/// representation (where the order is defined by a vector of indices; and variables might not
/// be contiguous over memory). This trait must satisfy Into<DMatrix<f64>> (organized over columns;
/// or over rows if transposed), which guarantees that a packed representation is
/// always achievable for either case.
///
/// Iterating over observations in a way agnostic to the representation guarantees a goods performance x
/// generality tradeoff for validation, resampling and ensemble generic algorithms, since a DMatrix can
/// always be built from the returned iterator. For observation-contiguous samples, splitting over rows is
/// cheap but over columns is expensive; for variable-contiguous samples, splitting over columns is cheap but
/// over rows is expensive.
///
/// The observation type parameter O this sample refers to is meant to be a cheap reference type,
/// such as RowDVector<'_, f64> or &[f64] (If the sample is observation-contiguous).
pub trait Sample<'a, O>
    where
        Self : Into<DMatrix<f64>>,
        O : Copy
{

    fn nrows(&self) -> usize;

    fn ncols(&self) -> usize;

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn split_rows(self, row : usize) -> (Self, Self);

    fn split_columns(self, col : usize) -> (Self, Self);

    /// Returns a block of observations from the implementor.
    fn observations(&'a self, from : usize, len : usize) -> Vec<O>;

    /// If the implementor is wide, returns iterator over the rows.
    fn units(&'a self) -> Option<Box<dyn Iterator<Item=&'a [f64]>>>;

    /// If the implementor is tall, returns iterator over columns.
    fn variables(&'a self) -> Option<Box<dyn Iterator<Item=&'a [f64]>>>;

    /// Erase all observations from the implementor, and re-populate
    /// it using the informed observations.
    fn repopulate(&mut self, obs : &[O]);

    /// Useful for algorithms requiring subsampling. Keep one implementor
    /// with size m < n; and a full implentor with size n. Draw a set of samples
    /// (ix_1 ... ix_m); and copy them into the subsample. Indices passed at
    /// ignore (if any) are not used in the resampling.
    fn draw_from(&mut self, other : &'a Self, n : usize, ignore : Option<&[usize]>) {
        let mut ord_ignore = Vec::from_iter(
            ignore.unwrap_or(&[])
            .iter()
            .cloned()
        );
        ord_ignore.sort();
        let mut selected_ix = Vec::with_capacity(n);
        for _ in 0..n {
            let ix = loop {
                let ratio : f64 = rand::random();
                let cand_ix = ((other.nrows() as f64) * ratio) as usize;
                let in_ignore = ord_ignore.binary_search(&cand_ix).is_ok();
                let in_selected = selected_ix.binary_search(&cand_ix).is_ok();
                if !in_ignore && !in_selected {
                    break cand_ix;
                }
            };
            if let Err(pos) = selected_ix.binary_search(&ix) {
                selected_ix.insert(pos, ix);
            }
        }
        let selected : Vec<O> = selected_ix.iter()
            .map(|ix| other.observations(*ix, 1)[0] )
            .collect();
        self.repopulate(&selected[..]);
    }

}


