/// Load data from tabular sources to in-memory double precision matrices.
/// Support csv files or relational databases.
pub mod table;

/// Load data from generic 8-bit time stream buffers
pub mod seq;

/// Load data from generic 8-bit image buffers
pub mod surf;

pub use seq::*;

pub use surf::*;

