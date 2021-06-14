use nalgebra::*;
use std::iter::FromIterator;
use std::collections::HashMap;
use std::cmp::Eq;
use std::hash::Hash;
use ::csv::{Reader, StringRecord};
pub mod csv;
use postgres;
use either::Either;
use std::iter::{self, Iterator};
use std::ops::Index;

use rand;

/// Variable iterator. Perhaps consider it &'a dyn Iterator<Item=T>
pub type VarIter<'a, T> = Box<dyn Iterator<Item=T> + 'a>;

/*pub enum Observation<'a> {
    Real(f64),
    Count(usize),
    Binary(bool),
    Factor(&'a str),
    Missing
}*/

/// Represents a memory-contiguous region on non-missing data
pub enum Column<'a> {
    Real(&'a [f64]),
    Count(&'a [usize]),
    Binary(&'a [bool]),
    Factor(&'a [&'a str])
}

impl<'b> Sample for HashMap<&str, Column<'b>> {

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        match self.get(name) {
            Some(Column::Real(r)) => Variable::Real(Box::new(r.iter().cloned())),
            Some(Column::Count(c)) => Variable::Count(Box::new(c.iter().cloned())),
            Some(Column::Binary(b)) => Variable::Binary(Box::new(b.iter().cloned())),
            Some(Column::Factor(f)) => Variable::Factor(Box::new(f.iter().cloned())),
            _ => Variable::Missing
        }
    }

}

/// Represents a complete, but not necessarily contiguous region of data.
pub type Real<'a> = Box<dyn Iterator<Item=f64> + 'a>;

pub type Count<'a> = Box<dyn Iterator<Item=usize> + 'a>;

pub type Binary<'a> = Box<dyn Iterator<Item=bool> + 'a>;

pub type Discrete<'a> = Box<dyn Iterator<Item=&'a str>>;

/// Represents an incomplete and not necessarily contiguous region of data.
pub type Missing<T> = Box<dyn Iterator<Item=Option<T>>>;

pub enum Variate<V> {
    Complete(V),
    Missing(Missing<V>),
    Unavailable
}

// Implemented by estimators which can iterate over the graph and attribute observations
// to named distributions.
/*pub trait Model
where
    Self : Estimator
{

    fn observe_sample(&mut self, sample : &dyn Sample);

}*/

/* The bayes crate makes the assumption that a sample is only independent given fixed
values for the model parameters. Correlations in the sample are accounted for when
the model is fully specified. This is why we view data "observations" as part of the model
definition step (models cannot be specified independent of the data they refer to). */

/// Generic trait for structures that contain random samples. If you store your observations
/// into a custom type T, You usually want to implement Variate for a container of T, such as
/// Vec<T> or HashMap<str, T>.
pub trait VariateTrait<'a> {

    fn real(&'a self, var : &str) -> Variate<Real<'a>>;

    fn count(&'a self, var : &str) -> Variate<Count<'a>>;

    fn binary(&'a self, var : &str) -> Variate<Binary<'a>>;

    fn discrete(&'a self, var : &'a str) -> Variate<Discrete<'a>>;

}

/*pub struct Table<'a> {
    content : HashMap<String, Column<'a>>
}

impl Table {

    pub fn new(content : &[(&str, Column)]) {
        let mut content = HashMap::new();
        for (key, col) in content.iter() {
            content.push(key.to_string(), content);
        }
        Self{ content }
    }
}*/

/* Example for custom user struct

pub struct Observation {
    days : usize,
    profits : f64
}

impl<'a> VariateTrait<'a> for Vec<Observation> {

    fn real(&'a self, var : &str) -> Variate<Real> {
        match var {
            "profits" => Variate::Complete(Box::new(self.iter().map(|o| o.profits ))),
            _ => Variate::Unavailable
        }
    }

    fn count(&'a self, var : &str) -> Variate<Count> {
        match var {
            "profits" => Variate::Complete(Box::new(self.iter().map(|o| o.days ))),
            _ => Variate::Unavailable
        }
    }

    fn binary(&'a self, var : &str) -> Variate<Binary> {
        Variate::Unavailable
    }

    fn categorical(&'a self, var : &'a str) -> Variate<Categorical<'a>> {
        Variate::Unavailable
    }
}*/

/*impl<'a> VariateTrait<'a> for HashMap<&str, Column<'a>> {

    fn real(&'a self, var : &str) -> Variate<Real> {
        match self.get(var) var {
            Some(Column::Real(r)) => Variate::Complete(Box::new(self.iter().map(|o| o.profits ))),
            _ => Variate::Unavailable
        }
    }

    fn count(&'a self, var : &str) -> Variate<Count> {
        match var {
            "profits" => Variate::Complete(Box::new(self.iter().map(|o| o.days ))),
            _ => Variate::Unavailable
        }
    }

    fn binary(&'a self, var : &str) -> Variate<Binary> {
        Variate::Unavailable
    }

    fn categorical(&'a self, var : &'a str) -> Variate<Categorical<'a>> {
        Variate::Unavailable
    }

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        match self.get(name) {
            Some(Column::Real(r)) => Variable::Real(Box::new(r.iter().cloned())),
            Some(Column::Count(c)) => Variable::Count(Box::new(c.iter().cloned())),
            Some(Column::Binary(b)) => Variable::Binary(Box::new(b.iter().cloned())),
            Some(Column::Factor(f)) => Variable::Factor(Box::new(f.iter().cloned())),
            _ => Variable::Missing
        }
    }

}*/

/*impl<V> Sample for HashMap<&str, &dyn FnMut()->V>
where
    V : Iterator<Item=Variable>
{

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {

    }
}*/

/*impl<'b, T> Sample for Vec<T>
where
    T : Index<&'b str, Output=Observation<'b>>
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        let mut real = None;
        let mut bin = None;
        let mut fac = None;
        let mut count = None;
        for el in self.iter() {
            match el[name] {
                Observation::
            }
        }
    }
}*/

//TODO rename to Variate
/// A variable type from a sample. Essentially, wraps one of all the possible
/// iterators that can be used to retrieve data. Does not imply how the data
/// is organized in memory (it might come from a packed vector, or unpacked JSON/database record),
/// which is why it wraps a boxed dynamic iterator. Each
/// distribution is interested in exactly one of the variants, and it will check
/// that the required variable name is admissible at runtime. Although all calculations
/// by the crate are done in double precision, we present the Sample/Variable API which
/// is a type safe way to match outside-world data to the data required by each Distribution:
/// | Distribution | Required variable type |
/// |--------------|------------------------|
/// | Gamma        | Real (f64 iterator)    |
/// | Normal       | Real (f64 iterator)    |
/// | Bernoulli    | Binary (bool iterator) |
/// | Categorical  | Factor (&str iterator) |
/// | Poisson      | Count (usize iterator) |
/// Variates do not really store the data: They are just rules to iterate over a container
/// structure that has possibly heterogenous data. They are lightweight structures that tell
/// the distributions how to pull data, irrespective of how this data is laid out in memory.
pub enum Variable<'a> {

    /// Real, unbounded quantities, usually read by Normal, Gamma or MultiNormal nodes
    Real(VarIter<'a, f64>),
    
    /// Countable quantities (>=0), usually read by Poisson nodes
    Count(VarIter<'a, usize>),
    
    /// Binary quantities, usually read by Bernoulli nodes
    Binary(VarIter<'a, bool>),
    
    /// Factor (named) nodes, usually read by Categorical nodes
    Factor(VarIter<'a, &'a str>),
    
    /// Variable not found.
    Missing
    
}

/*impl<'a, I, F> From<iter::Map<I, F>> for Variable<'a>
where
    I : Iterator<Item=f64> + 'a,
    F : FnMut(f64)->f64 + 'a
{
    fn from(s : iter::Map<I, F>) -> Self {
        Variable::Real(Box::new(s))
    }
}*/

/*impl<'a, I, F> From<iter::Map<I, F>> for Variable<'a>
where
    I : Iterator<Item=usize> + 'a,
    F : FnMut(usize)->usize + 'a
{
    fn from(s : iter::Map<I, F>) -> Self {
        Variable::Count(Box::new(s))
    }
}*/

/*impl<'a, I> From<I> for Variable<'a>
where
    I : Iterator<Item=f64> + 'a
{
    fn from(s : I) -> Self {
        Variable::Real(Box::new(s.into_iter()))
    }
}

impl<'a, I> From<I> for Variable<'a>
where
    I : Iterator<Item=usize> + 'a
{
    fn from(s : I) -> Self {
        Variable::Count(Box::new(s.into_iter()))
    }
}

impl<'a, I> From<I> for Variable<'a>
where
    I : Iterator<Item=bool> + 'a
{
    fn from(s : I) -> Self {
        Variable::Binary(Box::new(s.into_iter()))
    }
}

impl<'a, I> From<I> for Variable<'a>
where
    I : Iterator<Item=&str> + 'a
{
    fn from(s : I) -> Self {
        Variable::Factor(Box::new(s.into_iter()))
    }
}*/

/// A sample is a data structure from which three iterators can be retrieved:
/// One iterator over variable names (&str), another iterator over potentially non-contiguous row values (f64),
/// and another iterator over contiguous columns values &[f64]. No assumption
/// is made about how the names or values are laid out in memory, so the user can return both slices or
/// vectors built on-demand depending on how the structure organizes the data. Reading happens either row-wise
/// (A probabilistic model reads the first row (having the same layout as the variable names; then
/// the second row, and so on) or column-wise. The user might chose to implement only one iteration strategy
/// (and return a plain empty iterator for the other), and is up to the algorithm to decide how to read the data.
/// Typical implementors include strict or liberal CSV parsers and byte arrays
/// returned from database drivers, or homogeneous memory arrays carrying name metadata.
/// All the provided methods for random sampling works by manipulating
/// the iterators implemented by the user. All sample-reading methods will required that the length of
/// values is an integer multiple of the length of names. If the user implements columns(.) the rows(.) method
/// is already provided. But the user might decide to just return an empty columns structure and implement rows(.)
/// if the structure is not column-contiguous.
///
/// The trait sample works both for column-ordered data or row-ordered data, which is why you have
/// the freedom to define the types Row and Column. If your data is column-oriented, you can define 
/// column as a cheap reference access to &[f64] and row as a copy to Vec<f64>. If your data is row-
/// oriented, you can do the opposite. If your data is not memory-contiguous, you can define copy for both
/// methods. Some algorithms will work better with different options on how to organize your data. Data is assumed
/// distributed as the likelihood node and iid, or at least conditionally iid given the non-likelihood 
/// (prior and hyperprior) nodes. For performance reasons (for example, you want your user to just feed
/// column-oriented or row-oriented data) you can add trait bounds over rows and columns such as:
/// fn my_estimator(s : Sample<Row=R,Column=C> where R : AsRef<f64> or Col : AsRef<f64>.
pub trait Sample
// where
    // Self::Names : IntoIterator<Item=&'a str>,
    // Self::Row : IntoIterator<Item=&'a f64>,
    // Self::Column : IntoIterator<Item=&'a f64>
    // Self::Names : IntoIterator<Item=&'a str>
{

    // type Name;
    
    // type Row;
    
    // type Continuous;
    
    // Offers column names in an order consistent with Self::row, so the model can query
    // the position of the name and use it to index into Self::row. Might return None
    // if column names do not have a fixed order.
    //fn variables(&'a self) -> Option<Vec<&'a str>>;
    
    // For implementors which need to do operations row-wise (e.g. CSV parsers), it is best
    // to just implement row and leave column as a provided method. For already parsed
    // data structures, it is best to implement column and leave row as a provided method.
    // Implementing both row and column assume a certain order for Self::Name.
    // fn row(&'a self, ix : usize) -> Option<Self::Row>;

    // TODO rename to get_variate()
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a>;

    // fn all_variables() -> Box<dyn Index<&str>, String,

    // TODO add method iter_variates() to iterate over (name, Variate) pairs.

}

impl Sample for HashMap<String, Vec<f64>> 
//where
//    Self : 'a
{

    //type names = Vec<&'a str>;
    
    // type Row = Vec<&'a f64>;
    
    // type Continuous = &'a [f64];
    // type Continuous = impl Iterator<Item=&'a f64>;
    
    // type Names = Vec<&'a str>;
    
    /*fn variables(&self) -> Option<Vec<&'a str>> {
        None
    }

    /// This implementor does not know how to iterate over rows because
    /// the hashmap stores the columns in random order.
    fn row(&'a self, ix : usize) -> Option<Self::Row> {
        None
    }*/

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if let Some(col) = self.get(name) {
            Variable::Real(Box::new(col.iter().map(|val| *val)))
        } else {
            Variable::Missing
        }
    }

}

impl Sample for HashMap<String, Either<Vec<f64>, Vec<bool>>> {

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if let Some(col) = self.get(name) {
            match col {
                Either::Left(real) => Variable::Real(Box::new(real.iter().map(|val| *val))),
                Either::Right(binary) => Variable::Binary(Box::new(binary.iter().map(|val| *val)))
            }
        } else {
            Variable::Missing
        }
    }

}

/*match el.downcast_ref::<&[f64]>()
    Some(f) => Variable::Real(Box::new(f.iter())
    None => match el.downcast_ref::<&[usize]>() {
        Some(f) => Variable::Count(Box::new(f.iter())),
        None => match el.downcast_ref::<&[bool]>() {
            Some(f) => Variable::Count(Box::new(f.iter())),
            None => Variabe::
        }
    }
}*/

/*
If we determine that we always observe via &impl IntoIterator<Item=f64>,
then we have:

pub trait Sample {

    fn real(name : &str) -> Variate::Real;
}

let rows = cli.query("select * from patients;").unwrap();
let n = Normal::likelihood(rows.variate("this").assume_real());
let n = n.observe_missing(rows.variate("this").assume_missing_real());
*/

/*/// Implement Sample for Fn-like closures that reference arbitrary user-defined structures
/// and returns the required iterators. Perhaps we can further require that F just implements
/// Into<Variable> and let the user write standard structure-access iterators. In this implementation,
/// each distribution in the graph "knows" which closure to call to yield its data values. The implementation
/// below might be the basis for a #[derive(Sample)] for custom transparent types with public fields,
/// where the distributions should receive names matching the type names. This derive would write
/// an Into<Box<dyn Sample>> for &[T], by hiding the HashMap behind the Box.
///
/// ```
/// pub struct Coord { x : f64, y : f64 }
/// let coords = [Coord{x : 1.0, y : 2.0}, Coord{x : 2.0, y : 3.0}];
/// let mut sample = HashMap::new();
/// sample.insert("x", || coords.iter().map(|c| c.x) );
/// sample.insert("y", || coords.iter().map(|c| c.y) );
/// distr.fit(&sample);
/// ```
impl<'b, I, F> Sample for HashMap<I, F>
where
    I : AsRef<str> + Hash + Eq,
    F : Fn()->Variable<'b>,
    str : AsRef<I>
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a>
    where
        'b : 'a,
        Self : 'a + 'b
    {
        self.get(name.as_ref()) {
            Some(f) => f(),
            None => Variable::Missing
        }
    }
}*/

impl Sample for bool
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if name.is_empty() {
            Variable::Binary(Box::new(Some(self.clone()).into_iter()))
        } else {
            Variable::Missing
        }
    }
}

impl Sample for &[bool]
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if name.is_empty() {
            Variable::Binary(Box::new(self.iter().cloned()))
        } else {
            Variable::Missing
        }
    }
}

impl Sample for usize
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if name.is_empty() {
            Variable::Count(Box::new(Some(self.clone()).into_iter()))
        } else {
            Variable::Missing
        }
    }
}

impl Sample for &[usize]
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if name.is_empty() {
            Variable::Count(Box::new(self.iter().cloned()))
        } else {
            Variable::Missing
        }
    }
}

impl Sample for &str
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if name.is_empty() {
            Variable::Factor(Box::new(Some(self.clone()).into_iter()))
        } else {
            Variable::Missing
        }
    }
}

impl Sample for &[&str]
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if name.is_empty() {
            Variable::Factor(Box::new(self.iter().cloned()))
        } else {
            Variable::Missing
        }
    }
}

impl Sample for f64
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if name.is_empty() {
            Variable::Real(Box::new(Some(self.clone()).into_iter()))
        } else {
            Variable::Missing
        }
    }
}

impl Sample for &[f64]
{
    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if name.is_empty() {
            Variable::Real(Box::new(self.iter().cloned()))
        } else {
            Variable::Missing
        }
    }
}

/*impl<'a, I,F> Sample for iter::Map<I, F>
where
    I: Iterator<Item=&'a bool>,
    F: FnMut(<I as Iterator>::Item) -> &'a bool
{
    fn variable<'b>(&'b self, name : &'b str) -> Variable<'b>
    where
        'a : 'b
    {
        if name.is_empty() {
            let map = self.clone().map(|b| *b);
            Variable::Binary(Box::new(map))
        } else {
            Variable::Missing
        }
    }
}*/

/* or we do something like:
Normal::with_observer(|| t.field )
Where observer is a closure that takes no arguments but captures a generic structure and output a f64 field (or what the distribution
requires).
*/
/* The observer pattern might benefit from a Fn()->Box<dyn Iterator<Item=f64>> that takes no arguments, but captures an
slice of arbitrary object in its body (so it is a Fn), which returns the iterator required to yield values. If
a distribution has an observer, it does not require a name, and fit(.) does not require a &dyn Sample.
*/

impl Sample for HashMap<String, Either<Vec<f64>, Vec<usize>>> {

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if let Some(col) = self.get(name) {
            match col {
                Either::Left(real) => Variable::Real(Box::new(real.iter().cloned())),
                Either::Right(count) => Variable::Count(Box::new(count.iter().cloned()))
            }
        } else {
            Variable::Missing
        }
    }

}

/// Assign variates to columns of a matrix
impl Sample for (Vec<String>, DMatrix<f64>) {

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        if let Some(ix) = self.0.iter().position(|n| &n[..] == name ) {
            if ix < self.1.ncols() {
                let offset = self.1.nrows()*ix;
                let slice_range = offset..offset+self.1.nrows();
                let col_slice = &self.1.data.as_vec()[slice_range];
                Variable::Real(Box::new(col_slice.iter().map(|val| *val )))
            } else {
                Variable::Missing
            }
        } else {
            Variable::Missing
        }
    }
}

impl Sample for Vec<postgres::Row> {

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        let cols = self[0].columns();
        match cols.iter().position(|col| col.name() == name ) {
            Some(col_ix) => {
                let ty = cols[col_ix].type_();

                // Binary variant
                if *ty == postgres::types::Type::BOOL {
                    return Variable::Binary(Box::new(self.iter()
                        .map(move |row| row.get::<_, bool>(col_ix))
                    ))
                }

                // TODO verify if value is positive at runtime for Count variant.
                if *ty == postgres::types::Type::INT2 {
                    return Variable::Count(Box::new(self.iter()
                        .map(move |row| row.get::<_, i16>(col_ix) as usize )
                    ))
                }

                if *ty == postgres::types::Type::INT4 {
                    return Variable::Count(Box::new(self.iter()
                        .map(move |row| row.get::<_, i32>(col_ix) as usize )
                    ))
                }

                if *ty == postgres::types::Type::INT8 {
                    return Variable::Count(Box::new(self.iter()
                        .map(move |row| row.get::<_, i64>(col_ix) as usize )
                    ))
                }

                // Real variants
                if *ty == postgres::types::Type::FLOAT4 {
                    return Variable::Real(Box::new(self.iter()
                        .map(move |row| row.get::<_, f32>(col_ix) as f64 )
                    ))
                }
                if *ty == postgres::types::Type::FLOAT8 {
                    return Variable::Real(Box::new(self.iter()
                        .map(move |row| row.get::<_, f64>(col_ix) )
                    ))
                }

                // Factor variant
                if *ty == postgres::types::Type::TEXT {
                    return Variable::Factor(Box::new(self.iter()
                        .map(move |row| row.get::<_, &str>(col_ix) )
                    ))
                }

                // For any other type, treat as missing
                Variable::Missing
            },
            None => Variable::Missing
        }
    }

}

impl Sample for serde_json::Value {

    fn variable<'a>(&'a self, name : &'a str) -> Variable<'a> {
        match self {
            serde_json::Value::Object(map) => {
                match map.get(name) {
                    Some(serde_json::Value::Array(arr)) => {
                        match arr.get(0) {
                            Some(val) => match val {
                                serde_json::Value::Bool(_) => {
                                    let data : Vec<bool> = arr.iter()
                                        .filter_map(|obj| obj.as_bool() )
                                        .collect();
                                    if data.len() == arr.len() {
                                        Variable::Binary(Box::new(data.into_iter()))
                                    } else {
                                        Variable::Missing
                                    }
                                },
                                serde_json::Value::Number(n) => {
                                    if n.is_u64() {
                                        let data : Vec<usize> = arr.iter()
                                            .filter_map(|obj| obj.as_u64() )
                                            .map(|obj| obj as usize )
                                            .collect();
                                        if data.len() == arr.len() {
                                            Variable::Count(Box::new(data.into_iter()))
                                        } else {
                                            Variable::Missing
                                        }
                                    } else {
                                        let data : Vec<f64> = arr.iter()
                                            .filter_map(|obj| obj.as_f64() )
                                            .collect();
                                        if data.len() == arr.len() {
                                            Variable::Real(Box::new(data.into_iter()))
                                        } else {
                                            Variable::Missing
                                        }
                                    }
                                },
                                serde_json::Value::String(_) => {
                                    let data : Vec<&'a str> = arr.iter()
                                        .filter_map(|obj| obj.as_str() )
                                        .collect();
                                    if data.len() == arr.len() {
                                        Variable::Factor(Box::new(data.into_iter()))
                                    } else {
                                        Variable::Missing
                                    }
                                },
                                _ => Variable::Missing
                            },
                            None => Variable::Missing
                        }
                    },
                    _ => Variable::Missing
                }
            },
            _ => Variable::Missing
        }
    }

}

/*/// Read variables from a packed string record container, assuming the first record contains
/// variable names. 
impl Sample for &[StringRecord] {

    fn variable<'a>(&'a self, name : &str) -> Variable<'a> {    
        if let Some(ix) = self.iter().position(|rec| rec == Ok(name) ) {
        
        }
        if let Some(col) = self.get(name) {
            Variable::from(col.as_ref())
        } else {
            Variable::Missing
        }
    }
}*/

/*/// If you want to use bayes with your custom type T (for a type-safe alternative to HashMap<S,Ve<f64>>, 
/// you just have to implement this trait, and Vec<T> : Sample will be satisfied automatically. This implementation
/// assumes your type satisfies serde::Serialize (which is used to retrieve the column name information).
/// Only a single serialization is performed in the first data point to yield column names,
/// and all other values are simply read by assuming the type yields the same column positions.
pub trait Observation 
where
    Self : Serialize,
    Self::Length : LengthAtMost32
{
    type Length;
    
    fn observation(&self) -> [f64; I];
}*/

/*/// A runtime-defined sample access pattern, which might be column-oriented, row-oriented, or unordered.
/// If at the Estimator implementation you require that your sample satisfies Sample + Into<Table>,
/// then you algorithm can make use of this enumeration to iterate over your sample in
/// an optimal way (if it can), but still acquire the data successfully if it cannot. For example,
/// If your are using Vec<T> for your custom T : Deserialize, then you can implement the
/// conversion always to the Unstructured variant. If you are using a custom column container, you can implement
/// the conversion to the Column variant; if you are using a custom row container, you can implement the 
/// conversion to the Row variant. Note that using this runtime-defined data access strategy is completely
/// optional: The generic implementation to Estimator::fit<.> only requires a Sample implementation, which
/// just iterates over &'a f64 row-wise or column-wise. By requiring the trait bounds AsRef<[f64]> to either row
/// or column at your sample implementation, you forbid at compile time how the data should be structured.
pub enum Table<'a> {
    ColWise(Vec<&'a [f64]>),
    Row(Vec<&'a [f64]),
    Unstructured
}*/

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

/*
The macro #[derive(Sample)] should map structure field names to the column names.
Containers inside the structure should be named to field_name_$i where i is the
container index (constrained to be of the same type for all implementors). A generic
implementation for Vec<T> should be supplied for all T that implement sample. In this way,
users can re-use their custom structures for the inference problem without worrying
about conversions. This might work if distributions are built with a name that maps
to the field of interest:
let n = Normal::new(5, Some("my_field"), None, None);
let mn = MultiNormal::new(10, Some("my_field_$i"), None, None);
*/

/*/// Samples are types which hold independent (or at least conditionally independent)
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
    /// ignore (if any) are not used in the resampling. TODO consider fastrand::shuffle
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
}*/

/*// Suppose we have a packed structure such as:
#[repr(C)]
pub struct Data {
    a : f64,
    b : f64,
    c : f64
}

// Then &[Data] is consistent with a row-ordered data matrix. We provide the implementation:
// Which we can write a #[derive(Sample)] for any homogeneous, packed structure.
impl AsRef<[f64]> for &[Data] {

    fn as_ref(&self) -> &[f64] {
        unsafe { slice::from_raw_parts(&data[0] as *const _, data.len()) }
    }

}

Casting the above to DMatrixSlice<'_, f64> leaves us with a plain "wide" data matrix, since
matrices are column-ordered, and we can use a single transposition to create a distance
matrix, saving up one data copy.
*/


