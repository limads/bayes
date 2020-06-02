use postgres::{self, /*row::Row,*/ Client, /*types::FromSql, types::ToSql*/ };
// use std::ops::Index;
// use std::ops::Range;
use std::fs::{File};
use std::path::Path;
use std::convert::AsRef;
// use sqlparser::dialect::PostgreSqlDialect;
use sqlparser::parser::Parser;
use sqlparser::ast::Statement;
use nalgebra::*;
// use rust_decimal::Decimal;
use std::str::FromStr;
use std::io::Read;

pub mod csv;

// #[cfg(feature = "sql")]
pub mod sql;

//use sql::*;

#[derive(Clone, Copy)]
pub enum ColumnType {
    Integer,
    Long,
    Double,
    Float,
    Boolean,
    Numeric
}

pub enum ColumnIndex {
    Named(String),
    Pos(usize),
    Range(usize,usize),
    NamedRange(String,String)
}

impl From<usize> for ColumnIndex {

    fn from(ix : usize) -> Self {
        Self::Pos(ix)
    }
}

impl From<&str> for ColumnIndex {

    fn from(name : &str) -> Self {
        Self::Named(name.to_owned())
    }
}

impl From<(usize, usize)> for ColumnIndex {

    fn from(ixs : (usize, usize)) -> Self {
        Self::Range(ixs.0, ixs.1)
    }
}

impl From<(&str,&str)> for ColumnIndex {

    fn from(names : (&str, &str)) -> Self {
        Self::NamedRange(names.0.to_owned(), names.1.to_owned())
    }

}

pub enum TableSource {
    Unknown,
    File(File),
    Postgre(Client),
}

impl From<Client> for TableSource {

    fn from(cli : Client) -> Self {
        TableSource::Postgre(cli)
    }

}

/// Wraps a column-major double precision numeric data matrix,
/// which can be indexed by column name or position.
/// Keeps run-time types associated with each column
/// for insertion into relational databases or generating
/// CSV output. May or may not own a database connection
/// or maintain an open file for conveniently updating
/// its internal state or the remote source state.
pub struct Table {

    col_names : Vec<String>,

    col_types : Vec<ColumnType>,

    data : DMatrix<f64>,

    _source : TableSource
}

impl Table {

    pub fn open<P>(path : P) -> Result<Self, String>
        where P : AsRef<Path>
    {
        let mut f = File::open(path).map_err(|e| format!("{}", e) )?;
        let mut content = String::new();
        f.read_to_string(&mut content).map_err(|e| format!("{}", e) )?;
        let source = TableSource::File(f);
        let mut tbl : Table = content.parse().map_err(|e| format!("{}", e) )?;
        tbl._source = source;
        Ok(tbl)
    }

    pub fn save(&mut self) -> Result<(), String> {
        unimplemented!()
    }

    /// Generate a sequence of SQL insert statements for the current table and tries to
    /// insert them using the the relational database held by the table.
    /// If the informed name does not exist in the database, and create is true,
    /// creates the table before inserting. Just append the results to the existing
    /// table otherwise.
    pub fn insert(&mut self, _at : &str, _create : bool) -> Result<(), String> {
        unimplemented!()
    }

    // Re-uses the data buffer and query to update information.
    fn _update(&mut self) {
        unimplemented!()
    }

    fn index_pos(&self, ix : ColumnIndex) -> (Option<usize>, Option<usize>) {
        let search_name = |name : &str| -> Option<usize> { self.col_names.iter().position(|n| &n[..] == name) };
        match ix {
            ColumnIndex::Named(ref a) => (search_name(&a[..]), None),
            ColumnIndex::Pos(ix) => (Some(ix), None),
            ColumnIndex::Range(ix_a, ix_b) => (Some(ix_a), Some(ix_b)),
            ColumnIndex::NamedRange(a, b) => (search_name(&a[..]), search_name(&b[..]))
        }
    }

    fn index_range(&self, ix : ColumnIndex) -> Option<(usize, usize)> {
        let (opt_ix_a, opt_ix_b) = self.index_pos(ix.into());
        let ix_a = opt_ix_a?;
        let ix_len = opt_ix_b.map(|ix| ix - ix_a).unwrap_or(1);
        Some((ix_a, ix_len))
    }

    // Run-time checked column or column range access.
    pub fn at<I>(&self, ix : I) -> Option<DMatrixSlice<'_, f64>>
        where I : Into<ColumnIndex>
    {
        let ix_range = self.index_range(ix.into())?;
        Some(self.data.slice((0,ix_range.0), (self.data.nrows(), ix_range.1)))
    }

    pub fn cast<I>(&mut self, ix : I, col_type : ColumnType) -> Result<(), &'static str>
        where I : Into<ColumnIndex>
    {
        let ix_range = self.index_range(ix.into()).ok_or("Invalid index")?;
        self.col_types.iter_mut().skip(ix_range.0).take(ix_range.1)
            .for_each(|ct| *ct = col_type);
        Ok(())
    }

}

impl FromStr for Table {

    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (opt_header, data) = csv::load_matrix_from_str(s)?;
        let col_names = opt_header.ok_or(format!("Unable to parse header"))?;
        let col_types : Vec<ColumnType> =
            (0..col_names.len()).map(|_| ColumnType::Double ).collect();
        Ok(Self {
            _source : TableSource::Unknown,
            col_types,
            col_names,
            data
        })
    }

}

/*impl<I> Index<I> for Table
    where
        I : Into<ColumnIndex>,
        Self : 'a
{

    type Output = DMatrixSlice<'a,f64>;

    fn index(&'a self, ix: I) -> &'a Self::Output {
        self.at(ix).unwrap()
    }
}*/

