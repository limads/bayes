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
use std::fmt::{self, Display};
use nalgebra::storage::Storage;
use std::default::Default;

pub mod csv;

// #[cfg(feature = "sql")]
pub mod sql;

pub use sql::*;

#[derive(Clone, Copy)]
pub enum ColumnType {
    Integer,
    Long,
    Double,
    Float,
    Boolean,
    Numeric
}

pub enum NullAction {
    IgnoreRow,
    Error
}

impl Default for NullAction {

    fn default() -> Self {
        NullAction::Error
    }

}

impl Display for ColumnType {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnType::Integer => write!(f, "integer"),
            ColumnType::Long => write!(f, "bigint"),
            ColumnType::Double => write!(f, "double precision"),
            ColumnType::Float => write!(f, "real"),
            ColumnType::Boolean => write!(f, "bool"),
            ColumnType::Numeric => write!(f, "decimal"),
        }
    }

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

impl From<File> for TableSource {

    fn from(f : File) -> Self {
        TableSource::File(f)
    }

}

/// Load data from tabular sources to in-memory double precision matrices.
/// Support csv files or relational databases.
/// Wraps a column-major double precision numeric data matrix,
/// which can be indexed by column name or position.
/// Keeps run-time types associated with each column
/// for insertion into relational databases or generating
/// CSV output. May or may not own a database connection
/// or maintain an open file for conveniently updating
/// its internal state or the remote source state.
pub struct Sample {

    col_names : Vec<String>,

    col_types : Vec<ColumnType>,

    data : DMatrix<f64>,

    _source : TableSource
}

impl Sample {

    pub fn from_reader<R>(mut reader : R, null : NullAction) -> Result<Self, String>
        where R : Read
    {
        let mut content = String::new();
        reader.read_to_string(&mut content).map_err(|e| format!("{}", e) )?;
        let tbl : Sample = Self::load_with_action(&content[..], null)?;
        Ok(tbl)
    }

    fn load_from_file(mut f : File, null : NullAction) -> Result<Self, String> {
        let mut content = String::new();
        f.read_to_string(&mut content).map_err(|e| format!("{}", e) )?;
        let source = TableSource::File(f);
        let mut tbl : Sample = Self::load_with_action(&content[..], null)?;
        tbl._source = source;
        Ok(tbl)
    }

    pub fn open<P>(path : P, null : NullAction) -> Result<Self, String>
        where P : AsRef<Path>
    {
        let f = File::open(path).map_err(|e| format!("{}", e) )?;
        Self::load_from_file(f, null)
    }

    pub fn build<C, S>(
        columns : &[&str],
        data : &Matrix<f64, Dynamic, C, S>
    ) -> Result<Self, &'static str>
        where
            C : Dim,
            S : Storage<f64, Dynamic, C>
    {
        let cols : Vec<DVector<f64>> = data.column_iter().map(|c| c.clone_owned()).collect();
        let packed_data = DMatrix::from_columns(&cols[..]);
        if packed_data.ncols() != columns.len() {
            return Err("Invalid column dimensions");
        }
        let mut col_names = Vec::new();
        col_names.extend(columns.iter().map(|c| c.to_string()));
        let col_types : Vec<_> = (0..packed_data.ncols()).map(|_| ColumnType::Double ).collect();
        Ok(Self {
            col_names,
            col_types,
            _source : TableSource::Unknown,
            data : packed_data
        })
    }

    fn load_with_action(s : &str, action : NullAction) -> Result<Self, String> {
        let (opt_header, data) = csv::load_matrix_from_str(s, action)?;
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

    pub fn save(&mut self) -> Result<(), String> {
        unimplemented!()
    }

    pub fn take_content(self) -> DMatrix<f64> {
        self.data
    }

    /// Generate a sequence of SQL insert statements for the current table and tries to
    /// insert them using the the relational database held by the table.
    /// If the informed name does not exist in the database, and create is true,
    /// creates the table before inserting. Just append the results to the existing
    /// table otherwise.
    pub fn insert(&mut self, _at : &str, _create : bool) -> Result<(), String> {
        unimplemented!()
    }

    /*pub fn apply(src : S, dst : D, f : Fn(&Table)->Option<Table>) -> Result<(), String>
        where
            S : Into<TableSource>,
            D : Into<TableSource>
    {
        let src = src.into();
        let dst = dst.into();
        let ans = match src {
            TableSource::Unknown => return Err("Unknown source"),
            TableSource::Postgres(c) => {
                fn load_postgre(mut client : Client, sql : &str, max : usize)
            },
            TableSource::File(f) => {
                let tbl = Table::load_from_file(f);
                f(&tbl).ok_or(format!("User-function failed"))?;

            }
        }
    }*/

    /// Generates SQL statement to insert all rows of the matrix
    /// into the named table. If the numeric type cannot be coerced
    /// into the run-time type of the column, the method will fail.
    pub fn insert_stmt(&self, name : &str) -> Result<String, String> {
        let mut stmt = String::new();
        stmt += &format!("insert into {} values ", name);
        let ncols = self.data.ncols();
        let nrows = self.data.nrows();
        for (row_ix, row) in self.data.row_iter().enumerate() {
            stmt += "(";
            for (col_ix, (f, t)) in row.iter().zip(self.col_types.iter()).enumerate() {
                match t {
                    ColumnType::Integer => {
                        if f.fract() == 0.0 {
                            stmt += &format!("{}", *f as i32)[..];
                        } else {
                            return Err(format!("Conversion to integer failed"));
                        }
                    },
                    ColumnType::Long => {
                        if f.fract() == 0.0 {
                            stmt += &format!("{}", *f as i64)[..];
                        } else {
                            return Err(format!("Conversion to integer failed"));
                        }
                    },
                    ColumnType::Double | ColumnType::Float | ColumnType::Numeric => {
                        stmt += &format!("{}", f)[..];
                    },
                    ColumnType::Boolean => {
                        if f.fract() == 0.0 {
                            match *f as i32 {
                                0 => stmt += "'f'",
                                1 => stmt += "'t'",
                                _ => return Err(format!("Invalid boolean state"))
                            }
                        } else {
                            return Err(format!("Conversion to boolean failed"));
                        }
                    }
                };
                if col_ix < ncols - 1 {
                    stmt += ",";
                } else {
                    stmt += ")";
                    if row_ix < nrows - 1 {
                        stmt += ",\n";
                    } else {
                        stmt += ";";
                    }
                }
            }
        }
        Ok(stmt)
    }

    /// Generates create table SQL statement, using the
    /// current run-time type information. If data cannot
    /// be coerced from the matrix into any column format, the
    /// method will fail. If populate is false, just creates the
    /// table, without inserting any data.
    pub fn create_stmt(&self, name : &str, populate : bool) -> Result<String, String> {
        let mut stmt = format!("create table {} (", name);
        for (i, (name, tp)) in self.col_names.iter().zip(self.col_types.iter()).enumerate() {
            let name = match name.chars().find(|c| *c == ' ') {
                Some(_) => String::from("\"") + &name[..] + "\"",
                None => name.clone()
            };
            stmt += &format!("{} {}", name, tp);
            if i < self.col_names.len() - 1 {
                stmt += ","
            } else {
                stmt += ");\n"
            }
        }
        if populate {
            stmt += &self.insert_stmt(name)?[..];
        }
        Ok(stmt)
    }

    /// Re-uses the query/file-path to update information,
    /// potentially saving a matrix re-allocation if the data
    /// dimensionality did not increase. Trims data
    /// if it is smaller than the previous call.
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

    /// Casts the column(s) to informed type to change CSV/Sql output.
    /// All data is stored as a continuous f64 buffer irrespective of
    /// the output types.
    pub fn cast_output<I>(&mut self, ix : I, col_type : ColumnType) -> Result<(), &'static str>
        where I : Into<ColumnIndex>
    {
        let ix_range = self.index_range(ix.into()).ok_or("Invalid index")?;
        self.col_types.iter_mut().skip(ix_range.0).take(ix_range.1)
            .for_each(|ct| *ct = col_type);
        Ok(())
    }

    /// Randomly shuffle the rows of the sample in-place.
    pub fn shuffle(&mut self) {
        unimplemented!()
    }

    /// Returns (self.at[0..col), [col..max_col)
    pub fn split_columns(self, col : usize) -> (Sample, Sample) {
        let (nrow, ncol) = self.data.shape();
        let mut cols = self.col_names.clone();
        let mut types = self.col_types.clone();
        let mut data : Vec<f64> = self.data.data.into();
        let s2_data = data.split_off(col*nrow);
        let s2_cols = cols.split_off(col);
        let s2_types = types.split_off(col);
        let m1 = DMatrix::from_vec(nrow, col-1, data);
        let m2 = DMatrix::from_vec(nrow, ncol - col, s2_data);
        let s1 = Sample {
            col_names : cols,
            col_types : types,
            data : m1,
            _source : TableSource::Unknown
        };
        let s2 = Sample {
            col_names : s2_cols,
            col_types : s2_types,
            data : m2,
            _source : TableSource::Unknown
        };
        (s1, s2)
    }

    /// Returns self.rows[0..row), self.rows([row..max_row)
    pub fn split_rows(self, row : usize) -> (Sample, Sample) {
        let (nrow, ncol) = self.data.shape();
        let mut cols = self.col_names.clone();
        let mut types = self.col_types.clone();
        let mut m_data = self.data;
        m_data.transpose_mut();
        let mut data_vec : Vec<f64> = m_data.data.into();
        let s2_vec = data_vec.split_off(ncol*row);
        let mut s1_data = DMatrix::from_vec(ncol, row - 1, data_vec);
        let mut s2_data = DMatrix::from_vec(ncol, nrow - row, s2_vec);
        s1_data.transpose_mut();
        s2_data.transpose_mut();
        let s1 = Sample {
            col_names : cols.clone(),
            col_types : types.clone(),
            data : s1_data,
            _source : TableSource::Unknown
        };
        let s2 = Sample {
            col_names : cols,
            col_types : types,
            data : s2_data,
            _source : TableSource::Unknown
        };
        (s1, s2)
    }

    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.data.shape()
    }

}

impl FromStr for Sample {

    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::load_with_action(s, Default::default())
    }

}

impl Display for Sample {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut header = String::new();
        self.col_names.iter().take(self.col_names.len() - 1)
            .for_each(|name| { header += name; header += "," });
        if let Some(last) = self.col_names.last() {
            header += last;
        }
        let content = csv::build_string_packed(&self.data);
        write!(f, "{}\n{}", header, content)
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
