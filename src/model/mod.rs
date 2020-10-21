use nalgebra::{DVector, DMatrix, RowDVector, Dim, DMatrixSlice};
use serde_json::{self, Value};
use std::error::Error;
use crate::distr::*;
use crate::distr::multinormal::*;
use std::convert::{TryFrom, TryInto};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::fmt::{self, Display};
use std::convert::{AsRef, AsMut};
use std::path::Path;
use std::str::FromStr;

pub mod parse;

/// Enumeration representing any probabilistic model read from the outside world
/// (e.g. JSON file), anchored by its top-level node (its likelihood). Model
/// has methods mle(.) and visit_factors(.) just like the Likelihood implementors; those
/// calls dispatch to the respective implementations internally.
///
/// To get the underlying distribution, you can either match the enum or use the
/// conversion to the dynamic (mutable or plain) references Distribution or Posterior:
///
/// ```
/// let a : &dyn Distribution = (&Model).into();
/// let a : &dyn Posterior = (&Model).into();
/// ```
///
/// Which will allow you to write code that applies to any variant.
#[derive(Clone, Debug)]
pub enum Model {
    Other,
    MN(MultiNormal),
    Bern(Bernoulli)
}

// TODO implement AsRef<dyn Likelihood> for the model

impl Model {

    pub fn load_from_path<P>(path : P) -> Result<Self, Box<dyn Error>>
    where
        P : AsRef<Path>
    {
        let mut f = File::open(path)?;
        Self::load(f)
    }

    pub fn load<R>(mut reader : R) -> Result<Self, Box<dyn Error>>
    where
        R : Read
    {
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        let val : Value = serde_json::from_str(&content[..])?;
        let lik = val.try_into()?;
        Ok(lik)
    }

    pub fn save_to_path<P>(&self, path : P) -> Result<(), Box<dyn Error>>
    where
        P : AsRef<Path>
    {
        let file = OpenOptions::new().write(true).create(true).open(path)?;
        self.save(file)
    }

    pub fn save<W>(&self, mut writer : W) -> Result<(), Box<dyn Error>>
        where
            W : Write
    {
        let val : Value = match self {
            Model::MN(m) => m.clone().into(),
            Model::Bern(b) => b.clone().into(),
            Model::Other => unimplemented!()
        };
        let content = serde_json::to_string_pretty(&val)?;
        writer.write_all(content.as_bytes())?;
        Ok(())
    }

    /*pub fn try_get<L, C>(self) -> Option<L>
    where
        L : Likelihood<C>,
        C : Dim
    {
        match self {

        }
    }*/

    /*fn apply<F, L, C, R>(&mut self, f : F) -> Result<R, Box<dyn Error>>
    where
        F : Fn(&mut L)->Result<R, Box<dyn Error>>,
        L : Likelihood<C>,
        C : Dim
    {
        match self {
            Model::MN(ref mut m) => {
                let r = f(m);
                r
            },
            _ => unimplemented!()
        }
    }*/

    /// Dispatches to the MLE of the variant, returning the result
    /// wrapped in the same variant.
    pub fn mle(&self, y : DMatrixSlice<'_, f64>) -> Result<Self, anyhow::Error> {
        match self {
            Model::MN(m) => Ok(Model::MN(MultiNormal::mle(y)?)),
            _ => unimplemented!()
        }
    }

    /// Dispatches to the visit_factors of the respective variant.
    pub fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior) {
        match self {
            Model::MN(m) => m.visit_factors(f),
            Model::Bern(b) => b.visit_factors(f),
            _ => unimplemented!()
        }
    }

}

impl<'a> From<&'a mut Model> for &'a mut dyn Distribution {

    fn from(distr : &'a mut Model) -> Self {
        match distr {
            Model::MN(ref mut m) => m as &mut _,
            Model::Bern(ref mut b) => b as &mut _,
            _ => unimplemented!()
        }
    }

}

impl<'a> TryFrom<&'a mut Model> for &'a mut dyn Posterior {

    type Error = ();

    fn try_from(distr : &'a mut Model) -> Result<Self, ()> {
        match distr {
            Model::MN(ref mut m) => Ok(m as &mut _),
            Model::Bern(ref mut b) => Err(()),
            _ => unimplemented!()
        }
    }

}

impl<'a> TryFrom<&'a Model> for &'a dyn Posterior {

    type Error = ();

    fn try_from(distr : &'a Model) -> Result<Self, ()> {
        match distr {
            Model::MN(ref m) => Ok(m as &_),
            Model::Bern(ref b) => Err(()),
            _ => unimplemented!()
        }
    }

}

impl<'a> From<&'a Model> for &'a dyn Distribution {

    fn from(distr : &'a Model) -> Self {
        match distr {
            Model::MN(ref m) => m as &_,
            Model::Bern(ref b) => b as &_,
            _ => unimplemented!()
        }
    }

}

impl From<Bernoulli> for Model {

    fn from(bern : Bernoulli) -> Self {
        Self::Bern(bern)
    }
}

impl From<MultiNormal> for Model {

    fn from(mn : MultiNormal) -> Self {
        Self::MN(mn)
    }
}

impl FromStr for Model {

    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v: Value = serde_json::from_str(s)
            .map_err(|e| format!("{}", e) )?;
        let m = Self::try_from(v)?;
        Ok(m)
    }

}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v : Value = self.clone().into();
        write!(f, "{}", v)
    }
}

impl TryFrom<serde_json::Value> for Model {

    type Error = String;

    fn try_from(mut val : Value) -> Result<Self, String> {
        match MultiNormal::try_from(val.clone()) {
            Ok(mn) => Ok(Model::MN(mn)),
            _ => match Bernoulli::try_from(val) {
                Ok(bern) => Ok(Model::Bern(bern)),
                _ => Err(format!("Invalid likelihood node"))
            }
        }
    }

}

impl Into<serde_json::Value> for Model {

    fn into(self) -> serde_json::Value {
        match self {
            Model::MN(m) => m.into(),
            Model::Bern(b) => b.into(),
            Model::Other => unimplemented!()
        }
    }

}

impl Into<Box<dyn Distribution>> for Model {

    fn into(self) -> Box<dyn Distribution> {
        match self {
            Model::MN(m) => Box::new(m),
            Model::Bern(b) => Box::new(b),
            Model::Other => unimplemented!()
        }
    }
}

pub fn vector_to_value(dvec : &DVector<f64>) -> Value {
    let v : Vec<f64> = dvec.clone().data.into();
    v.into()
}

pub fn matrix_to_value(dmat : &DMatrix<f64>) -> Value {
    let mut rows = Vec::new();
    for r in dmat.row_iter() {
        let mut row : Vec<f64> = r.iter().cloned().collect();
        rows.push(row);
    }
    rows.into()
}

pub fn parse_vector(val : &Value) -> Result<DVector<f64>, String> {
    match val {
        Value::Array(arr_v) => {
            let mut vec = Vec::new();
            for (i, v) in arr_v.iter().enumerate() {
                match v {
                    Value::Number(n) => {
                        let el = n.as_f64().ok_or(String::from("Unable to parse number as f64"))?;
                        vec.push(el);
                    },
                    _ => return Err(format!("{}-th entry in mean vector is not a numeric value", i))
                }
            }
            Ok(DVector::from_vec(vec))
        },
        _ => return Err(format!("Entry should be a numeric array"))
    }
}

pub fn parse_matrix(val : &Value) -> Result<DMatrix<f64>, String> {
    match val {
        Value::Array(val_rows) => {
            let mut mat_rows = Vec::new();
            for (i, r) in val_rows.iter().enumerate() {
                match r {
                    Value::Array(cv) => {
                        let mut row = Vec::new();
                        for (j, v) in cv.iter().enumerate() {
                            match v {
                                Value::Number(n) => {
                                    let el = n.as_f64().ok_or(String::from("Unable to parse number as f64"))?;
                                    row.push(el);
                                },
                                _ => return Err(format!("{}-th x {}-th entry in the matrix vector is not a numeric value", i, j))
                            }
                        }
                        mat_rows.push(RowDVector::from_vec(row));
                    },
                    _ => return Err(format!("{}-th row of matrix is not an array", i))
                }
            }
            let row_len = mat_rows.iter().next()
                .map(|r| r.len())
                .ok_or(String::from("Matrix does not have a first row"))?;
            for (i, r) in mat_rows.iter().enumerate() {
                if r.len() != row_len {
                    return Err(format!("{}th row has invalid length", i));
                }
            }
            Ok(DMatrix::from_rows(&mat_rows[..]))
        },
        _ => Err(format!("Entry is not a nested numeric array"))
    }
}


