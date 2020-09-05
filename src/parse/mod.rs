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

/// Enumeration representing any probabilistic model read from the outside world
/// (e.g. JSON file), anchored by its top-level node (its likelihood). AnyLikelihood
/// has methods mle(.) and visit_factors(.) just like the Likelihood implementors; those
/// calls dispatch to the respective implementations internally.
///
/// To get the underlying distribution, you can either match the enum or use the
/// conversion to the dynamic (mutable or plain) references Distribution or Posterior:
///
/// ```
/// let a : &dyn Distribution = (&AnyLikelihood).into();
/// let a : &dyn Posterior = (&AnyLikelihood).into();
/// ```
///
/// Which will allow you to write code that applies to any variant.
#[derive(Clone)]
pub enum AnyLikelihood {
    Other,
    MN(MultiNormal)
}

impl AnyLikelihood {

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
            AnyLikelihood::MN(m) => m.clone().into(),
            AnyLikelihood::Other => unimplemented!()
        };
        let content = serde_json::to_string(&val)?;
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
            AnyLikelihood::MN(ref mut m) => {
                let r = f(m);
                r
            },
            _ => unimplemented!()
        }
    }*/

    /// Dispatches to the MLE of the variant, returning the result
    /// wrapped in the same variant.
    pub fn mle(&self, y : DMatrixSlice<'_, f64>) -> Self {
        match self {
            AnyLikelihood::MN(m) => AnyLikelihood::MN(MultiNormal::mle(y)),
            _ => unimplemented!()
        }
    }

    /// Dispatches to the visit_factors of the respective variant.
    pub fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior) {
        match self {
            AnyLikelihood::MN(m) => m.visit_factors(f),
            _ => unimplemented!()
        }
    }

}

impl<'a> From<&'a mut AnyLikelihood> for &'a mut dyn Distribution {

    fn from(distr : &'a mut AnyLikelihood) -> Self {
        match distr {
            AnyLikelihood::MN(ref mut m) => m as &mut _,
            _ => unimplemented!()
        }
    }

}

impl<'a> From<&'a mut AnyLikelihood> for &'a mut dyn Posterior {

    fn from(distr : &'a mut AnyLikelihood) -> Self {
        match distr {
            AnyLikelihood::MN(ref mut m) => m as &mut _,
            _ => unimplemented!()
        }
    }

}

impl<'a> From<&'a AnyLikelihood> for &'a dyn Posterior {

    fn from(distr : &'a AnyLikelihood) -> Self {
        match distr {
            AnyLikelihood::MN(ref m) => m as &_,
            _ => unimplemented!()
        }
    }

}

impl<'a> From<&'a AnyLikelihood> for &'a dyn Distribution {

    fn from(distr : &'a AnyLikelihood) -> Self {
        match distr {
            AnyLikelihood::MN(ref m) => m as &_,
            _ => unimplemented!()
        }
    }

}

impl fmt::Display for AnyLikelihood {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v : Value = self.clone().into();
        write!(f, "{}", v)
    }
}

impl TryFrom<serde_json::Value> for AnyLikelihood {

    type Error = String;

    fn try_from(mut val : Value) -> Result<Self, String> {
        match val.get("multinormal") {
            Some(v) => Ok(AnyLikelihood::MN(MultiNormal::try_from(v.clone())?)),
            _ => Err(format!("Invalid likelihood node"))
        }
    }

}

impl Into<serde_json::Value> for AnyLikelihood {

    fn into(self) -> serde_json::Value {
        match self {
            AnyLikelihood::MN(m) => m.into(),
            AnyLikelihood::Other => unimplemented!()
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


