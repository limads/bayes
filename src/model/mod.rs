use nalgebra::*;
use serde_json::{self, Value};
use std::error::Error;
use crate::prob::*;
// use crate::prob::multinormal::*;
use std::convert::{TryFrom, TryInto};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::fmt::{self, Display};
use std::convert::{AsRef, AsMut};
use std::path::Path;
use std::str::FromStr;
use crate::sample::Sample;

pub mod parse;

/// Supports the derivation of optimized decision rules based on comparison
/// of posterior log-probabilities (work in progress). TODO move under bayes::model,
/// which will concentrate model comparison routines.
pub mod decision;

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
    
    // pub fn posterior(&)

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

    /*/// Se documentation for apply(.)
    pub fn apply_mut<F>(&mut self, f : F) -> Result<(), Box<dyn Error>>
    where
        F : Fn(&mut impl Likelihood)->Result<(), Box<dyn Error>>
    {
        match &mut self {
            Model::MN(m) => f(m),
            Model::Bern(b) => f(b),
            _ => unimplemented!()
        }
    }
    
    /// Since Likelihood cannot be made into a trait object (preventing us to implement AsRef<dyn Likelihood>
    /// or something similar) we resort to matching the Model enum to access the Likelihood implementors
    /// (all variants are implement it). The apply/apply_mut receive a generic closure that will
    /// be called on the current likelihood node, so the user does not have to 
    /// resolve the match by hand to execute a functionality that would apply to different variants.
    pub fn apply<F,R>(&self, f : F) -> Result<R, Box<dyn Error>>
    where
        F : Fn(&impl Likelihood) -> Result<R, Box<dyn Error>>
    {
        match &self {
            Model::MN(m) => f(m),
            Model::Bern(b) => f(b),
            _ => unimplemented!()
        }
    }*/
    
    // Dispatches to the MLE of the variant, returning the result
    // wrapped in the same variant.
    /*pub fn mle(&self, y : DMatrixSlice<'_, f64>) -> Result<Self, anyhow::Error> {
        match self {
            Model::MN(m) => Ok(Model::MN(MultiNormal::mle(y)?)),
            _ => unimplemented!()
        }
    }*/

    // Dispatches to the visit_factors of the respective variant.
    /*pub fn visit_factors<F>(&mut self, f : F) where F : Fn(&mut dyn Posterior) {
        match self {
            Model::MN(m) => m.visit_factors(f),
            Model::Bern(b) => b.visit_factors(f),
            _ => unimplemented!()
        }
    }*/
    
    /*pub fn factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>) {
        match self {
            Model::MN(m) => m.factors_mut(),
            Model::Bern(b) => b.factors_mut(),
            _ => unimplemented!()
        }
    }*/

}

/*/// Model implements Distribution by dispatching the calls to its variant
/// TODO Make this AsRef<dyn Distribution>
impl Distribution for Model
    where Self : Sized
{

    fn set_parameter(&mut self, mu : DVectorSlice<'_, f64>, natural : bool) {
        match self {
            Model::MN(m) => m.set_parameter(mu, natural),
            Model::Bern(b) => b.set_parameter(mu, natural),
            _ => unimplemented!()
        }
    }

    fn set_natural<'a>(&'a mut self, eta : &'a mut dyn Iterator<Item=&'a f64>) {
        unimplemented!()
    }
    
    fn view_parameter(&self, natural : bool) -> &DVector<f64> {
        match self {
            Model::MN(m) => m.view_parameter(natural),
            Model::Bern(b) => b.view_parameter(natural),
            _ => unimplemented!()
        }
    }

    fn mean<'a>(&'a self) -> &'a DVector<f64> {
        match self {
            Model::MN(m) => m.mean(),
            Model::Bern(b) => b.mean(),
            _ => unimplemented!()
        }
    }

    fn mode(&self) -> DVector<f64> {
        match self {
            Model::MN(m) => m.mode(),
            Model::Bern(b) => b.mode(),
            _ => unimplemented!()
        }
    }

    fn var(&self) -> DVector<f64> {
        match self {
            Model::MN(m) => m.var(),
            Model::Bern(b) => b.var(),
            _ => unimplemented!()
        }
    }

    fn cov(&self) -> Option<DMatrix<f64>> {
        match self {
            Model::MN(m) => m.cov(),
            Model::Bern(b) => b.cov(),
            _ => unimplemented!()
        }
    }

    fn cov_inv(&self) -> Option<DMatrix<f64>> {
        match self {
            Model::MN(m) => m.cov_inv(),
            Model::Bern(b) => b.cov_inv(),
            _ => unimplemented!()
        }
    }

    fn joint_log_prob(&self, /*y : DMatrixSlice<f64>, x : Option<DMatrixSlice<f64>>*/ ) -> Option<f64> {
        match self {
            Model::MN(m) => m.joint_log_prob(),
            Model::Bern(b) => b.joint_log_prob(),
            _ => unimplemented!()
        }
    }

    fn sample_into(&self, mut dst : DMatrixSliceMut<'_, f64>) {
        match self {
            Model::MN(m) => m.sample_into(dst),
            Model::Bern(b) => b.sample_into(dst),
            _ => unimplemented!()
        }
    }

}*/

/*/// Model implements Likelihood by dispatching the calls to its variant
impl Likelihood for Model {

    fn view_variables(&self) -> Option<Vec<String>> {
        match self {
            Model::MN(m) => m.view_variables(),
            Model::Bern(b) => b.view_variables(),
            _ => unimplemented!()
        }
    }
    
    fn factors_mut<'a>(&'a mut self) -> (Option<&'a mut dyn Posterior>, Option<&'a mut dyn Posterior>) {
        match self {
            Model::MN(m) => m.factors_mut(),
            Model::Bern(b) => b.factors_mut(),
            _ => unimplemented!()
        }
    }
    
    fn with_variables(&mut self, vars : &[&str]) -> &mut Self {
        match self {
            Model::MN(m) => { m.with_variables(vars); self }
            Model::Bern(b) => { b.with_variables(vars); self }
            _ => unimplemented!()
        }
    }
    
    fn observe<'a>(&'a mut self, sample : &dyn Sample) -> &'a mut Self {
        /*match self {
            Model::MN(m) => { m.observe(sample) },
            Model::Bern(b) => { b.observe(sample) },
            _ => unimplemented!()
        }*/
        unimplemented!()
    }
    
}*/

impl<'a> From<&'a mut Model> for &'a mut dyn Distribution {

    fn from(distr : &'a mut Model) -> Self {
        match distr {
            Model::MN(ref mut m) => m as &mut _,
            Model::Bern(ref mut b) => b as &mut _,
            _ => unimplemented!()
        }
    }

}

/*impl<'a> TryFrom<&'a mut Model> for &'a mut dyn Posterior {

    type Error = ();

    fn try_from(distr : &'a mut Model) -> Result<Self, ()> {
        match distr {
            Model::MN(ref mut m) => Ok(m as &mut _),
            Model::Bern(ref mut b) => Err(()),
            _ => unimplemented!()
        }
    }

}*/

/*impl<'a> TryFrom<&'a Model> for &'a dyn Posterior {

    type Error = ();

    fn try_from(distr : &'a Model) -> Result<Self, ()> {
        match distr {
            Model::MN(ref m) => Ok(m as &_),
            Model::Bern(ref b) => Err(()),
            _ => unimplemented!()
        }
    }

}*/

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

/*// Likelihood cannot be made into a object because it resuts &mut self at variables and has
// the compare method.
impl<'a> AsRef<dyn Likelihood + 'a> for Model {
    fn as_ref(&self) -> &(dyn Likelihood + 'a) {
        match &self {
            Model::MN(m) => m as &dyn Likelihood,
            Model::Bern(b) => b as &dyn Likelihood,
            Model::Other => unimplemented!()
        }
    }
}

impl<'a> AsMut<dyn Likelihood + 'a> for Model {
    fn as_mut(&mut self) -> &mut (dyn Likelihood + 'a) {
        match self {
            Model::MN(m) => m as &mut dyn Likelihood,
            Model::Bern(b) => b as &mut dyn Likelihood,
            Model::Other => unimplemented!()
        }
    }
}*/

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


