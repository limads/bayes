use super::*;
use rand_distr;
use rand;
use crate::inference::sim::*;
use std::default::Default;
use std::fmt::{self, Display};
use anyhow;
use serde_json::{self, Value, map::Map};
use crate::model::Model;
use std::convert::{TryFrom, TryInto};
use crate::model;
pub type BernoulliFactor = UnivariateFactor<Beta>;
use argmin;
use either::Either;

type Parameter = Either<f64, DVector<f64>>;

fn parse_sample_n<D>(d : &D, val : &Value) -> Result<usize, String>
where
    D : Distribution
{
    if let Some(Value::Number(nv)) = val.get("n") {
        if let Some(n) = nv.as_u64() {
            if let Some(obs) = d.observations() {
                if obs.nrows() != n as usize {
                    return Err(format!(
                        "Entry 'n' has size {} but observation vector has size {}",
                        n,
                        obs.nrows()
                    ));
                }
            }
            Ok(n as usize)
        } else {
            Err(format!("n should be a positive integer"))
        }
    } else {
        Err(format!("Missing sample size (n) entry"))
    }
}

/// If reading from a multinormal failed, and reading from a conjugate factor failed,
/// then the parameter can only be a scalar (left) or vector (right), determined from
/// this function.
fn reset_numeric_parameter<D>(d : &mut D, val : &Value, loc_name : &str) -> Result<(), String>
where
    D : Distribution
{
    let loc_val = val.get(loc_name)
        .ok_or(format!("Missing {} parameter", loc_name))?;
    let res_param = match loc_val.as_f64() {
        Some(loc) => Ok(Parameter::Left(loc)),
        None => match parse::parse_vector(loc_val) {
            Ok(loc_vec) => Ok(Parameter::Right(loc_vec)),
            Err(e) => Err(format!("Invalid entry for parameter {}: {}", loc_name, e))
        }
    };
    match res_param {
        Ok(Parameter::Left(loc_val)) => {
            let n = match d.observations() {
                Some(obs) => obs.nrows(),
                None => parse_sample_n(d, &val)?
            };
            let param = DVector::from_element(n, loc_val);
            d.set_parameter((&param).into(), false);
            Ok(())
        },
        Ok(Parameter::Right(loc_vec)) => {
            if let Some(obs) = d.observations() {
                if loc_vec.nrows() != obs.nrows() {
                    return Err(format!(
                        "Entry observation vector has size {} but parameter vector has size {}",
                        obs.nrows(),
                        loc_vec.nrows()
                    ));
                }
            }
            d.set_parameter((&loc_vec).into(), false);
            Ok(())
        },
        Err(e) => Err(e)
    }
}

/// Reset the parameter of this distribution (the parameter can be either a constant
/// vector or scalar, conjugate factor or multinormal factor). The distribution observations
/// (if any) are already assumed to have been read from val into D.
fn reset_parameter<D, P>(d : &mut D, val : &Value, loc_name : &str) -> Result<(), String>
where
    D : Distribution + Default + Conditional<P> + Conditional<MultiNormal>,
    P : Distribution + TryFrom<serde_json::Value>
{
    let mut d : D = Default::default();
    let loc_val = val.get(loc_name).ok_or(format!("Missing {} parameter", loc_name))?;
    match MultiNormal::try_from(loc_val.clone()) {
        Ok(mn) => {
            let eta = mn.sample();
            let mut d = d.condition(mn);
            d.set_parameter((&eta).into(), true);
            Ok(())
        },
        Err(_) => {
            match P::try_from(loc_val.clone()) {
                Ok(p) => {
                    let theta = p.sample();
                    let mut d = d.condition(p);
                    d.set_parameter((&theta).into(), false);
                    Ok(())
                },
                Err(_) => reset_numeric_parameter(&mut d, &val, loc_name)
            }
        }
    }
}

pub fn parse_univariate<D, P>(val : &Value, loc_name : &str) -> Result<D,String>
where
    D : Distribution + Default + Conditional<P> + Conditional<MultiNormal>,
    P : Distribution + TryFrom<serde_json::Value>
{
    let mut d : D = Default::default();
    if let Some(obs) = val.get("obs") {
        let v = parse::parse_vector(&obs)?;
        let obs = DMatrix::from_columns(&[v]);
        d.set_observations((&obs).into());
    }
    reset_parameter::<D, P>(&mut d, &val, loc_name)?;
    println!("Parsed distribution: {:?}", d);
    Ok(d)
}

pub fn sample_to_string(distr : &str) -> String {
    let model : Model = distr.parse().unwrap();
    let distr : &Distribution = (&model).into();
    let sample = distr.sample();
    let mat_v = crate::model::matrix_to_value(&sample);
    let mat_string = mat_v.to_string();
    mat_string
}

