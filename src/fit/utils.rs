use crate::prob::*;
use anyhow::Error;
use std::cell::RefCell;
use nalgebra::*;

/// Iterates the factor graph in a depth-first fashion, copying all
/// natural parameter values into the v vector buffer, if its size allows it.
/// The iteration does not change the factors (although a mutable reference is
/// required to satisfy the Posterior interface).
pub fn collect_parameters(distr : &mut dyn Posterior, output : &mut [f64]) -> Result<(), Error> {
    let nat_param = distr.view_parameter(true);
    let param_len = nat_param.nrows();
    if param_len > output.len() {
        return Err(Error::msg("Invalid parameter split position"));
    }
    let (param_slice, rem) = output.split_at_mut(param_len);
    param_slice.copy_from_slice(nat_param.as_slice());
    let (opt_f1, opt_f2) = distr.dyn_factors_mut();
    let (has_f1, has_f2) = (opt_f1.is_some(), opt_f2.is_some());
    if let Some(f1) = opt_f1 {
        collect_parameters(f1, rem)?;
    } 
    if let Some(f2) = opt_f2 {
        collect_parameters(f2, rem)?;
    }
    if !has_f1 && !has_f2 {
        let rem = output.len() - param_len;
        if rem == 0 {
            Ok(())
        } else {
            let msg = format!("Invalid parameter vector length (remaining: {})", rem);
            Err(Error::msg(msg))
        }
    } else {
        Ok(())
    }
}

/// Calculates the full parameter vector length required for this probabilistic graph.
/// Start with param_vec_len(post, 0);
pub fn param_vec_length(distr : &mut dyn Posterior, mut len : usize) -> usize {
    len += distr.view_parameter(true).nrows();
    let (opt_f1, opt_f2) = distr.dyn_factors_mut();
    if let Some(f1) = opt_f1 {
        len += param_vec_length(f1, len);
    } 
    if let Some(f2) = opt_f2 {
        len += param_vec_length(f2, len);
    }
    len
}

/// Iterates the factor graph in a depth-first fashion, copying the natural
/// parameter values from v into the corresponding distributions, if its size
/// allows it.
pub fn update_parameters(distr : &mut dyn Posterior, values : &[f64]) -> Result<(), Error> {
    let param_len = distr.view_parameter(true).nrows();
    let (param_slice, rem) = values.split_at(param_len);
    let vs : DVectorSlice<'_, f64> = DVectorSlice::from(param_slice);
    distr.set_parameter(vs, true);
    let (opt_f1, opt_f2) = distr.dyn_factors_mut();
    let (has_f1, has_f2) = (opt_f1.is_some(), opt_f2.is_some());
    if let Some(f1) = opt_f1 {
        update_parameters(f1, rem)?;
    } 
    if let Some(f2) = opt_f2 {
        update_parameters(f2, rem)?;
    }
    if !has_f1 && !has_f2 {
        let rem = values.len() - param_len;
        if rem == 0 {
            Ok(())
        } else {
            let msg = format!("Invalid parameter vector length (remaining: {})", rem);
            Err(Error::msg(msg))
        }
    } else {
        Ok(())
    }
}

/// Assuming samples carry the result of natural parameter iterations,
/// distribute the samples over RandomWalk structures maintained by
/// each posterior node.
pub fn set_rw(distr : &mut dyn Posterior, samples : &DMatrix<f64>) -> Result<(), Error> {
    unimplemented!()
}
