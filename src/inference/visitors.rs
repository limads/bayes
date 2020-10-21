use crate::distr::*;
use anyhow::Error;
use std::cell::RefCell;
use nalgebra::*;

/// Iterates the factor graph in a depth-first fashion, copying all
/// natural parameter values into the v vector buffer, if its size allows it.
/// The iteration does not change the factors (although a mutable reference is
/// required to satisfy the Posterior interface).
pub fn collect_parameters<V>(distr : &mut dyn Posterior, v : V) -> Result<V, Error>
where
    V : AsMut<[f64]>
{
    // The interface is a little awkward for now: We cannot use FnMut at the
    // Posterior interface, or else we cannot call the closure indefinitely through
    // the graph, so we stick with Fn and pass references to refcells to the captured
    // variables. We cannot also require the Fn to be of a generic auxiliary data type
    // or else we could not let the Posterior to be a dynamic trait. An alternative would be
    // to make the closure accept the auxiliary data at its parameter and pass it to
    // the child factors by return values and re-capture it (using, for example, Box<dyn Any>
    // as auxiliary data), although that would be too convoluted to do.
    let offset = RefCell::new(0);
    let err : RefCell<Option<Error>> = RefCell::new(None);
    let v = RefCell::new(v);
    let f = |factor : &mut dyn Posterior| {
        if err.borrow().is_some() {
            return;
        }
        let nat_param = factor.view_parameter(true);
        let param_len = nat_param.nrows();
        let off = offset.borrow();
        if let Some(param_slice) = v.borrow_mut().as_mut().get_mut(*off..(*off + param_len)) {
            param_slice.copy_from_slice(nat_param.as_slice());
            *(offset.borrow_mut()) += param_len;
        } else {
            let msg = format!("Invalid parameter offset ({}) at factor {}", off, factor.to_string());
            *(err.borrow_mut()) = Some(Error::msg(msg));
        }
    };
    distr.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
    match err.into_inner() {
        Some(e) => Err(e.into()),
        None => Ok(v.into_inner())
    }
}

/// Calculates the full parameter vector length required for this probabilistic graph
pub fn param_vec_length(distr : &mut dyn Posterior) -> usize {
    let len = RefCell::new(0);
    let f = |factor : &mut dyn Posterior| {
        let nat_param = factor.view_parameter(true);
        let param_len = nat_param.nrows();
        *(len.borrow_mut()) += param_len;
    };
    distr.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
    len.into_inner()
}

/// Iterates the factor graph in a depth-first fashion, copying the natural
/// parameter values from v into the corresponding distributions, if its size
/// allows it.
pub fn update_parameters<V>(distr : &mut dyn Posterior, v : &V) -> Result<(), Error>
where
    V : AsRef<[f64]>
{
    let err : RefCell<Option<Error>> = RefCell::new(None);
    let offset = RefCell::new(0);
    let f = |factor : &mut dyn Posterior| {
        if err.borrow().is_some() {
            return;
        }
        let off = offset.borrow_mut();
        let param_len = factor.view_parameter(true).nrows();
        if let Some(param_vals) = v.as_ref().get(*off..(*off+param_len)) {
            let vs : DVectorSlice<'_, f64> = param_vals.into();
            factor.set_parameter(vs, true);
            *(offset.borrow_mut()) += param_len;
        } else {
            let msg = format!("Invalid parameter offset ({}) at factor {}", off, factor.to_string());
            *(err.borrow_mut()) = Some(Error::msg(msg));
        }
        let nat_param = factor.view_parameter(true);
    };
    distr.visit_post_factors(&f as &dyn Fn(&mut dyn Posterior));
    match err.into_inner() {
        Some(e) => Err(e.into()),
        None => Ok(())
    }
}

