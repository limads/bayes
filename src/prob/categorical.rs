use core::borrow::Borrow;
use std::convert::{TryFrom, Into};
use std::fmt;
use std::error::Error;
use std::cmp::Eq;
use std::collections::HashMap;

/// Wraps an u8 that is known to be in the interval [0, N]
pub struct Bounded<const N : usize>(u8);

#[derive(Debug)]
pub struct OutsideBounds;

impl fmt::Display for OutsideBounds {

    fn fmt(&self, f : &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.write_str("OutsideBounds")
    }

}

impl Error for OutsideBounds { }

impl<const N : usize> TryFrom<u8> for Bounded<N> {

    type Error = OutsideBounds;

    fn try_from(v : u8) -> Result<Self, OutsideBounds> {
        if usize::from(v) <= N {
            Ok(Self(v))
        } else {
            Err(OutsideBounds{})
        }
    }
}

impl<const N : usize> Into<u8> for Bounded<N> {

    fn into(self) -> u8 {
        self.0
    }

}

pub struct FactorMap<E>
where
    E : Eq + Clone
{
    factors : HashMap<u8, E>,
    indices : HashMap<E, u8>,
    bounds : u8
}

impl<E> FactorMap<E>
where
    E : Eq + Clone
{

    /// Produce a closure that maps an iterator over comparable elements to a set
    /// of factors. Fails when the number of distinct items is greater than 256
    /// or number of classes found at runtime is bigger than K.
    /// The index of the factor maps to the order they appear.
    ///
    /// ```rust
    /// let items = ["orange", "apple", "orange", "orange"];
    /// let cat : Categorical<2> = Categorical::likelihood(
    ///    FactorMap::try_new(items.iter()).unwrap().indices()
    /// );
    /// ```
    fn try_new<const K : u8>(iter : impl Iterator<Item=E>) -> Option<FactorMap<E>>
    where
        E : Eq + Clone + std::hash::Hash
    {
        let mut factors = HashMap::new();
        let mut indices = HashMap::new();
        let mut curr_ix : u8 = 0;
        for item in iter {
            if indices.get(&item).is_none() {
                if curr_ix < 255 && curr_ix <= K {
                    factors.insert(curr_ix, item.clone());
                    indices.insert(item.clone(), curr_ix);
                    curr_ix += 1;
                } else {
                    return None;
                }
            }
        }
        Some(FactorMap { factors, indices, bounds : K })
    }

    /*/// Should fail when ix > K
    pub fn factor<const K : usize>(&self, ix : Bounded<K>) -> &E {
        self.factors[ix.into::<u8>()]
    }

    /// Should fail if instance of e was not present in the original sample.
    pub fn index<const K : usize>(&self, e : &E) -> Bounded<K> {
        Bounded::try_from(self.indices[e]).unwrap()
    }

    pub fn indices<const K :usize>(&self, iter : impl Iterator<Item=E>) -> impl Iterator<Item=Bounded<K>> {
        iter.map(|item| self.index(&item) )
    }

    pub fn factors<const K : usize>(&self, iter : impl Iterator<Item=Bounded<K>>) -> impl Iterator<Item=&E> {
        iter.map(|ix| self.factor(&ix) )
    }*/
}

#[derive(Debug)]
pub struct Categorical<const N : usize> {

}

impl<const N : usize> Categorical<N> {

}

