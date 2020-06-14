use nalgebra::*;
use simba::scalar::RealField;
use std::convert::TryInto;
use std::sync::{Arc, Mutex};
use nalgebra::storage::*;
use super::freq::dwt::gsl::*;
use std::marker::PhantomData;
use crate::basis::freq::FrequencyBasis;
use std::cmp::PartialEq;
use std::fmt::Debug;
use nalgebra::storage::*;
use super::*;
use crate::sample::*;

#[cfg(feature="mkl")]
use super::freq::fft::mkl::FFTPlan;

enum Method<N>
    where N : Scalar + From<f32> + Copy + Debug
{
    // N just serves as phantom data if user did not compile with MKL feature.
    None(N),

    #[cfg(feature="mkl")]
    FFT(FFTPlan<N, U1>),

    DWT(DWTPlan<U1>)
}

impl<N> Method<N>
    where N : Scalar + From<f32> + Copy + Debug
{

    #[cfg(feature = "mkl")]
    fn is_fft(&self) -> bool {
        match &self {
            Method::FFT(_) => true,
            _ => false
        }
    }

    fn is_dwt(&self) -> bool {
        match &self {
            Method::DWT(_) => true,
            _ => false
        }
    }

}

/*impl<N> PartialEq for Method<N> {

    fn eq(&self, other: &Rhs) -> bool {
        match (&self, &other) {
            (Method::None(_), Method::None(_)) => true,

            #[cfg(feature="mkl")]
            (Method::FFT(_), Method::FFT(_)) => true,

            (Method::DWT(_), Method::FFT(_)) => true,
            _ => false
        }
    }

}*/

/// A structure that converts segments of an one-dimensional
/// 8-bit buffer into a double-precision vector, and can be
/// decomposed with a frequnecy-domain method (fft or dwt).
pub struct Sequence<N>
    where
        N : Scalar + RealField + From<f32> + Copy + Debug
{

    //source : &'a [u8],

    //offset : usize,

    size : usize,

    cvt : Option<DVector<N>>,

    method : Method<N>

}

impl<N> Sequence<N>
    where
        N : Scalar + RealField + From<f32> + Copy + Debug
{

    pub fn new_empty(size : usize) -> Self {
        let cvt = DVector::zeros(size);
        Self{ cvt : Some(cvt), /*offset : 0,*/ size, method : Method::<N>::None(N::from(0.)) }
    }

    //pub fn update(&mut self, source : &'a [u8]) {
    //    self.source = source;
    //}

    //pub fn reposition(&mut self, offset : usize, size : usize) {
    //    self.offset = offset;
    //    self.size = size;
    //}

    #[cfg(feature = "mkl")]
    pub fn fft(&mut self) -> DVectorSliceMut<'_, Complex<N>> {
        let cvt = self.cvt.take().unwrap();
        if !self.method.is_fft() {
            self.method = Method::FFT(FFTPlan::new((cvt.nrows(), 1)).unwrap());
        }
        if let Method::FFT(ref mut plan) = self.method {
            let ans = plan.forward(&cvt);
            let ans_slice = ans.rows_mut(0, ans.nrows());
            self.cvt = Some(cvt);
            ans_slice
        } else {
            panic!()
        }
    }

    #[cfg(feature = "mkl")]
    pub fn ifft<S>(&self, dst : &mut Matrix<N, Dynamic, U1, S>)
        where
            S : ContiguousStorageMut<N, Dynamic, U1>
    {
        let n = self.cvt.as_ref().unwrap().nrows();
        if !self.method.is_fft() {
            //self.method = Method::FFT(FFTPlan::new((n, 1)).unwrap());
            panic!("Self method should be fft");
        }

        if let Method::FFT(ref plan) = self.method {
            plan.backward_to(dst);
        } else {
            panic!()
        }
    }

    pub fn copy_from_slice(&mut self, data : &[N]) {
        self.cvt.iter_mut().for_each(|cvt| cvt.copy_from_slice(data) );
    }

    pub fn slice(&self, offset : usize, size : usize) -> DVectorSlice<'_, N> {
        self.cvt.as_ref().unwrap().rows(offset, size)
    }

    pub fn slice_mut(&mut self, offset : usize, size : usize) -> DVectorSliceMut<'_, N> {
        self.cvt.as_mut().unwrap().rows_mut(offset, size)
    }

}

impl<N> From<Vec<N>> for Sequence<N>
        where N : Scalar + From<f32> + Copy + Debug + RealField
{

    fn from(data : Vec<N>) -> Self {
        let dv = DVector::from_vec(data);
        dv.into()
    }

}

impl<N> From<DVector<N>> for Sequence<N>
    where N : Scalar + From<f32> + Copy + Debug + RealField
{

    fn from(data : DVector<N>) -> Self {
        Self {
            size : data.nrows(),
            cvt : Some(data),
            method : Method::None(N::from(0.))
        }
    }

}

impl From<Sample> for Vec<Sequence<f64>> {

    fn from(sample : Sample) -> Vec<Sequence<f64>> {
        unimplemented!()
    }

}

impl Sequence<f32> {

    pub fn copy_from_raw(&mut self, data : &[u8], enc : Encoding) {
        if enc == Encoding::U8 {
            convert_f32_slice(
                &data,
                &mut self.cvt.as_mut().unwrap().data.as_mut_slice()
            );
        } else {
            unimplemented!()
        }
    }

}

impl Sequence<f64> {

    pub fn copy_from_raw(&mut self, data : &[u8], enc : Encoding) {
        if enc == Encoding::U8 {
            convert_f64_slice(
                &data /*[self.offset..(self.offset+self.size)]*/ ,
                &mut self.cvt.as_mut().unwrap().data.as_mut_slice() /*[self.offset..(self.offset+self.size)]*/
            );
        } else {
            unimplemented!()
        }
    }

    pub fn dwt(&mut self) -> impl Iterator<Item=DVectorSlice<'_, f64>> {
        let cvt = self.cvt.take().unwrap();
        if !self.method.is_dwt() {
            self.method = Method::DWT(DWTPlan1D::new(Basis::Daubechies(6, true), cvt.nrows()).unwrap());
        }
        if let Method::DWT(ref mut plan) = self.method {
            let ans = plan.forward(&cvt);
            let levels = plan.iter_levels();
            self.cvt = Some(cvt);
            levels
        } else {
            panic!()
        }
    }

    pub fn idwt<S>(&self, dst : &mut Matrix<f64, Dynamic, U1, S>)
        where
            S : ContiguousStorageMut<f64, Dynamic, U1>
    {
        let n = self.cvt.as_ref().unwrap().nrows();
        if !self.method.is_dwt() {
            //self.method = Method::FFT(FFTPlan::new((n, 1)).unwrap());
            panic!("Self method should be fft");
        }
        if let Method::DWT(ref plan) = self.method {
            plan.backward_to(dst);
        } else {
            panic!()
        }
    }

}

// cargo test --all-features -- sequence_fft --nocapture
#[cfg(feature = "mkl")]
#[test]
fn sequence_decomposition() {
    let mut seq = Sequence::<f64>::new_empty(16);
    println!("{}", seq.fft());
    for s in seq.dwt() {
        println!("{}", s);
    }
}

/*
pub fn decode(&self, dec : &[u8]) -> Result<DMatrix<f32>, &'static str> {
        let mut content = Vec::<f32>::new();
        for w in dec.windows(4) {
            let buffer : Result<[u8; 4], _> = w.try_into();
            if let Ok(b) = buffer {
                let u = u32::from_ne_bytes(b);
                content.push(f32::from_bits(u))
            } else {
                return Err("Could not parse buffer as array");
            }
        }
        Ok(DMatrix::from_vec(self.data.nrows(), self.data.ncols(), content))
    }
*/


