use nalgebra::*;
use crate::gsl::wavelet::{self, *};
use crate::gsl::wavelet2d::{self};
use nalgebra::storage::*;
use super::iter::*;
use crate::signal::*;

#[derive(Clone)]
pub enum Basis {
    // Centered or not
    Haar(bool),

    // k = 4..20 even; centered or not
    Daubechies(usize, bool),

    // k = 103, 105, 202..208 even, 301..309 odd; centered or not
    BSpline(usize, bool)
}

#[derive(Clone)]
pub struct DWTPlan<C>
    where
        C : Dim
{
    nrows : usize,
    ncols : usize,
    n_elems : usize,
    pub buf : Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>,
    w_ptr : *mut gsl_wavelet,
    ws : *mut gsl_wavelet_workspace,
    basis : Basis
}

impl<C> DWTPlan<C>
    where
        C : Dim //+ DimName
{

    fn init_ws(
        sz : usize,
        k : usize,
        w_type : *const gsl_wavelet_type
    ) -> (*mut gsl_wavelet, *mut gsl_wavelet_workspace) {
        unsafe {
            let w_ptr = gsl_wavelet_alloc(w_type, k);
            let ws = gsl_wavelet_workspace_alloc(sz);
            (w_ptr, ws)
        }
    }

    /// Creates a new DWTPlan.
    /// size should be the vector size or matrix side size, such that the total
    /// number of elements for a 2D transform will be size^2
    pub fn new(basis : Basis, size : usize) -> Result<Self,&'static str> {
         if (size as f32).log2().fract() > 0.0 {
            return Err("Number of rows and columns should be a power of 2");
        }
        let nrows = size;
        let ncols = match C::try_to_usize() {
            Some(sz) => {
                if sz > 1 {
                    return Err("Dimension C should be dynamic or U1");
                }
                1
            },
            None => size
        };
        let n_elems = match ncols {
            1 => size,
            _ => size.pow(2)
        };
        let mut data : Vec<f64> = Vec::with_capacity(n_elems);
        data.extend( (0..n_elems).map(|_| 0.0) );
        let buf = Matrix::from_vec_generic(Dynamic::new(size), C::from_usize(ncols), data);
        unsafe {
            let (w_type, k) = match basis {
                Basis::Haar(c) => {
                    let t = match c {
                        true => gsl_wavelet_haar_centered,
                        false => gsl_wavelet_haar
                    };
                    (t, 2)
                },
                Basis::Daubechies(k, c) => {
                    let t = match c {
                        true => gsl_wavelet_daubechies_centered,
                        false => gsl_wavelet_daubechies
                    };
                    (t, k)
                },
                Basis::BSpline(k, c) => {
                    let t = match c {
                        true => gsl_wavelet_bspline_centered,
                        false => gsl_wavelet_bspline
                    };
                    (t, k)
                }
            };
            let (w_ptr, ws) = Self::init_ws(n_elems, k, w_type);
            Ok(Self{n_elems, nrows, ncols, buf, w_ptr, ws, basis})
        }
    }
}

impl<'a, C> DWTPlan<C>
    where
        C : Dim
{

    fn check_error(ans : i32) -> Result<(), &'static str> {
        match ans {
            0 => Ok(()),
            _ => {
                println!("GSL Error: {}", ans);
                Err("Error calculating wavelet.")
            }
        }
    }

    fn forward_mut<S>(
        src : &Matrix<f64, Dynamic, C, S>,
        dst : &mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>,
        n_elems : usize,
        ws : *mut gsl_wavelet_workspace,
        w_ptr : *mut gsl_wavelet,
        nrows : usize,
        ncols : usize
    ) -> Result<(), &'static str>
    where
        C : Dim, //+ DimName,
        S : Storage<f64, Dynamic, C>,
        VecStorage<f64, Dynamic, C> : ContiguousStorageMut<f64, Dynamic, C>,
    {
        dst.copy_from(&src);
        unsafe {
            match ncols {
                1 => {
                    let ans = wavelet::gsl_wavelet_transform_forward(
                        w_ptr,
                        &mut dst.as_mut_slice()[0] as *mut f64,
                        1,
                        n_elems,
                        ws
                    );
                    Self::check_error(ans)?;
                },
                _ => {
                    let ans = wavelet2d::gsl_wavelet2d_nstransform_forward(
                        w_ptr,
                        &mut dst.as_mut_slice()[0] as *mut f64,
                        ncols,
                        nrows,
                        ncols,
                        ws
                    );
                    Self::check_error(ans)?;
                }
            }
        }
        Ok(())
    }

    fn backward_mut<S, SD>(
        src : &Matrix<f64, Dynamic, C, S>,
        dst : &mut Matrix<f64, Dynamic, C, SD>,
        n_elems : usize,
        ws : *mut gsl_wavelet_workspace,
        w_ptr : *mut gsl_wavelet,
        nrows : usize,
        ncols : usize
    ) -> Result<(), &'static str>
    where
        C : Dim,
        S : Storage<f64, Dynamic, C>,
        SD : ContiguousStorageMut<f64, Dynamic, C>
    {
        dst.copy_from(&src);
        unsafe {
            match ncols {
                1 => {
                    let ans = wavelet::gsl_wavelet_transform_inverse(
                        w_ptr,
                        &mut dst.as_mut_slice()[0] as *mut f64,
                        1,
                        n_elems,
                        ws
                    );
                    Self::check_error(ans)?;
                },
                _ => {
                    let ans = wavelet2d::gsl_wavelet2d_nstransform_inverse(
                        w_ptr,
                        &mut dst.as_mut_slice()[0] as *mut f64,
                        ncols,
                        nrows,
                        ncols,
                        ws
                    );
                    Self::check_error(ans)?;
                }
            }
        }
        Ok(())
    }

    pub fn iter_levels(
        &'a self
    ) -> DWTIteratorBase<&'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>> {
        DWTIteratorBase::<&'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>::new_ref(&self.buf)
    }

    pub fn iter_levels_mut(
        &'a mut self
    ) -> DWTIteratorBase<&'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>> {
        DWTIteratorBase::<&'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>::new_mut(&mut self.buf)
    }

}

pub type DWTPlan2D = DWTPlan<Dynamic>;

pub type DWTPlan1D = DWTPlan<U1>;

impl<C> FrequencyBasis<f64, f64, C> for DWTPlan<C>
    where
        C : Dim
{

    fn try_forward<'a, S>(
        &'a mut self,
        src : &Matrix<f64,Dynamic,C,S>
    ) -> Option<&'a mut Matrix<f64,Dynamic,C,VecStorage<f64, Dynamic, C>>>
        where
            S : ContiguousStorage<f64, Dynamic, C>
    {
        Self::forward_mut(src, &mut self.buf, self.n_elems, self.ws, self.w_ptr, self.nrows, self.ncols)
            .map_err(|e| println!("{}", e) ).ok()?;
        Some(&mut self.buf)
    }

    fn try_backward_to<'a, 'b, /*S,*/ SB>(
        &'a self,
        // src : &'a Matrix<f64,Dynamic,C,S>,
        dst : &'b mut Matrix<f64,Dynamic,C,SB>
    ) -> Option<&'b mut Matrix<f64,Dynamic,C,SB>>
        where
            //S : ContiguousStorage<f64, Dynamic, C>,
            SB : ContiguousStorageMut<f64, Dynamic, C>
    {
        Self::backward_mut(&self.buf, dst, self.n_elems, self.ws, self.w_ptr, self.nrows, self.ncols)
            .map_err(|e| println!("{}", e) ).ok()?;
        Some(dst)
    }

}

#[cfg(test)]
pub mod test {

    use super::*;
    const EPS : f64 = 1E-8;

    #[test]
    fn dwt2d() {
        let mut plan = DWTPlan2D::new(Basis::Daubechies(6, true), 8).unwrap();
        let mut impulse : DMatrix<f64> = DMatrix::zeros(8, 8);
        let mut backward : DMatrix<f64> = DMatrix::zeros(8, 8);
        impulse[0] = 1.;
        println!("Impulse: {}", impulse);
        let forward = plan.forward(&impulse);
        println!("DWT transform of impulse: {}", forward);
        plan.backward_to(&mut backward);
        assert!((backward.norm() - impulse.norm()).abs() < EPS);
        println!("DWT inverse transform of impulse: {}", backward);
    }

    #[test]
    fn dwt1d() {
        let mut plan = DWTPlan1D::new(Basis::Daubechies(6, true), 8).unwrap();
        let mut impulse : DVector<f64> = DVector::zeros(8);
        let mut backward : DVector<f64> = DVector::zeros(8);
        impulse[0] = 1.;

        println!("Impulse: {}", impulse);
        let forward = plan.forward(&impulse);
        println!("DWT transform of impulse: {}", forward);
        plan.backward_to(&mut backward);
        assert!((backward.norm() - impulse.norm()).abs() < EPS);
        println!("DWT inverse transform of impulse: {}", backward);
    }

}
