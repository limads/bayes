use nalgebra::*;
use nalgebra::storage::*;
use std::default::Default;
use std::ops::{Add, Mul};

/// Convolution utilities
pub mod conv;

/// Utilities for interpolating time series and surfaces, offered by GSL. (Work in progress)
pub mod interp;

mod sampling_utils;

use sampling_utils::*;

/// Signals are types which holds dynamically-allocated
/// data which is read by re-sampling a source slice (raw data
/// comming from a sensor or a network service, for example).
/// The implementor should know how to re-sample this slice
/// by using its own stride information and the user-supplied
/// sampling configuration. The implementor might read the step and decide
/// at runtime to do a scalar copy/conversion in case of size
/// n>1; or do a vectorized copy/conversion in case of size n=1.
/// Specialized vectorized (simd) calls can be written for all
/// possible (N,M) pairs.
///
/// Signals have a source S from which data is downsampled
/// and into which data might be upsampled. N is the scalar type
/// at the (downsampled) destination. downsampling/upsampling with step 1
/// is can use a simple vectorized conversion.
///
/// In contrast to Sample implementors, signals do not interface directly
/// with probabilistic graphs. Since they have a natural ordering, they cannot
/// have their samples interchanged and keep the same meaning. Samples are
/// compressed into sufficient statistic as they are distributed over the probabilistic graph,
/// signals do not. To transform a signal into a sample, some feature extraction algorithm
/// must be applied first, to extract a scalar or vector-valued quantity
/// which can be modelled probabilistically (such as Fourier/Wavelet/PCA/LDA
/// coefficients or signal comparison metrics. Such feature-extraction algorithms
/// can be described as part of a probabilistic model by changing the definition of the
/// conditional expectation from scale: { n : N } to scale : feature : { algorithm : A : { params.. } }
/// All parameters that characterize a feature extraction algorithm must be used at their
/// construction, so users need to concern only with algorithm textual description.
///
/// Although signal samples cannot be interchanged, signals can meaningfully be
/// circularly-shifted, since certain feature extraction algorithms are invariant
/// to circular shifts.
pub trait Signal<M>
    //where
    //    M : Into<N> + Copy + Scalar,
    //    N : Scalar + Copy
{

    fn downsample(&mut self, src : &[M], sampling : Sampling);

    fn upsample(&self, src : &mut [M], sampling : Sampling);

    fn circ_shift(by : (i32, i32));

}

#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub offset : (usize, usize),
    pub size : (usize, usize),
    pub step : usize,
}

/*impl Default for Sampling {
    fn default() -> Self {
    }
}*/

impl<N> Signal<u8> for DVector<N>
where
    N : Scalar + Copy + From<u8> + From<f32> + Add<Output=N> + Mul<Output=N>
    //u8 : From<N>
{

    fn downsample(&mut self, src : &[u8], sampling : Sampling) {
        assert!(sampling.offset.1 == 0);
        if sampling.step == 1 {
            convert_slice(
                &src[sampling.offset.0..],
                self.data.as_mut_slice()
            );
        } else {
            let ncols = self.ncols();
            assert!(src.len() / sampling.step == self.nrows(), "Dimension mismatch");
            subsample_convert(src, self.data.as_mut_slice(), ncols, sampling.step, false);
        }
    }

    fn upsample(&self, dst : &mut [u8], sampling : Sampling) {
        unimplemented!()
    }

    fn circ_shift(by : (i32, i32)) {
        unimplemented!()
    }

}

impl<N> Signal<u8> for DMatrix<N>
where
    N : Scalar + Copy + From<u8> + /*Into<u8> +*/ From<f32> + Add<Output=N> + Mul<Output=N>,
    f64 : From<N>
{

    fn downsample(&mut self, src : &[u8], sampling : Sampling) {
        let (nrows, ncols) = self.shape();
        let rows : Vec<&[u8]> = src.chunks(sampling.size.1)
            .skip(sampling.offset.0)
            .step_by(sampling.step)
            .take(self.nrows())
            .map(|r| &r[sampling.offset.1..(sampling.offset.1+self.ncols()*sampling.step)] )
            .collect();
        //println!("{:?}", rows);
        copy_from_slices(self, &rows, sampling.step);
    }

    fn upsample(&self, dst : &mut [u8], sampling : Sampling) {
        if sampling.step == 1 {
            write_matrix_to_slice(
                &self,
                dst,
                sampling.size,
                sampling.offset,
                None,
                None
            );
        } else {
            unimplemented!()
        }
    }

    fn circ_shift(by : (i32, i32)) {
        unimplemented!()
    }

}

#[cfg(test)]
mod test {

    use nalgebra::*;
    use super::*;

    #[test]
    fn image_subsample() {
        let source : [u8; 16] = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0];
        let mut img : DMatrix<f32> = DMatrix::zeros(4, 4);
        img.downsample(&source[..], Sampling{ offset : (0, 0), size : (4, 4), step : 1});
        println!("img1 = {}", img);
        let mut img2 : DMatrix<f32> = DMatrix::zeros(2, 2);
        img2.downsample(&source[..], Sampling{ offset : (0, 0), size : (4, 4), step : 2});
        println!("img2 = {}", img2);
        let mut img3 : DMatrix<f32> = DMatrix::zeros(3, 3);
        img3.downsample(&source[..], Sampling{ offset : (1, 1), size : (4, 4), step : 1});
        println!("img3 = {}", img3);
        let mut img4 : DMatrix<f32> = DMatrix::zeros(3, 3);
        img4.downsample(&source[..], Sampling{ offset : (0, 0), size : (4, 4), step : 1});
        println!("img4 = {}", img4);
    }

    #[test]
    fn image_upsample() {
        let img : DMatrix<f32> = DMatrix::from_fn(4, 4, |i, j| (i*4 + j) as f32);
        let mut dst : [u8; 16] = [0; 16];
        img.upsample(&mut dst, Sampling{ offset : (0, 0), size : (4, 4), step : 1});
        println!("{:?}", dst);
    }

}
