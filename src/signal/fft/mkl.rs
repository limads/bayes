use crate::mkl::dfti::*;
use crate::mkl::utils::fetch_ptrs;
use std::os::raw::c_long;
use std::ffi::CStr;
use nalgebra::*;
//use std::mem::MaybeUninit;
use nalgebra::base::storage::Storage;
use nalgebra::base::storage::ContiguousStorage;
use std::fmt::Debug;
use super::super::*;
//use nalgebra::storage::*;

/// A structure which is generic over the second dimension
/// (can either be U1 for unidimensional FFTs or Dynamic for
/// 2D FFTs.
pub struct FFTPlan<N, C>
    where
        //R : Dim,
        C : Dim , // DimName
        N : Scalar + Clone + From<f32> + Copy + Debug,
        Complex<N> : Scalar + Clone,
{
    // Dimensionality of FFT, informed at compile time
    //dims : (usize, usize),
    //out_dims : (usize, usize),

    /* Points to the owned descriptor. Valid for the lifetime of the struct. */
    pub handle : *mut DFTI_DESCRIPTOR,

    pub handle_backward : *mut DFTI_DESCRIPTOR,

    /* The FFTPlanner owns the underlying descriptor opaque C struct */
    // descriptor : DFTI_DESCRIPTOR,
    // type Precision = S,
    // input_dims : Dim<I>,

    pub forward_buffer : Matrix<Complex<N>, Dynamic, C, VecStorage<Complex<N>, Dynamic, C>>,

    backward_buffer : Matrix<N, Dynamic, C, VecStorage<N, Dynamic, C>>,

    //forward_calls : usize,

    //backward_calls : usize
}

pub type FFTPlan1D<N> = FFTPlan<N, U1>;

pub type FFTPlan2D<N> = FFTPlan<N, Dynamic>;

fn check_dfti_status(fail : u32, prefix_if_fail : &str) -> Result<(), String> {
    if fail == DFTI_NO_ERROR {
        return Ok(());
    }
    unsafe {
        let msg_ptr = DftiErrorMessage(fail as c_long);
        match CStr::from_ptr(msg_ptr).to_str() {
            Ok(msg) => {
                return Err( format!("{} \n MKL Message : {}",
                    prefix_if_fail, msg) );
            }
            Err(err) => {
                return Err( format!("{} \n Could not decode MKL message: {}",
                    prefix_if_fail, err) );
            }
        }
    }
}

fn build_descriptor(dims : (usize, usize), backward : bool, double_prec : bool)
    -> Result<*mut DFTI_DESCRIPTOR, String> {

    if dims.0 < 2 || dims.0 % 2 != 0 || dims.1 % 2 != 0 {
        return Err(String::from(
            "First dimension must have non-zero size. \
            All non-zero dimensions should have an even \
            number of elements"));
    }
    let ndims = if dims.1 > 0 {
        2
    } else {
        1
    };

    let dims_arr = if ndims == 1 {
        [dims.0 as c_long, 0 as c_long]
    } else {
        [dims.0 as c_long, dims.1 as c_long]
    };

    unsafe {
        let mut descriptor : DFTI_DESCRIPTOR =
            std::mem::uninitialized();
            //MaybeUninit::uninit();
        let mut handle : *mut DFTI_DESCRIPTOR =
            &mut descriptor as *mut DFTI_DESCRIPTOR;
        let handle_ptr : *mut *mut DFTI_DESCRIPTOR =
            &mut handle as *mut *mut DFTI_DESCRIPTOR;

        let prec_const = if double_prec {
            DFTI_CONFIG_VALUE_DFTI_DOUBLE
        } else {
            DFTI_CONFIG_VALUE_DFTI_SINGLE
        };

        // pass [ulong;2][0] by value for dims = 1/
        // pass pointer to [ulong;2] for dims > 2;
        let fail = if ndims == 1 {
            DftiCreateDescriptor(
                handle_ptr,
                prec_const,
                DFTI_CONFIG_VALUE_DFTI_REAL,
                ndims as c_long,
                dims_arr
            ) as u32
        } else {
            DftiCreateDescriptor(
                handle_ptr,
                prec_const,
                DFTI_CONFIG_VALUE_DFTI_REAL,
                ndims as c_long,
                &dims_arr
            ) as u32
        };

        check_dfti_status(fail as u32, "Could not construct Planner.")?;
        // println!("here");

        let fail = DftiSetValue(handle,
            DFTI_CONFIG_PARAM_DFTI_PLACEMENT,
            DFTI_CONFIG_VALUE_DFTI_NOT_INPLACE);
        check_dfti_status(fail as u32, "Could not set placement.")?;

        // DFTI_PERM_FORMAT only available format for 1D transforms
        // DFTI_CCE_FORMAT only available format for ND > 1 transforms
        // DFTI_REAL_REAL - Store output in two separate real/imaginary components.

        //address of X(k 1 , k 2 , ..., k d ) = the pointer supplied to the compute function + s 0 + k 1 *s 1 + k 2 *s 2 + ...+ k d *s d ,
        // where k is the size of the dimension and s the stride (should be k-1)
        // Default strides are { 0, n d-1 *...*n 2 *n 1 , ..., n 2 *n 1 *1, n 1 *1, 1 }

        // DFTI_COMPLEX_COMPLEX implies
        // Each sequence m in the D-dimensional cube is stored together
        // A[m*distance + stride0 + k1*stride1 + k2*stride2 + ... + kd*strided]

        // For correct access, specify strided slice. Should access each element, col-wise, by
        // incrementing the stride by ncols. For accessing next col (1) should also stride
        // by ncols, but substracting (nrows * 1).
        // So row stride = ncols; col stride = ((-1)*nrows).

        /* Setting this value automatically implies data
        will be in the CCE format. */
        let fail = DftiSetValue(handle,
            DFTI_CONFIG_PARAM_DFTI_CONJUGATE_EVEN_STORAGE,
            DFTI_CONFIG_VALUE_DFTI_COMPLEX_COMPLEX);
        check_dfti_status(fail as u32, "Could not set conjugate-even storage.")?;

        //let out_dims = (dims.0 / 2 + 1, dims.0 / 2 + 1);
        let mut input_strides : [c_long;4] = [0,0,0,0];
        let mut output_strides : [c_long;4] = [0,0,0,0];
        DftiGetValue(
            handle,
            DFTI_CONFIG_PARAM_DFTI_INPUT_STRIDES,
            &input_strides as *const c_long
        );
        DftiGetValue(
            handle,
            DFTI_CONFIG_PARAM_DFTI_OUTPUT_STRIDES,
            &output_strides as *const c_long
        );

        if ndims >= 2 && backward {
            input_strides [1] = input_strides[1] / 2 + 1;
            DftiSetValue(
                handle,
                DFTI_CONFIG_PARAM_DFTI_INPUT_STRIDES,
                &input_strides as *const c_long
            );
        }

        if ndims >= 2 && !backward {
            output_strides [1] = input_strides[1] / 2 + 1;
            DftiSetValue(
                handle,
                DFTI_CONFIG_PARAM_DFTI_OUTPUT_STRIDES,
                &output_strides as *const c_long
            );
        }
        let fail = DftiCommitDescriptor(handle) as u32;
        if fail == DFTI_NO_ERROR {
            Ok(handle)
        } else {
            Err(String::from("Could not commit descriptor"))
        }
    }
}

/// S should implement From<f32> and Copy and Zero because we are going to initialize an
/// output buffer array from a f32 literal. Actually just f32, f64 can
/// be used by the MKL API.
impl<'a, N, C> FFTPlan<N, C>
    where
        //R : Dim,
        C : Dim,
        N : Scalar + Clone + From<f32> + Copy + Debug,
        Complex<N> : Scalar + Clone,
        VecStorage<N,Dynamic,C> : ContiguousStorage<N, Dynamic, C>,
        VecStorage<Complex<N>,Dynamic,C> : ContiguousStorage<Complex<N>, Dynamic, C>,
        VecStorage<N,Dynamic,C> : ContiguousStorage<N, Dynamic, C>,
        VecStorage<Complex<N>,Dynamic,C> : ContiguousStorage<Complex<N>, Dynamic, C>,
{

    pub fn new(sz : (usize, usize)) -> Result<FFTPlan::<N, C>, String> {
        let (forward_buffer, backward_buffer) =
            FFTPlan::<N, C>::initialize_buffers(sz);
        let dims = match C::try_to_usize() {
            Some(_) => (sz.0, 0),
            None => sz
        };
        let double_prec = N::is::<f64>();
        let handle_forward = build_descriptor(dims, false, double_prec)?;
        let handle_backward = build_descriptor(dims, true, double_prec)?;
        Ok( FFTPlan::<N, C> {
            handle : handle_forward,
            handle_backward,
            forward_buffer,
            backward_buffer,
            //forward_calls : 0,
            //backward_calls : 0
        })
    }

    /*type RealMatrix =
        Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>;
    type ComplexMatrix =
        Matrix<Complex<N>,Dynamic,C,VecStorage<Complex<N>, Dynamic, C>>;*/

    /// Used internaly by FFTPlanner constructor.
    /// The out buffer should be initialized as an
    /// array of complex numbers of half the size
    /// the input dims in single or two dimensions.
    /// TODO remove here. Duplicated at FrequencyDomain.
    fn initialize_forward_buffer(
        input_dims : (usize, usize)
    ) -> Matrix<Complex<N>,Dynamic,C,VecStorage<Complex<N>, Dynamic, C>> {
        let out_dims : (usize, usize) =
            (input_dims.0 / 2 + 1, input_dims.1);
        let zero_cplx : Complex<N> = Complex::<N>::new(N::from(0.0), N::from(0.0));
        let cplx_vec : Vec<Complex<N>> = vec![zero_cplx; (out_dims.0)*(out_dims.1)];
        let vs = VecStorage::new(Dim::from_usize(out_dims.0), Dim::from_usize(out_dims.1), cplx_vec);
        Matrix::<Complex<N>,Dynamic,C,VecStorage<Complex<N>,Dynamic,C>>::from_data(vs)
    }

    fn initialize_backward_buffer(
        input_dims : (usize, usize)
    ) -> Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>> {
        let v : Vec<N> = vec![N::from(0.0); input_dims.0 * input_dims.1];
        let vs = VecStorage::new(Dim::from_usize(input_dims.0), Dim::from_usize(input_dims.1), v);
        Matrix::<N,Dynamic,C,VecStorage<N,Dynamic,C>>::from_data(vs)
    }

    fn initialize_buffers(
        input_dims : (usize, usize)
    ) -> (Matrix<Complex<N>,Dynamic,C,VecStorage<Complex<N>, Dynamic, C>>,
          Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>) {
        let forward = Self::initialize_forward_buffer(input_dims);
        let backward = Self::initialize_backward_buffer(input_dims);
        (forward, backward)
    }

    pub fn check_valid_forward_dim(&self, dim : (usize, usize)) -> Result<(), String> {
        // Note that the input dimension to be tested must
        // be compared to the backward buffer
        let buf_dims = (
            self.backward_buffer.nrows(),
            self.backward_buffer.ncols(),
        );
        if !(buf_dims == dim) {
            Err("Invalid dimension at forward pass.".to_string())
        } else {
            Ok(())
        }
    }

    pub fn check_valid_backward_dim(&self, dim : (usize, usize)) -> Result<(), String> {
        // Note that the input dimension to be tested must
        // be compared to the forward buffer
        let buf_dims = (
            self.forward_buffer.nrows(),
            self.forward_buffer.ncols(),
        );
        if !(buf_dims == dim) {
            Err("Invalid dimension at backward pass.".to_string())
        } else {
            Ok(())
        }
    }

    /*pub fn new<I : ?Sized> (dims : Dim<I>, complex_out : bool, double_prec : bool)
            -> Result<FFTPlan<I>, String> {
        if dims.len() > 2 {
            let msg = "Invalid number of dimensions \
                (Must be 1 or 2)".to_string()
            return Err(msg);
        }
    }*/

    pub unsafe fn forward_mut<S, SC>(
        handle : *mut DFTI_DESCRIPTOR,
        src : &Matrix<N,Dynamic,C,S>,
        dst : &mut Matrix<Complex<N>,Dynamic,C,SC>
    ) -> i64
        where
            S : Storage<N, Dynamic, C>,
            SC : Storage<Complex<N>, Dynamic, C>
    {
        let (in_ptr, _, out_ptr, _) =
            fetch_ptrs(&src, dst);
        let status = DftiComputeForward(
            handle, in_ptr, out_ptr);
        status
    }

    pub unsafe fn backward_mut<S, SC>(
        handle_backward : *mut DFTI_DESCRIPTOR,
        src : &Matrix<Complex<N>,Dynamic,C,SC>,
        dst : &mut Matrix<N,Dynamic,C,S>
    ) -> i64
        where
            S : Storage<N, Dynamic, C>,
            SC : Storage<Complex<N>, Dynamic, C>
    {
        let (in_ptr, _, out_ptr, _) =
            fetch_ptrs(&src, dst);
        let status = DftiComputeBackward(
            handle_backward, in_ptr, out_ptr);
        status
    }

    // Mutates the owned frequency domain buffer by transforming the informed matrix, returning a mutable
    // reference to it in the case of a succesful transform, so the user can manipulate it
    // before the next transform can be called. The user is responsible for preserving
    // the dimensionality of the buffer, or the next call using this planner will incurr in
    // error. If there is a requirement for dimensionality changes, call `plan.forward(arr).clone()`
    // and perform any changes on the returned owned value.
    // This pattern of returning references to internal buffers has the drawback that any references
    // to self will be elided, so self cannot be used again in the same scope. If self is required
    // to be used again in the same scope, use the pattern:
    // if let Ok(_) = self.forward() {
    //     compute(&self.forward_ref());
    //     if let Ok(_) = self.backward(&self.forward_ref()) {
    //         compute(&self.backward_ref());
    //     }
    // }

    /*pub fn forward<S>(
        &'a mut self,
        arr : &Matrix<N,Dynamic,C,S>
    ) -> Result<
        &'a mut Matrix<Complex<N>,Dynamic,C,VecStorage<Complex<N>,Dynamic,C>>,
        String >
        where
            S : Storage<N, Dynamic, C>
    {
        self.check_valid_forward_dim((arr.nrows(), arr.ncols(),))?; //.shape
        unsafe {
            /* Note that the Rust Array should have size/2 since each element
            now occupies two floating/double slots. */

            //println!("before comp : {:?}", self.forward_buffer.shape());
            let (in_ptr, in_n, out_ptr, out_n) =
                fetch_ptrs(&arr, &mut self.forward_buffer);

            /*let out_ptr : *mut std::ffi::c_void =
                out.as_mut_ptr() as *mut std::ffi::c_void;
            let arr_ptr : *mut std::ffi::c_void =
                arr.as_mut_ptr() as *mut std::ffi::c_void;*/
            //println!("preparing dfti compute");
            let status = DftiComputeForward(
                self.handle, in_ptr, out_ptr); // HERE

            if status == 0 {
                // let s = self.forward_buffer.slice_mut((0,0,),self.forward_buffer.shape());
                //let refmut : &'a mut Matrix<Complex<N>,Dynamic,C,VecStorage<Complex<N>,Dynamic,C>> = &mut self.forward_buffer;
                //println!("compute successful");
                self.forward_calls += 1;
                return Ok(&mut self.forward_buffer);
            } else {
                check_dfti_status(status as u32, "Error computing DFT.")?;
            }
        }
        Err(String::from("Error computing forward DFT."))
    }*/

    /*/// Mutates the owned original domain buffer, returning a mutable
    /// copy to it in the case of a succesful transform, so the user can manipulate it
    /// before the next transform can be called. The user is responsible for preserving
    /// the dimensionality of the buffer, or the next call using this planner will incurr in
    /// error.
    pub fn backward<S>(
        &'a mut self,
        arr : &Matrix<Complex<N>,Dynamic,C,S>
    ) -> Result<&'a mut Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>, String>
        where
            S : Storage<Complex<N>, Dynamic, C>
    {
        self.check_valid_backward_dim((arr.nrows(), arr.ncols(),))?;

        // let mut out_shape = arr.raw_dim().clone();
        // let mut shape_arr = out_shape.as_array_view_mut();
        // shape_arr.map_mut(|x|{ *x = *x*2 - 1 });
        // let mut out = Array::<A,Dim<I>>::from_elem(out_shape, S::from(0.0));

        unsafe {
            let (in_ptr, in_n, out_ptr, out_n) =
                fetch_ptrs(&arr, &mut self.backward_buffer);
            let status = DftiComputeBackward(
                self.handle_backward, in_ptr, out_ptr);
            if status == 0 {
                self.backward_calls += 1;
                return Ok(&mut self.backward_buffer);
            } else {
                check_dfti_status(status as u32, "Error computing DFT.")?;
            }
        }
        Err(String::from("Error computing backward DFT."))
    }*/

    /*pub fn backward_ref(
        &'a self
    ) -> Result<&'a Matrix<N,R,C,VecStorage<N, R, C>>, &'static str> {
        if self.backward_calls > 0 {
            Ok(&self.backward_buffer)
        } else {
            Err("No backward calls made yet.")
        }
    }*/

    /*pub fn backward_ref_mut(
        &'a mut self
    ) -> Result<&'a mut Matrix<N,Dynamic,C,VecStorage<N, Dynamic, C>>, &'static str> {
        if self.backward_calls > 0 {
            Ok(&mut self.backward_buffer)
        } else {
            Err("No backward calls made yet.")
        }
    }*/

}

impl<N, C> FrequencyBasis<N, Complex<N>, C> for FFTPlan<N, C>
    where
        N : Scalar+ Clone + From<f32> + Copy + Debug,
        C : Dim ,
        N : Scalar + Clone + From<f32> + Copy + Debug,
        Complex<N> : Scalar + Clone,
{

    fn try_forward<'a, S>(
        &'a mut self,
        src : &Matrix<N,Dynamic,C,S>
    ) -> Option<&'a mut Matrix<Complex<N>,Dynamic,C,VecStorage<Complex<N>, Dynamic, C>>>
        where
            S : ContiguousStorage<N, Dynamic, C>
    {
        //println!("src dims: {:?}", src.shape());
        //println!("backward buffer dims: {:?}", self.backward_buffer.shape());
        self.check_valid_forward_dim((src.nrows(), src.ncols(),))
            .map_err(|e| eprintln!("{}", e) ).ok()?;
        let dst = &mut self.forward_buffer;
        unsafe {
            let status = FFTPlan::<N, C>::forward_mut(self.handle, src, dst);
            if status == 0 {
                Some(dst)
            } else {
                eprintln!("Error computing DFT: {}", status);
                if status > 0 {
                    if let Err(e) = check_dfti_status(status as u32, "(MKL message)") {
                        eprintln!("{}", e);
                    }
                }
                None
            }
        }
    }

    fn try_backward_to<'a, 'b, /*S,*/ SB>(
        &'a self,
        //src : &'a Matrix<Complex<N>,Dynamic,C,S>,
        dst : &'b mut Matrix<N,Dynamic,C,SB>
    ) -> Option<&'b mut Matrix<N,Dynamic,C,SB>>
        where
          //  S : ContiguousStorage<Complex<N>, Dynamic, C>,
            SB : ContiguousStorageMut<N, Dynamic, C>
    {
        self.check_valid_backward_dim((self.forward_buffer.nrows(), self.forward_buffer.ncols(),))
            .map_err(|e| eprintln!("{}", e)).ok()?;
        unsafe {
            let status = FFTPlan::<N, C>::backward_mut(self.handle_backward, &self.forward_buffer, dst);
            if status == 0 {
                Some(dst)
            } else {
                eprintln!("Error computing DFT: {}", status);
                if status > 0 {
                    if let Err(e) = check_dfti_status(status as u32, "(MKL message)") {
                        eprintln!("{}", e);
                    }
                }
                None
            }
        }
    }

}

#[cfg(test)]
pub mod test {

    use super::*;
    const EPS : f64 = 1E-8;

    #[test]
    fn fft2d() {
        let mut plan = FFTPlan2D::<f64>::new((8,8)).unwrap();
        let mut impulse : DMatrix<f64> = DMatrix::zeros(8, 8);
        let mut backward : DMatrix<f64> = DMatrix::zeros(8, 8);
        impulse[0] = 1.;
        println!("Impulse: {}", impulse);
        let forward = plan.forward(&impulse);
        println!("Forward transform of impulse: {}", forward);
        plan.backward_to(&mut backward);
        backward.unscale_mut((8.).powf(2.));
        assert!((backward.norm() - impulse.norm()).abs() < EPS);
        println!("Backward transform of impulse: {}", backward);
    }

    #[test]
    fn fft1d() {
        let mut plan = FFTPlan1D::<f64>::new((8,1)).unwrap();
        let mut impulse : DVector<f64> = DVector::zeros(8);
        let mut backward : DVector<f64> = DVector::zeros(8);
        impulse[0] = 1.;
        println!("Impulse: {}", impulse);
        let forward = plan.forward(&impulse);
        println!("Forward transform of impulse: {}", forward);
        plan.backward_to(&mut backward);
        backward.unscale_mut(8.);
        assert!((backward.norm() - impulse.norm()).abs() < EPS);
        println!("Backward transform of impulse: {}", backward);
    }

}
