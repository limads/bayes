use nalgebra::*;
use nalgebra::storage::*;
use std::os::raw::c_int;
use crate::mkl::{self, vsl::*};
use std::fmt::Debug;
use std::mem;

pub enum KernelPtr {
    Single(*mut f32),
    Double(*mut f64)
}

pub struct ConvTask<N, D1, D2>
    where
        D1 : Dim,
        D2 : Dim,
        N : Scalar + Clone + From<f32> + Copy + Debug,
{
    // Kernel is associated with task at constructor and then not used anymore;
    // Keep it here to guarantee lifetime.
    _kernel : Matrix<N, D1, D2, VecStorage<N,D1,D2>>,
    out_buffer : Matrix<N, D1, D2, VecStorage<N,D1,D2>>,
    task_ptr : VSLConvTaskPtr,
    data_stride : [c_int; 2],
    out_stride : [c_int; 2]
}

unsafe fn create_conv_task(
    f_nrows : usize,
    f_ncols : usize,
    k_nrows : usize,
    k_ncols : Option<usize>,
    kernel_ptr : KernelPtr,
    r_decimation : usize,
    c_decimation : usize
) -> Result<VSLConvTaskPtr, &'static str> {
    let xshape : [c_int; 2] = [k_nrows as c_int, k_ncols.unwrap_or(1) as c_int];
    let yshape : [c_int; 2] = [f_nrows as c_int, f_ncols as c_int];
    let zshape : [c_int; 2] = [f_nrows as c_int, f_ncols as c_int];
    let xshape_ptr : *const c_int = &xshape[0];
    let yshape_ptr : *const c_int = &yshape[0];
    let zshape_ptr : *const c_int = &zshape[0];
    let dims : c_int = if f_ncols > 1 {
        2
    } else {
        1
    };
    let x_stride : [c_int; 2] = [1, k_ncols.unwrap_or(1) as c_int];
    let x_stride_ptr : *const c_int = &x_stride[0];
    let mut task_ptr : VSLConvTaskPtr = mem::uninitialized();
    let status = match kernel_ptr {
        KernelPtr::Double(k_ptr) => {
            vsldConvNewTaskX(
                &mut task_ptr as *mut VSLConvTaskPtr,
                VSL_CONV_MODE_AUTO as i32,
                dims ,
                xshape_ptr ,
                yshape_ptr ,
                zshape_ptr,
                k_ptr,
                x_stride_ptr
            )
        },
        KernelPtr::Single(k_ptr) => {
            vslsConvNewTaskX(
                &mut task_ptr as *mut VSLConvTaskPtr,
                VSL_CONV_MODE_AUTO as i32,
                dims ,
                xshape_ptr ,
                yshape_ptr ,
                zshape_ptr,
                k_ptr,
                x_stride_ptr
            )
        }
    };
    if status != VSL_STATUS_OK as i32 {
        println!("Error creating convolution task: {}", status);
        return Err("Non-zero status at task creation. Aborting.");
    }
    match kernel_ptr {
        KernelPtr::Double(_) => {
            vslConvSetInternalPrecision(
                task_ptr,
                VSL_CONV_PRECISION_DOUBLE as i32
            );
        },
        KernelPtr::Single(_) => {
            vslConvSetInternalPrecision(
                task_ptr,
                VSL_CONV_PRECISION_SINGLE as i32
            );
        }
    };
    let decimation : [c_int; 2] = [
        r_decimation as c_int,
        c_decimation as c_int
    ];
    let status = vslConvSetDecimation(
        task_ptr,
        &decimation[0] as *const _
    );
    if status != VSL_STATUS_OK as i32 {
        println!("Error setting convolution task precision: {}", status);
        return Err("Non-zero status at task creation. Aborting.");
    }
    let start : [c_int; 2] = [0, 0];
    let status = vslConvSetStart(
        task_ptr,
        &start[0] as *const _
    );
    if status != VSL_STATUS_OK as i32 {
        println!("Error setting convolution task start: {}", status);
        return Err("Non-zero status at task creation. Aborting.");
    }
    Ok(task_ptr)
}

impl<'a, N, D1, D2> ConvTask<N, D1, D2>
    where
        D1 : Dim,
        D2 : Dim, //+ DimName,
        N : Scalar + Clone + From<f32> + Copy + Debug, //,
        VecStorage<N,D1,D2> : ContiguousStorage<N, D1, D2>
        //VecStorage<N,Dynamic,D2> : ContiguousStorage<N, Dynamic, D1>,
{

    pub fn convolve<S>(
        &'a mut self,
        data : &Matrix<N,D1,D2,S>
    ) -> Result<&'a mut Matrix<N,D1,D2,VecStorage<N, D1, D2>>, &'static str>
        where S : ContiguousStorage<N,D1,D2>
    {
        unsafe {
            let data_ptr = mkl::utils::fetch_ptr(data).0;
            let result_ptr = mkl::utils::fetch_ptr(&self.out_buffer).0;
            let status = vslsConvExecX(
                self.task_ptr,
                data_ptr as *mut f32,
                &self.data_stride[0] as *const i32,
                result_ptr as *mut f32,
                &self.out_stride[0] as *const i32
            );
            if status != VSL_STATUS_OK as i32 {
                println!("Convolution error: {}", status);
                Err("Error performing convolution. Aborting.")
            } else {
                Ok(&mut self.out_buffer)
            }
        }
    }

}

// Return input and output strides
pub fn get_strides(
    _nrows : usize,
    ncols : usize
) -> [c_int; 2] {
    match ncols {
        1 => {
            [1, 1 as c_int]
        }
        _ => {
            [1 as c_int, ncols as c_int]
        }
    }
}

impl ConvTask<f32, Dynamic, U1> {

    pub fn new(
        kernel : DVector<f32>,
        nrows : usize
    ) -> Result<ConvTask<f32, Dynamic, U1>, &'static str> {
        unsafe {
            let kernel_ptr = mkl::utils::fetch_ptr(&kernel).0 as *mut f32;
            let task_ptr = create_conv_task(nrows, 1, kernel.nrows(), None, KernelPtr::Single(kernel_ptr), 1, 1)
                .map_err(|_|{ "Could not create descriptor" })?;
            Ok(ConvTask{
                _kernel : kernel,
                out_buffer : DVector::<f32>::from_element(nrows, 0.0),
                task_ptr : task_ptr,
                data_stride : get_strides(nrows, 1),
                out_stride : get_strides(nrows, 1)
            })
        }
    }

}

impl ConvTask<f64, Dynamic, U1> {

    pub fn new(
        kernel : DVector<f64>,
        nrows : usize
    ) -> Result<ConvTask<f64, Dynamic, U1>, &'static str> {
        unsafe {
            let kernel_ptr = mkl::utils::fetch_ptr(&kernel).0 as *mut f64;
            let task_ptr = create_conv_task(nrows, 1, kernel.nrows(), None, KernelPtr::Double(kernel_ptr), 1, 1)
                .map_err(|_|{ "Could not create descriptor" })?;
            Ok(ConvTask{
                _kernel : kernel,
                out_buffer : DVector::<f64>::from_element(nrows, 0.0),
                task_ptr : task_ptr,
                data_stride : get_strides(nrows, 1),
                out_stride : get_strides(nrows, 1)
            })
        }
    }

}

impl ConvTask<f32, Dynamic, Dynamic>  {

    pub fn new(
        kernel : DMatrix<f32>,
        nrows : usize,
        ncols : usize
    ) -> Result<ConvTask<f32, Dynamic, Dynamic>, &'static str> {
        unsafe {
            let kernel_ptr = mkl::utils::fetch_ptr(&kernel).0 as *mut f32;
            let task_ptr = create_conv_task(nrows, ncols, kernel.nrows(), Some(kernel.ncols()), KernelPtr::Single(kernel_ptr), 1, 1)
                .map_err(|_| "Could not create descriptor")?;
            Ok(ConvTask{
                _kernel : kernel,
                out_buffer : DMatrix::<f32>::from_element(nrows, ncols, 0.0),
                task_ptr : task_ptr,
                data_stride : get_strides(nrows, ncols),
                out_stride : get_strides(nrows, ncols)
            })
        }
    }
}

impl ConvTask<f64, Dynamic, Dynamic>  {

    pub fn new(
        kernel : DMatrix<f64>,
        nrows : usize,
        ncols : usize
    ) -> Result<ConvTask<f64, Dynamic, Dynamic>, &'static str> {
        unsafe {
            let kernel_ptr = mkl::utils::fetch_ptr(&kernel).0 as *mut f64;
            let task_ptr = create_conv_task(nrows, ncols, kernel.nrows(), Some(kernel.ncols()), KernelPtr::Double(kernel_ptr), 1, 1)
                .map_err(|_| "Could not create descriptor")?;
            Ok(ConvTask{
                _kernel : kernel,
                out_buffer : DMatrix::<f64>::from_element(nrows, ncols, 0.0),
                task_ptr : task_ptr,
                data_stride : get_strides(nrows, ncols),
                out_stride : get_strides(nrows, ncols)
            })
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::mkl::*;
    use super::mkl::*;
    use super::*;
    use super::iter::*;

    #[test]
    fn base_conv() -> Result<(), ()> {
        let mut a = DMatrix::<f64>::from_element(7, 7, 0.0);

        // The impulse just reproduce the kernel with a relative shift
        // given by the impulse position. Can just use .imax() to get the
        // peak position.
        *a.get_mut((2,0)).unwrap() = 1.0;

        let mut k = DMatrix::<f64>::from_element(3,3,1.0 / 9.0);
        let mut task = ConvTask::<f64, Dynamic, Dynamic>::new(
            k.clone(),
            a.nrows(),
            a.ncols()
        ).unwrap();
        println!("{:.8}", task.convolve(&a).unwrap());
        Ok(())
    }
}

