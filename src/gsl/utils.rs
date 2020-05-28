use nalgebra::*;
use nalgebra::storage::Storage;
use crate::gsl::{block_double::gsl_block, vector_double::gsl_vector, matrix_double::gsl_matrix};
use std::slice;
use std::ptr;
use std::mem;
use std::convert::*;

impl From<DVector<f64>> for gsl_vector {
    fn from(v : DVector<f64>) -> Self {
        let mut block = gsl_block{
            size : v.nrows(),
            data : v.data.ptr() as *mut f64
        };
        let vec = gsl_vector {
            size : v.nrows(),
            stride : 1,
            data : v.data.ptr() as *mut f64,
            block : &mut block as *mut gsl_block,
            owner : 1
        };
        mem::forget(v);
        vec
    }
}

impl From<DMatrix<f64>> for gsl_matrix {

    fn from(m : DMatrix<f64>) -> Self {
        let m_t = m.transpose();
        let mut block = gsl_block{
            size : m.nrows() * m.ncols(),
            data : m_t.data.ptr() as *mut f64
        };
        let mat = gsl_matrix {
            size1 : m.nrows(),
            size2 : m.ncols(),
            tda : m.ncols(), // nrows
            data : block.data,
            block : &mut block as *mut gsl_block,
            owner : 1
        };
        mem::forget(m_t);
        mat
    }

}

impl Into<DVector<f64>> for gsl_vector {

    fn into(mut self) -> DVector<f64> {
        let buf_sz = self.size;
        self.owner = 0;
        unsafe {
            let buf = slice::from_raw_parts(self.data, buf_sz);
            DVector::<f64>::from_column_slice(buf)
        }
    }
}

impl Into<DMatrix<f64>> for gsl_matrix {

    fn into(mut self) -> DMatrix<f64> {
        let nrows = self.size1 as usize;
        let ncols = self.size2 as usize;
        let buf_sz = nrows*ncols;
        self.owner = 0;
        unsafe {
            let buf = slice::from_raw_parts(self.data as *const f64, buf_sz);
            DMatrix::<f64>::from_row_slice(nrows, ncols, buf)
        }
    }
}

pub trait SliceView<'a> {

    // Returns non-owning gsl_vector from nalgebra structure (similar to DVectorSlice)
    fn as_gsl_slice(&self) -> gsl_vector;

    // Returns DVectorSlice from an owning gsl_vector
    unsafe fn slice_from_gsl(v : &'a gsl_vector) -> Self;

}

impl<'a> SliceView<'a> for DVectorSlice<'a, f64> {

    fn as_gsl_slice(&self) -> gsl_vector {
        let mut block = gsl_block{
            size : self.nrows(),
            data : self.data.ptr() as *mut f64
        };
        gsl_vector {
            size : self.nrows(),
            stride : 1,
            data : self.data.ptr() as *mut f64,
            block : &mut block as *mut gsl_block,
            owner : 0
        }
    }

    unsafe fn slice_from_gsl(v : &'a gsl_vector) -> Self {
        let s = SliceStorage::from_raw_parts(
            v.data,
            (Dim::from_usize(v.size as usize), Dim::from_usize(1)),
            (Dim::from_usize(1), Dim::from_usize(1))
        );
        DVectorSlice::from_data(s)
    }

}


/*impl AsRef<DVectorSlice<f64>> for gsl_vector {

    fn as_ref(&self) -> &DVectorSlice<f64> {

    }
}

impl AsRef<gsl_vector> for DVector<f64> {

    fn as_ref(&self) -> &gsl_vector {

    }
}*/


#[derive(Debug)]
pub enum GSLStatus {
    Success,
    Failure,
    Continue,
    EDom,
    ERange,
    EFault,
    EInval,
    EFactor,
    ESanity,
    EnoMem,
    EBadFunc,
    ERunaway,
    EMaxIter,
    EZeroDiv,
    EBadTol,
    ETol,
    EUndrflw,
    EOvrflw,
    ELoss,
    ERound,
    EBadLen,
    ENotSqr,
    ESing,
    EDiverge,
    EUnsup,
    EUnimp,
    ECache,
    ETable,
    EnoProg,
    EnoProgJ,
    ETolF,
    ETolX,
    ETolG
}

impl GSLStatus {
    pub fn from_code(code : i32) -> Self {
        match code {
            0 => GSLStatus::Success,
            -1 => GSLStatus::Failure,
            -2 => GSLStatus::Continue,
            1 =>  GSLStatus::EDom,
            2 => GSLStatus::ERange,
            3 => GSLStatus::EFault,
            4 => GSLStatus::EInval,
            6 => GSLStatus::EFactor,
            7 => GSLStatus::ESanity,
            8 => GSLStatus::EnoMem,
            9 => GSLStatus::EBadFunc,
            10 => GSLStatus::ERunaway,
            11 => GSLStatus::EMaxIter,
            12 => GSLStatus:: EZeroDiv,
            13 => GSLStatus::EBadTol,
            14 => GSLStatus::ETol,
            15 => GSLStatus::EUndrflw,
            16 => GSLStatus::EOvrflw,
            17 => GSLStatus::ELoss,
            18 => GSLStatus::ERound,
            19 => GSLStatus::EBadLen,
            20 => GSLStatus::ENotSqr,
            21 => GSLStatus::ESing,
            22 => GSLStatus::EDiverge,
            23 => GSLStatus::EUnsup,
            24 => GSLStatus::EUnimp,
            25 => GSLStatus::ECache,
            26 => GSLStatus::ETable,
            27 => GSLStatus::EnoProg,
            28 => GSLStatus::EnoProgJ,
            29 => GSLStatus::ETolF,
            30 => GSLStatus::ETolX,
            31 => GSLStatus::ETolG,
            _ => GSLStatus::Failure
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn matrix_conversion() {
        let data : Vec<f64> = (0..15).map(|i| i as f64).collect();
        let m = DMatrix::<f64>::from_vec(3,5,data);
        let gsl_m : gsl_matrix = m.clone().into();
        let m2 : DMatrix<f64> = gsl_m.into();
        assert_eq!(m, m2);
    }

    #[test]
    fn vector_conversion() {
        let data : Vec<f64> = (0..15).map(|i| i as f64 + 1000.0).collect();
        let v = DVector::<f64>::from_vec(data);
        let gsl_v : gsl_vector = v.clone().into();
        let v2 : DVector<f64> = gsl_v.into();
        assert_eq!(v, v2);
    }
}
