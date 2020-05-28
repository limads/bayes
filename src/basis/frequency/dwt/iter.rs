use nalgebra::*;
use nalgebra::base::storage::Storage;
use std::convert::TryInto;

#[derive(Clone, Copy)]
pub enum DWTFilter {
    Vertical,   // HL
    Horizontal, // LH
    Both        // HH
}

pub struct DWTIteratorBase<T> {
    lvl : usize,
    region : DWTFilter,
    full : T
}

impl<T> DWTIteratorBase<T> {

    pub fn new_ref<'a, C>(
        full : &'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>
    ) -> DWTIteratorBase<&'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>
        where
            C : Dim
    {
        let lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIteratorBase::<&'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>{
            lvl,
            region : DWTFilter::Vertical,
            full
        }
    }

    pub fn new_mut<'a, C>(
        full : &'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>
    ) -> DWTIteratorBase<&'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>
        where
            C : Dim
    {
        let lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIteratorBase::<&'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>{
            lvl,
            region : DWTFilter::Vertical,
            full
        }
    }

    // Iteration over 1D vectors
    pub fn update_1d(&mut self) -> Option<()> {
        match self.lvl {
            0 => { return None; },
            _ => { self.lvl -= 1; }
        }
        Some(())
    }

    // Iteration over 2D matrices
    pub fn define_next_region(reg : DWTFilter, lvl : usize) -> Option<(DWTFilter, usize)> {
        match lvl {
            0 => None,
            lvl => {
                match reg {
                    DWTFilter::Vertical => {
                        Some((DWTFilter::Both, lvl))
                    },
                    DWTFilter::Both => {
                        Some((DWTFilter::Horizontal, lvl))
                    },
                    DWTFilter::Horizontal => {
                        Some((DWTFilter::Vertical, lvl - 1))
                    }
                }
            }
        }
    }

    pub fn update_region(&mut self) -> Option<()> {
        let (new_region, new_lvl) = Self::define_next_region(self.region, self.lvl)?;
        self.region = new_region;
        self.lvl = new_lvl;
        Some(())
    }

}

pub type DWTIterator2D<'a> = DWTIteratorBase<&'a DMatrix<f64>>;

impl<'a> DWTIterator2D<'a> {

    pub fn new(full : &'a DMatrix<f64>) -> Self {
        let lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIterator2D{ lvl, region : DWTFilter::Vertical, full }
    }

}

impl<'a> Iterator for DWTIterator2D<'a> {

    type Item = DMatrixSlice<'a, f64>;

    fn next(&mut self) -> Option<DMatrixSlice<'a, f64>> {
        let ans = get_level_slice_2d(self.full, self.lvl, self.region);
        self.update_region()?;
        Some(ans)
    }

}

pub type DWTIteratorMut2D<'a> = DWTIteratorBase<&'a mut DMatrix<f64>>;

impl<'a> DWTIteratorMut2D<'a> {

    pub fn new(full : &'a mut DMatrix<f64>) -> Self
        where Self : 'a
    {
        let lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIteratorMut2D{ lvl, region : DWTFilter::Vertical, full }
    }

}

pub type DWTIterator1D<'a> = DWTIteratorBase<&'a DVector<f64>>;

impl<'a> Iterator for DWTIterator1D<'a> {

    type Item = DVectorSlice<'a, f64>;

    fn next(&mut self) -> Option<DVectorSlice<'a, f64>>
        where Self : 'a
    {
        let slice = match self.lvl {
            0 => { return None; },
            lvl => {
                get_level_slice_1d(&self.full, lvl)
            }
        };
        self.update_1d();
        Some(slice)
    }

}

pub type DWTIteratorMut1D<'a> = DWTIteratorBase<&'a mut DVector<f64>>;

impl<'a> Iterator for DWTIteratorMut1D<'a> {

    type Item = DVectorSliceMut<'a, f64>;

    fn next(&mut self) -> Option<DVectorSliceMut<'a, f64>>
        where Self : 'a
    {
        let slice = match self.lvl {
            0 => { return None; },
            lvl => {
                get_level_slice_1d(&self.full, lvl)
            }
        };
        //let strides = slice.strides();
        let shape = slice.shape();
        let ptr : *mut f64 = slice.data.ptr() as *mut _;
        self.update_1d();
        unsafe {
            let storage = SliceStorageMut::from_raw_parts(
                ptr,
                (Dynamic::from_usize(shape.0), U1::from_usize(1)),
                (U1::from_usize(1), Dynamic::from_usize(shape.1))
            );
            let slice_mut = DVectorSliceMut::from_data(storage);
            Some(slice_mut)
        }
    }
}

impl<'a> Iterator for DWTIteratorMut2D<'a> {

    type Item = DMatrixSliceMut<'a, f64>;

    fn next(&mut self) -> Option<DMatrixSliceMut<'a, f64>>
        where Self : 'a
    {
        let ans = get_level_slice_2d(self.full, self.lvl, self.region);
        let strides = ans.strides();
        let shape = ans.shape();
        let ptr : *mut f64 = ans.data.ptr() as *mut _;
        self.update_region()?;
        unsafe {
            let storage = SliceStorageMut::from_raw_parts(
                ptr,
                (Dynamic::from_usize(shape.0), Dynamic::from_usize(shape.1)),
                (U1::from_usize(strides.0), Dynamic::from_usize(strides.1))
            );
            let slice_mut = DMatrixSliceMut::from_data(storage);
            Some(slice_mut)
        }
    }

}

fn define_level_bounds(lvl : usize, region : DWTFilter) -> ((usize, usize), (usize, usize)) {
    let lvl_sz = (2 as i32).pow(lvl.try_into().unwrap()) as usize;
    //let lvl_offset = (0..lvl).fold(0, |p, l| p + (2 as i32).pow(l.try_into().unwrap()));
    let region_offset = match region {
        DWTFilter::Both => (lvl_sz, lvl_sz),
        DWTFilter::Vertical => (lvl_sz as usize, 0),
        DWTFilter::Horizontal => (0, lvl_sz as usize)
    };
    (region_offset, (lvl_sz, lvl_sz))
}

fn get_level_slice_2d<'a>(
    m : &'a DMatrix<f64>,
    lvl : usize,
    region : DWTFilter
) -> DMatrixSlice<'a, f64>
{
    let (off, bounds) = define_level_bounds(lvl, region);
    m.slice(off, bounds)
}

fn get_level_slice_1d<'a>(
    v : &'a DVector<f64>,
    lvl : usize
) -> DVectorSlice<'a, f64> {
    let lvl_sz = (2 as i32).pow(lvl.try_into().unwrap()) as usize;
    v.rows(lvl_sz, lvl_sz)
}

