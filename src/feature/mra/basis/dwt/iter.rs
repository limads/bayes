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
    max_lvl : usize,
    curr_lvl : usize,
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
        if C::try_to_usize().is_none() {
            assert!(full.nrows() == full.ncols());
        }
        assert!( (full.nrows() as f64).log2().fract() == 0.0 );
        let max_lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIteratorBase::<&'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>> {
            max_lvl,
            curr_lvl : 0,
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
        let max_lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIteratorBase::<&'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>{
            max_lvl,
            curr_lvl : 0,
            region : DWTFilter::Vertical,
            full
        }
    }

    // Iteration over 1D vectors
    pub fn update_1d(&mut self) -> Option<()> {
        if self.curr_lvl == self.max_lvl + 1 {
            return None;
        }
        self.curr_lvl += 1;
        Some(())
    }

    pub fn update_2d(&mut self) -> Option<()> {
        if self.curr_lvl == self.max_lvl + 1 {
            return None;
        }
        if self.curr_lvl == 0 {
            self.curr_lvl += 1;
            Some(())
        } else {
            let (new_region, new_lvl) = match self.region {
                DWTFilter::Vertical => {
                    (DWTFilter::Both, self.curr_lvl)
                },
                DWTFilter::Both => {
                    (DWTFilter::Horizontal, self.curr_lvl)
                },
                DWTFilter::Horizontal => {
                    (DWTFilter::Vertical, self.curr_lvl + 1)
                }
                };
            self.region = new_region;
            self.curr_lvl = new_lvl;
            Some(())
        }
    }
}

pub type DWTIterator2D<'a> = DWTIteratorBase<&'a DMatrix<f64>>;

impl<'a> DWTIterator2D<'a> {

    pub fn new(full : &'a DMatrix<f64>) -> Self {
        let max_lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIterator2D{ max_lvl, curr_lvl : 0, region : DWTFilter::Vertical, full }
    }

}

impl<'a> Iterator for DWTIterator2D<'a> {

    type Item = DMatrixSlice<'a, f64>;

    fn next(&mut self) -> Option<DMatrixSlice<'a, f64>> {
        let ans = get_level_slice_2d(self.full, self.curr_lvl, self.region)?;
        self.update_2d()?;
        Some(ans)
    }

}

pub type DWTIteratorMut2D<'a> = DWTIteratorBase<&'a mut DMatrix<f64>>;

impl<'a> DWTIteratorMut2D<'a> {

    pub fn new(full : &'a mut DMatrix<f64>) -> Self
        where Self : 'a
    {
        let max_lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIteratorMut2D{ max_lvl, curr_lvl : 0, region : DWTFilter::Vertical, full }
    }

}

pub type DWTIterator1D<'a> = DWTIteratorBase<&'a DVector<f64>>;

impl<'a> Iterator for DWTIterator1D<'a> {

    type Item = DVectorSlice<'a, f64>;

    fn next(&mut self) -> Option<DVectorSlice<'a, f64>>
        where Self : 'a
    {
        let slice = get_level_slice_1d(&self.full, self.curr_lvl)?;
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
        let slice = get_level_slice_1d(&self.full, self.curr_lvl)?;
        //let strides = slice.strides();
        let shape = slice.shape();
        let ptr : *mut f64 = slice.data.ptr() as *mut _;
        self.update_1d()?;
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
        let ans = get_level_slice_2d(self.full, self.curr_lvl, self.region)?;
        let strides = ans.strides();
        let shape = ans.shape();
        let ptr : *mut f64 = ans.data.ptr() as *mut _;
        self.update_2d()?;
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
    let lvl_pow = (2 as i32).pow(lvl.try_into().unwrap()) as usize;
    //let lvl_offset = (0..lvl).fold(0, |p, l| p + (2 as i32).pow(l.try_into().unwrap()));
    let region_offset = match region {
        DWTFilter::Both => (lvl_pow, lvl_pow),
        DWTFilter::Vertical => (lvl_pow as usize, 0),
        DWTFilter::Horizontal => (0, lvl_pow as usize)
    };
    match lvl {
        0 => ((0, 0), (2, 2)),
        _ => (region_offset, (lvl_pow, lvl_pow))
    }
}

fn get_level_slice_2d<'a>(
    m : &'a DMatrix<f64>,
    lvl : usize,
    region : DWTFilter
) -> Option<DMatrixSlice<'a, f64>>
{
    let (off, bounds) = define_level_bounds(lvl, region);
    if bounds.0 > m.nrows() / 2 {
        return None;
    }
    Some(m.slice(off, bounds))
}

fn get_level_slice_1d<'a>(
    v : &'a DVector<f64>,
    lvl : usize
) -> Option<DVectorSlice<'a, f64>> {
    let lvl_pow = (2 as i32).pow(lvl.try_into().unwrap()) as usize;
    if lvl_pow > v.nrows() / 2 {
        return None;
    }
    let (lvl_off, lvl_sz) = match lvl {
        0 => (0, 2),
        l => (lvl_pow, lvl_pow)
    };
    Some(v.rows(lvl_off, lvl_sz))
}

#[test]
fn dwt_iter_1d() {
    let d = DVector::from_row_slice(
        &[0., 0., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.]
    );
    let mut iter = DWTIterator1D::new_ref(&d);
    while let Some(s) = iter.next() {
        println!("{}", s);
    }
}

#[test]
fn dwt_iter_2d() {
    let d = DMatrix::from_row_slice(16, 16,
        &[0., 0., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          0., 0., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
        ]);
    let mut iter = DWTIterator2D::new_ref(&d);
    while let Some(s) = iter.next() {
        println!("{}", s);
    }
}

#[test]
fn dwt_iter_64() {
    let m = DMatrix::zeros(64,64);
    let mut iter = DWTIterator2D::new_ref(&m);
    while let Some(s) = iter.next() {
        println!("{}", s);
    }
}
