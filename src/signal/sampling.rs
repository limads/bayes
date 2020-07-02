use nalgebra::Scalar;

#[inline(always)]
pub fn convert_f32_slice(src : &[u8], dst : &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f32;
    }
}

#[inline(always)]
pub fn convert_f64_slice(src : &[u8], dst : &mut [f64]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = *s as f64;
    }
}

#[inline(always)]
pub fn convert_f32_slice_strided(src : &[u8], dst : &mut [f32], cstride : usize) {
    for i in 0..dst.len() {
        dst[i] = src[i*cstride] as f32
    }
}

#[inline(always)]
pub fn convert_f64_slice_strided(src : &[u8], dst : &mut [f64], cstride : usize) {
    for i in 0..dst.len() {
        dst[i] = src[i*cstride] as f64
    }
}

#[inline(always)]
pub fn subsample<T>(content : &[T], dst : &mut [T], ncols : usize, sample_n : usize)
    where T : Scalar + Copy
{
    assert!(ncols < content.len(), "ncols smaller than content length");
    assert!(content.len() % ncols == 0);
    let nrows = content.len() / ncols;
    let sparse_ncols = if ncols > 1 { ncols / sample_n } else { 1 };
    let sparse_nrows = nrows / sample_n;
    if dst.len() != sparse_nrows * sparse_ncols {
        panic!("Dimension mismatch");
    }
    for r in 0..sparse_nrows {
        for c in 0..sparse_ncols {
            dst[r*sparse_ncols + c] = content[r*sample_n*ncols + c*sample_n];
        }
    }
}

#[inline(always)]
pub fn subsample_convert_generic<T,U>(content : &[T], dst : &mut [U], ncols : usize, sample_n : usize, transpose : bool)
    where
        T : Into<U>,
        T : Scalar + Copy,
        U : Scalar + Copy
{
    assert!(ncols < content.len(), "ncols smaller than content length");
    assert!(content.len() % ncols == 0);
    let nrows = content.len() / ncols;
    let sparse_ncols = if ncols > 1 { ncols / sample_n } else { 1 };
    let sparse_nrows = nrows / sample_n;
    if dst.len() != sparse_nrows * sparse_ncols {
        panic!("Dimension mismatch");
    }
    for r in 0..sparse_nrows {
        for c in 0..sparse_ncols {
            let dst_ix = if transpose { r + c*sparse_nrows } else { r*sparse_ncols + c };
            dst[dst_ix] = content[r*sample_n*ncols + c*sample_n].into();
        }
    }
}

#[inline(always)]
pub fn subsample_convert_f32(content : &[u8], dst : &mut [f32], ncols : usize, sample_n : usize,  transpose : bool) {
    assert!(ncols < content.len(), "ncols smaller than content length");
    assert!(content.len() % ncols == 0);
    let nrows = content.len() / ncols;
    let sparse_ncols = if ncols > 1 { ncols / sample_n } else { 1 };
    let sparse_nrows = nrows / sample_n;
    if dst.len() != sparse_nrows * sparse_ncols {
        panic!("Dimension mismatch");
    }
    for r in 0..sparse_nrows {
        for c in 0..sparse_ncols {
            let dst_ix = if transpose { r + c*sparse_nrows } else { r*sparse_ncols + c };
            dst[dst_ix] = content[r*sample_n*ncols + c*sample_n] as f32;
        }
    }
}

#[inline(always)]
pub fn subsample_convert_f64(content : &[u8], dst : &mut [f64], ncols : usize, sample_n : usize,  transpose : bool) {
    assert!(ncols < content.len(), "ncols smaller than content length");
    assert!(content.len() % ncols == 0);
    let nrows = content.len() / ncols;
    let sparse_ncols = if ncols > 1 { ncols / sample_n } else { 1 };
    let sparse_nrows = nrows / sample_n;
    if dst.len() != sparse_nrows * sparse_ncols {
        panic!("Dimension mismatch");
    }
    for r in 0..sparse_nrows {
        for c in 0..sparse_ncols {
            let dst_ix = if transpose { r + c*sparse_nrows } else { r*sparse_ncols + c };
            dst[dst_ix] = content[r*sample_n*ncols + c*sample_n] as f64;
        }
    }
}

/* pub fn decode(&self, dec : &[u8]) -> Result<DMatrix<f32>, &'static str> {
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

