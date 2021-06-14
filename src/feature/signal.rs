use std::iter::Iterator;

/// Processes are a sequence of random scalars or vector realizations that are indexed
/// by an orderable index (such as usize). This is a tool to model real-world uni- or
/// multivariate temporal signals or natural/transformed images (by marching over rows or columns of a
/// contiguous image memory buffer, or any transforms applied to this buffer, in which case the
/// "state" is the current row or column interpreted as a vector). The advantage of marching over
/// rows (or row subslices) is that we can consider them (borrowed) column vectors
/// without copying image data.
pub trait Process<'a>
where
    Self : Iterator<Item=&'a f64>
{

    pub fn innovation<F>(&'a mut self, F) -> Option<&'a f64>
    where
        F : Fn(&f64) -> f64
    {
        // self.next()
        // Generate a new value from a rng and a closure that takes the last value
        // and this RNG and applies a function.
        unimplemeneted!()
    }

    pub fn generate(&'a mut self, n : usize) -> Self {
        let mut vals = Vec::new();
        let mut ix = 0;
        while let Some(v) = self.next() {
            if ix == 0 {
                vals.push(v);
            } else {
                let last = vals[ix];
                vals.push(v + last);
            }
        }
        unimplemented!()
    }
}


