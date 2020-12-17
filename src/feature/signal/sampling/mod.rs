/// Represents a downsampling action, which might imply a low-pass filtering and
/// taking every ith value of a buffer. The downsampling ratio should be defined
/// by the dimension mismatch between the source and the destination. Usually, source/dest
/// will have their dimensions mismatching by a fixed factor. Might work in either direction
/// from owned or borrowed structures (downsampling into a mutable window should change the reference
/// it points into, assuming a same lifetime from both buffers).
pub trait Downsample<D> {
    
    fn downsample(&self, dst : &mut D);
    
}

/// Represents an upsampling action, which usually implies interpolation. Like downsample,
/// this trait works by querying the dimension mismatch between input and output. Might work
/// from dense-to-dense or sparse-to-dense structures (in which case we have a simple
/// upsampling operation). As Downsample, it might work in either direction
/// from owned or borrowed structures.
pub trait Upsample<D> {

    fn upsample(&self, dst : &mut D);
    
}

pub(crate) mod slices;
