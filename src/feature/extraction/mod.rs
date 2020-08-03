/// Joint criteria to use to preserve signal features.
pub struct Criteria {

    // Returns the positions of at most the n largest coefficients
    // relative to their local neighborhoods (absolute value
    // difference between largest and smallest coefficients
    // in a neighborhood). The second argument gives the symmetric
    // window size.
    local_maxima : Option<(usize, usize)>,

    // Returns the positions of the overall n largest coefficients in absolute value.
    // This is applied after local_maxima is applied, and the value should
    // be smaller than or equal to the first entry of loc_maxima.
    global_maxima : Option<usize>,

    // Returns the positions of coefficients with absolute value
    // higher than this threshold. This is the last criterion applied.
    thresh : Option<f64>

}

pub struct LocalMaxima {

}

pub struct GlobalMaxima {

}

pub struct Threshold {

}
