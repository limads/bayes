// Separate items into clusters as long as dist(t1, t2) < tol, where tol can be an f64 epsilon (strict equality)
// or other relatively small distance value. This is an instance of correlation clustering. In general, if
// the aggregating feature is a binary decision (yes/no), the custering algorithm does not require that
// the number of clusters is specified in advance. If the aggregating feature is continuous (e.g. distance-based)
// then the number of clusters must be specified in advance.
pub struct StrictAggregator {

}


