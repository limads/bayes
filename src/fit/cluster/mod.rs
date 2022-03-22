// TODO build Aggregtor trait shared by all clustering algorithms.

use std::cmp::Eq;

/// Hierarchical clustering implementation
pub mod dendro;

/// Spatial density clustering implementation
pub mod space;

/// Centroid-based (or prototype-based) clustering implementation
pub mod center;

pub(crate) mod distance;

/*pub struct Cluster<'a, T> {
    items : Box<dyn Iterator<Item=&'a T>
}

pub trait Clustering<'a> {

    pub fn clusters<T, M>(&'a mut self, pts : &'a [T]) -> Vec<Cluster<'a, T>>
    where
        T : Distance<M>;

}

pub struct Clusters<'a> {
    vec::Drain<Cluster<'a, T>>
}

impl Iterator for Clusters {

}
pub trait ClusterAdapter {

    pub fn cluster_by(&sef, clustering : C)
    where
        C : Clustering;

}

impl<I> ClusterAdapter for I
where I : Iterator<Item=T>
{

    pub fn cluster(&self);

    pub fn cluster_by(&self);

}*/

/// Types which can be thought of as living in a space and can have their distance compared.
/// The generic type parameter M designates a metric. Types might implement multiple metrics
/// when this make sense, or might implement just a subset of them. Metric is usually a zero-sized
/// type just used to disambiguate from them. Perhaps this generic type should be used to define
/// valid metrics for a type.
pub trait Distance<M>
where
    M : ?Sized
{

	fn distance(&self, other : &Self) -> f64;

}

pub trait Metric<T>
where
    T : Distance<Self> + ?Sized
{

    fn metric(a : &T, b : &T) -> f64;

}

pub struct Euclidian { }

impl<T> Metric<T> for Euclidian
where
    T : Distance<Self> + ?Sized
{

    fn metric(a : &T, b : &T) -> f64 {
        a.distance(b)
    }

}

pub struct Manhattan { }

impl<T> Metric<T> for Manhattan
where
    T : Distance<Self> + ?Sized
{

    fn metric(a : &T, b : &T) -> f64 {
        a.distance(b)
    }

}

pub struct Hamming { }

impl<T> Metric<T> for Hamming
where
    T : Distance<Self> + ?Sized{

    fn metric(a : &T, b : &T) -> f64 {
        (a).distance(b)
    }

}

impl<T, U> Distance<Euclidian> for (T, U)
where
    f64 : From<T>,
    T : Copy,
    f64 : From<U>,
    U : Copy
{

    fn distance(&self, other : &Self) -> f64 {
        Euclidian::metric(&[f64::from(self.0), f64::from(self.1)][..], &[f64::from(other.0), f64::from(other.1)][..])
    }

}

impl<T, U> Distance<Manhattan> for (T, U)
where
    f64 : From<T>,
    T : Copy,
    f64 : From<U>,
    U : Copy
{

    fn distance(&self, other : &Self) -> f64 {
        Manhattan::metric(&[f64::from(self.0), f64::from(self.1)][..], &[f64::from(other.0), f64::from(other.1)][..])
    }

}

impl<T, U, V> Distance<Euclidian> for (T, U, V)
where
    f64 : From<T>,
    T : Copy,
    f64 : From<U>,
    U : Copy,
    f64 : From<V>,
    V : Copy
{

    fn distance(&self, other : &Self) -> f64 {
        Euclidian::metric(&[f64::from(self.0), f64::from(self.1), f64::from(self.2)][..], &[f64::from(other.0), f64::from(other.1), f64::from(other.2)][..])
    }

}

impl<T, U, V> Distance<Manhattan> for (T, U, V)
where
    f64 : From<T>,
    T : Copy,
    f64 : From<U>,
    U : Copy,
    f64 : From<V>,
    V : Copy
{

    fn distance(&self, other : &Self) -> f64 {
        Manhattan::metric(&[f64::from(self.0), f64::from(self.1), f64::from(self.2)][..], &[f64::from(other.0), f64::from(other.1), f64::from(other.2)][..])
    }

}

impl<T> Distance<Euclidian> for [T]
where
    f64 : From<T>,
    T : Copy
{

    fn distance(&self, other : &Self) -> f64 {
        assert!(self.len() == other.len());
        self.iter().zip(other.iter())
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powf(2.) )
            .sum::<f64>()
            .sqrt()
    }

}

impl<const N : usize> Distance<Euclidian> for [f64; N]
{

    fn distance(&self, other : &Self) -> f64 {
        Euclidian::metric(&self[..], &other[..])
    }

}

impl<T> Distance<Manhattan> for [T]
where
    f64 : From<T>,
    T : Copy
{

    fn distance(&self, other : &Self) -> f64 {
        assert!(self.len() == other.len());
        self.iter().zip(other.iter())
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs() )
            .sum::<f64>()
    }

}

impl<const N : usize> Distance<Manhattan> for [f64; N]
{

    fn distance(&self, other : &Self) -> f64 {
        Manhattan::metric(&self[..], &other[..])
    }

}

impl<T> Distance<Hamming> for [T]
where
    T : Eq
{

    fn distance(&self, other : &Self) -> f64 {
        assert!(self.len() == other.len());
        self.iter().zip(other.iter())
            .map(|(a, b)| if a.eq(b) { 0.0 } else { 1.0 } )
            .sum::<f64>()
    }

}

impl<T, const N : usize> Distance<Hamming> for [T; N]
where
    T : Eq
{

    fn distance(&self, other : &Self) -> f64 {
        Hamming::metric(&self[..], &other[..])
    }

}

mod tests {

    use super::*;

    #[test]
    fn euclidian() {

        #[repr(C)]
        pub struct MyType {
            a : i32,
            b : i32
        }

        impl MyType {
            fn empty() -> Self {
                Self { a : 0, b : 0 }
            }
        }

        impl Distance<Euclidian> for MyType {
            fn distance(&self, other : &Self) -> f64 {
                Euclidian::metric(self.as_ref(), other.as_ref())
            }
        }

        impl AsRef<[i32]> for MyType {
            fn as_ref(&self) -> &[i32] {
                unsafe { std::slice::from_raw_parts(&self.a as *const _, 2) }
            }
        }

        // No disambiguation required
        let a : [bool; 2] = [true,true];
        let b : [bool; 2] = [true,true];
        println!("{}", Hamming::metric(&a[..], &b[..]) );

        // Disambiguation required.
        let a : [u32; 2] = [1,1];
        let b : [u32; 2] = [1,1];

        println!("{}", Hamming::metric(&a[..], &b[..]) );

        // Must disambiguate here.
        let a : [f64; 2] = [1.,1.];
        let b : [f64; 2] = [1.,1.];
        println!("{}", Euclidian::metric(&a[..], &b[..]) );
        // println!("{}", a.distance(&b) );

        // No ambiguity here.
        let a = MyType::empty();
        let b = MyType::empty();
        println!("{}", Euclidian::metric(&a, &b) );
        println!("{}", a.distance(&b) );

    }
}

// impl Distance<Metric=Euclidian>
/*use std::cmp::Eq;

/// Perhaps make Metric a super-trait of the specific metrics.
/// Then we use GATs to derive the specific implementations:
/// impl Metric<Norm> for T where T : RealMetric
/// impl Metric<Discrete> for T where T : DiscreteMetric
/// impl Metric<Max> for T where T : MaxMetric (zero norm metric)
/// Then the algorithm traits Cluster and Matching just need to
/// be generic over this GAT, which just need to know that the Metric::metric function
/// outputs an f64, where the implementation of this function is dispatched to the
/// specific implementation.
pub trait DiscreteMetric {

    fn discrete(&self, other : &Self) -> f64;

}

/// TODO also offer generic implementation for RealMetric where T : AsRef<[f64]>. If the
/// user type does not satisfy that, the user just needs to use cluster_by(|a| [a.0 as f64, a.1 as f64] ),
/// which will implement AsRef<[f64]>.
impl<T> DiscreteMetric for T
where
    T : Eq
{

    fn discrete(&self, other : &Self) -> f64 {
        if *self == *other {
            1.0
        } else {
            0.0
        }
    }
}

// impl Metric for &dyn DiscreteMetric {
// }

/// This trait represents metrics generated by a norm (translation-invariant, scale-commutative).
/// To use the infinity norm (highest value), set Order=0 (norm-zero metric). Perhaps create a separate trait for this case.
pub trait Metric {

    /// Metric function, generic over the metric order: 0 (Infinity), 1 (Manhattan), 2 (Euclidian).. pth (Minkowsky).
    fn minkowsky<const O : usize>(&self, other : &Self) -> f64;

    fn euclidian(&self, other : &Self) -> f64 {
        self.minkowsky::<2>(&other)
    }

    fn manhattan(&self, other : &Self) -> f64 {
        self.minkowsky::<1>(&other)
    }

    // As p->inf the root -> 0, and the result -> 1, making the norm the limiting value (highest element in vector).
    // fn minkowsky(&self, other : &Self, order : usize) -> f64 {
    //    unimplemented!()
    // }
}

/*/// Implement FromIter for this, so the user can cluster into a vector then call Clusters::from(vec![n1, n2]);
/// This should be an iterable over &[M; N] by calling vec.iter() in its implementation.
pub struct Clusters<M, const N : usize> {
    clusters : Vec<[M; N]>
}

pub trait Cluster
where
    Self::Item : Metric
{

    type Item;

    // Cluster items into clusters of size N by a given metric defined at runtime.
    fn cluster<const N : usize>(&self) -> Clusters<M, N>;

    // Cluster items by a custom metric function.
    fn cluster_by<const N : usize>(&self, f : Fn(&Self::Item)->f64) -> Clusters<M, N>;

}

/// Returns best matches over two sets (not necessarily of same size).
pub trait Match
where
    Self::Item : Metric
{

    type Item;

    // item.map(|a| [a.0 as f64, a.1 as f64] ) over some item already implements cluster, so cluster_by
    // is not really required...
    fn matches(&self, other : impl Iterator<Item=M>) -> (M, M);

    fn match_by(&self, other : impl Iterator<Item =
}

impl<I, M> Cluster<M> for I
where
    I : Iterator<Item=M>
    M : Metric
{

}

impl<I, M> Match<M> for I
where
    I : Iterator<Item=M>
    M : Metric
{

}

*/

/// Perhaps rename "metric" as "real" since the objects are a point in real space
/// not the metric. Then real types are types for which a distance might be calculated.
pub trait Real {

}

/// Or implement generically for T : AsRef<[f64]>. But then we couldn't implement for
/// (T, T), etc. But having it in this way allow the user to implement it trivially
/// for his type as minkowsky(mytype.as_ref()) any time his type is AsRef<[f64]>.
impl Metric for [f64] {

    fn minkowsky<const O : usize>(&self, other : &Self) -> f64 {
        generic_metric::<O>(self.as_ref(), other.as_ref())
    }

}

impl<T> Metric for (T, T)
where
    T : Into<f64> + Copy
{
    fn minkowsky<const O : usize>(&self, other : &Self) -> f64 {
        generic_metric::<O>(&[self.0.into(), self.1.into()], &[other.0.into(), other.1.into()])
    }
}

pub fn generic_metric<const O : usize>(a : &[f64], b : &[f64]) -> f64 {
    assert!(a.len() >= 1 && b.len() >= 1);
    match O {
        0 => {
            // Just take the max of the elements.
            unimplemented!()
        },
        1 => {
            a.iter().zip(b.iter()).map(|(a, b)| (a - b).abs() ).sum::<f64>()
        },
        2 => {
            a.iter().zip(b.iter()).map(|(a, b)| (a - b).powf(2.) ).sum::<f64>().sqrt()
        },
        p => {
            let root = 1. / p as f64;
            a.iter().zip(b.iter())
                .map(|(a, b)| (a - b).powf(p as f64) )
                .sum::<f64>()
                .powf(root)
        }
    }
}

// pub mod cluster;

/*pub struct Apple {
    width : f64
}

pub struct Orange {
    width : f64
}

let apple = Apple { width : f64 };
let orange = Orange { widht : f64 };

// Can't compare that!
apple.manhattan(&orange)

// Ok here.
orange.euclidian(&orange, 2);*/
*/

