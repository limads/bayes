// Represents a distribution by a random, exchangeable sequence of draws from a source distribution.
// This can be the basis for importance sampling: let p : Draw<MultiNormal> = p.draw();, then use
// the weights f(x|theta)h(theta)/p(theta) for draw distribution p to calculate estimates.
pub struct Draw {

}


