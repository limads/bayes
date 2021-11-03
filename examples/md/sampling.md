```bash
bundle add -p /home/diego/Software/bayes
weave examples/md/sampling.md -o examples/html/sampling.html
```

```toml
[dependencies]
bayes = { path = "/home/diego/Software/bayes" }
```

# How to build this document

This is a literate Rust document that can be built with the [weave](http://crates.io/literate)
command line tool. 

Example:

```bash
cargo install literate
weave -d '{ bayes = "0.0.1" }
weave examples/md/sampling.md -o examples/html/sampling.html && 
    firefox examples/html/sampling.html
```

# Sampling

Sampling is very easy:

```rust

// Use declarations might point to local modules.
mod mymod {

}

// Use external module. All use declarations must point to local or external crate modules.
use bayes::prob::*;
// use bayes::prob::Prior;
// use rand::distributions::Distribution;
// use rand_distr::Distribution;
// use rand::distributions::Distribution;
use std::default::Default;

// use rng::thread_rng();

let mut n = Normal::prior(1.0, None);
let v : f64 = n.sample(&mut rand::thread_rng());
v
```


