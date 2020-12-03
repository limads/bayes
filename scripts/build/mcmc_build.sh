# This script will verify if the user has mcmclib, stat and gcem installed on his
# system as per instructions in https://github.com/kthohr/mcmc. If not, it will
# attempt to clone the code repositories, and compile the libraries into target/debug/deps
# or target/release/deps depending on the build strategy. The user must have the C++
# toolchain installed (Make, g++). 

cd target && git clone https://github.com/kthohr/mcmc && cd mcmc
mkdir ../debug/deps/lib

# Install include and lib dirs under target/debug
./configure -i $(pwd)/../debug/deps && make && make install

# Call the compilation steps at the crate root
cd ..
g++ -c -fPIC src/foreign/mcmc/mcmc.cpp -o target/debug/deps/lib/mcmc/mcmc.o \
	-I./target/debug/deps/include/mcmc/mcmclib/include -I./target/debug/deps/include/gcem/include \
	-I./lib/mcmc/stats/include -L./lib -L./lib/mcmc/mcmclib/lib -lmcmc
g++ -shared lib/mcmc/mcmc.o -o lib/mcmc/libmcmcwrapper.so 

# Install gcem header-only library (mostly special functions)
cd target
git clone https://github.com/kthohr/gcem /target/debug/deps/include
g++ -c src/foreign/gcem.cpp -o target/debug/gcem.o -I./target/debug/deps/include/gcem/include

# Compile bindings to gcem

bindgen src/foreign/gcem.h -o src/foreign/gcem.rs --no-rustfmt-bindings
rustfmt src/foreign/gcem.rs --force
rm src/foreign/gcem.rs.bk


