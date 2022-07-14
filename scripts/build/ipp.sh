ippcore.h 

bindgen /opt/intel/oneapi/ipp/latest/include/ippcore.h -o src/foreign/ipp/ippcore.rs --no-rustfmt-bindings
rustfmt src/foreign/ipp/ippcore.rs

bindgen /opt/intel/oneapi/ipp/latest/include/ippvm.h -o src/foreign/ipp/ippvm.rs --no-rustfmt-bindings
rustfmt src/foreign/ipp/ippvm.rs

bindgen /opt/intel/oneapi/ipp/latest/include/ippi.h -o src/foreign/ipp/ippi.rs --no-rustfmt-bindings
rustfmt src/foreign/ipp/ippi.rs

bindgen /opt/intel/oneapi/ipp/latest/include/ipps.h -o src/foreign/ipp/ipps.rs --no-rustfmt-bindings
rustfmt src/foreign/ipp/ipps.rs

bindgen /opt/intel/oneapi/ipp/latest/include/ippcv.h -o src/foreign/ipp/ippcv.rs --no-rustfmt-bindings
rustfmt src/foreign/ipp/ippcv.rs
