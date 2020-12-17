# To use MKL in C, one usually #include <mkl/mkl.h>.
# This will automatically include all the headers below. In Rust,
# we want to separate them into modules (since the generated file by including
# everything at the same time is too big).

# bindgen /usr/include/mkl/mkl_version.h -o mkl_version.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_version.rs --force

# bindgen /usr/include/mkl/mkl_types.h -o mkl_types.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_types.rs --force

# bindgen /usr/include/mkl/mkl_blas.h -o mkl_blas.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_blas.rs --force

# bindgen /usr/include/mkl/mkl_trans.h -o mkl_trans.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_trans.rs --force

# bindgen /usr/include/mkl/mkl_cblas.h -o mkl_cblas.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_cblas.rs --force

# bindgen /usr/include/mkl/mkl_spblas.h -o mkl_spblas.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_spblas.rs --force

# bindgen /usr/include/mkl/mkl_lapack.h -o mkl_lapack.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_lapack.rs --force

# bindgen /usr/include/mkl/mkl_lapacke.h -o mkl_lapacke.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_lapack.rs --force

# bindgen /usr/include/mkl/mkl_pardiso.h -o mkl_pardiso.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_pardiso.rs --force

# Note : Manually delete the included definitions of mkl_types.h after source is generated.
# bindgen sparse_handle_aux.h -o mkl_sparse_handle.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_sparse_handle.rs --force

# bindgen /usr/include/mkl/mkl_dss.h -o mkl_dss.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_dss.rs --force

# bindgen /usr/include/mkl/mkl_rci.h -o mkl_rci.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_rci.rs --force

# bindgen /usr/include/mkl/mkl_vml.h -o mkl_vml.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_vml.rs --force

bindgen /usr/include/mkl/mkl_vsl.h -o src/mkl/vsl.rs --verbose --no-rustfmt-bindings
rustfmt mkl_vsl.rs --force

# bindgen /usr/include/mkl/mkl_df.h -o mkl_df.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_df.rs --force

# bindgen /usr/include/mkl/mkl_service.h -o mkl_service.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_service.rs --force

bindgen /usr/include/mkl/mkl_dfti.h -o /src/mkl/dfti.rs --verbose --no-rustfmt-bindings
rustfmt mkl_dfti.rs --force

# bindgen /usr/include/mkl/mkl_trig_transforms.h -o mkl_trig_transforms.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_trig_transforms.rs --force

# bindgen /usr/include/mkl/mkl_poisson.h -o mkl_poisson.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_poisson.rs --force

# bindgen /usr/include/mkl/mkl_solvers_ee.h -o mkl_solvers_ee.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_solvers_ee.rs --force

# bindgen /usr/include/mkl/mkl_direct_call.h -o mkl_direct_call.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_direct_call.rs --force

# bindgen /usr/include/mkl/mkl_dnn.h -o mkl_dnn.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_dnn.rs --force

# bindgen /usr/include/mkl/mkl_compact.h -o mkl_compact.rs --verbose --no-rustfmt-bindings
# rustfmt mkl_compact.rs --force

rm src/mkl/*.bk
