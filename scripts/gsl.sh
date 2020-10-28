# Not required
# pub type size_t = usize;
# cp /usr/include/gsl/gsl_block_double.h headers/
# cp /usr/include/gsl/gsl_vector.h headers/
# cp /usr/include/gsl/gsl_multifit.h headers/
# Remove all headers

bindgen /usr/include/gsl/gsl_block_double.h -o src/gsl/block_double.rs \
    --no-layout-tests --no-rustfmt-bindings \
    --whitelist-function "gsl.*"
rustfmt src/gsl/block_double.rs --force

bindgen /usr/include/gsl/gsl_vector_double.h -o src/gsl/vector_double_src.rs \
    --no-layout-tests --no-rustfmt-bindings \
    --no-recursive-whitelist \
    --whitelist-type "gsl_vector.*|_gsl.*" --whitelist-function "gsl.*"
rustfmt src/gsl/vector_double_src.rs --force
sed '1s;^;use crate::gsl::block_double::*\;\n\n;' src/gsl/vector_double_src.rs > src/gsl/vector_double.rs
rm src/gsl/vector_double_src.rs

bindgen /usr/include/gsl/gsl_matrix_double.h -o src/gsl/matrix_double_src.rs \
    --no-layout-tests --no-rustfmt-bindings \
    --no-recursive-whitelist \
    --whitelist-type "gsl_matrix.*|_gsl.*" --whitelist-function "gsl.*"
rustfmt src/gsl/matrix_double_src.rs --force
sed '1s;^;use crate::gsl::block_double::*\;\nuse crate::gsl::vector_double::*\;\n\n;' src/gsl/matrix_double_src.rs > src/gsl/matrix_double.rs
rm src/gsl/matrix_double_src.rs

bindgen /usr/include/gsl/gsl_multifit.h -o src/gsl/multifit.rs \
    --no-layout-tests --no-recursive-whitelist --no-rustfmt-bindings \
    --whitelist-type "size_t|gsl_multifit.*" --whitelist-function "gsl_multifit.*"
rustfmt src/gsl/multifit.rs --force
sed -i '1s;^;use crate::gsl::vector_double::*\;\nuse crate::gsl::matrix_double::*\;\nuse crate::gsl::block_double::*\;\n\n;' src/gsl/multifit.rs

# The GslMatrix/GslVectors should be erased after bindings are generated, and re-imported on the
# generated bindgen sources from gsl_multifit.rs so Rust recognizes them as the same type.
bindgen /usr/include/gsl/gsl_multifit_nlinear.h -o src/gsl/multifit_nlinear.rs \
	--no-rustfmt-bindings --no-layout-tests --no-recursive-whitelist \
	--whitelist-type "size_t|gsl_multifit_nlinear.*" \
	--whitelist-function "gsl_multifit_nlinear.*" \
	--whitelist-var "gsl_multifit_nlinear.*"
rustfmt src/gsl/multifit_nlinear.rs --force
sed -i '1s;^;use crate::gsl::vector_double::*\;\nuse crate::gsl::matrix_double::*\;\nuse crate::gsl::block_double::*\;\n\n;' src/gsl/multifit_nlinear.rs

bindgen /usr/include/gsl/gsl_rng.h -o src/gsl/rng.rs \
	--no-rustfmt-bindings --no-layout-tests \
	--no-recursive-whitelist \
	--whitelist-type "size_t|FILE|_IO_lock_t|__off64_t|__off_t|_IO_marker|gsl_rng.*|_IO_codecvt|_IO_wide_data" \
	--whitelist-function "gsl_rng.*" \
	--whitelist-var "gsl_rng.*"
rustfmt src/gsl/rng.rs --force

bindgen /usr/include/gsl/gsl_randist.h -o src/gsl/randist.rs \
	--no-rustfmt-bindings --no-layout-tests \
	--no-recursive-whitelist \
	--whitelist-type "size_t.*|gsl_ran.*|gsl_rng.*" --whitelist-function "gsl_ran|gsl_cdf.*"
rustfmt src/gsl/randist.rs --force
sed -i '1s;^;use crate::gsl::vector_double::*\;\nuse crate::gsl::matrix_double::*\;\nuse crate::gsl::block_double::*\;use crate::gsl::rng;\n\n;' src/gsl/randist.rs
#use crate::gsl::vector_double::*;
#use crate::gsl::matrix_double::*;
#use crate::gsl::block_double::*;
#use crate::gsl::rng::*;

#bindgen /usr/include/gsl/gsl_matrix_double.h -o src/gsl/matrix_double.rs \
#	--no-rustfmt-bindings --no-layout-tests --no-derive-copy --no-derive-debug \
#	--whitelist-function "gsl_matrix.*"
#rustfmt src/gsl/matrix_double.rs --force

bindgen /usr/include/gsl/gsl_errno.h -o src/gsl/errno.rs \
	--no-rustfmt-bindings --no-layout-tests --no-derive-copy --no-derive-debug
rustfmt src/gsl/errno.rs --force

bindgen /usr/include/gsl/gsl_multimin.h -o src/gsl/multimin.rs \
	--no-rustfmt-bindings --no-layout-tests --no-recursive-whitelist \
	--whitelist-type "gsl_multimin.*" \
	--whitelist-function "gsl_multimin.*" \
	--whitelist-var "gsl_multimin.*"
rustfmt src/gsl/multimin.rs --force
sed -i '1s;^;use crate::gsl::vector_double::*\;\nuse crate::gsl::matrix_double::*\;\nuse crate::gsl::block_double::*\;\n\n;' src/gsl/multimin.rs

bindgen /usr/include/gsl/gsl_bspline.h -o src/gsl/bspline_src.rs \
    --no-layout-tests --no-derive-copy --no-derive-debug --no-rustfmt-bindings \
    --no-recursive-whitelist \
    --whitelist-type "gsl_bspline.*" \
    --whitelist-function "gsl_bspline.*" \
    --whitelist-var "gsl_bspline.*"
rustfmt src/gsl/bspline_src.rs --force
sed '1s;^;use crate::gsl::vector_double::*\;\n\n;' src/gsl/bspline_src.rs > src/gsl/bpsline.rs
rm src/gsl/bspline_src.rs

bindgen /usr/include/gsl/gsl_sf_gamma.h -o src/gsl/gamma.rs \
    --no-layout-tests --no-derive-copy --no-derive-debug --no-rustfmt-bindings \
    --no-recursive-whitelist \
    --whitelist-function "gsl_sf_gamma"
    --whitelist-function "gsl_sf_lngamma"
    --whitelist-function "gsl_sf_gammainv"
rustfmt src/gsl/gamma.rs --force

rm src/gsl/*.bk

# Still has to manyally remove a few constants re-defined multiple places at multifit_nlinear.rs
# Still has to allow non-snake case and other style issues.
