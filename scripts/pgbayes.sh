# bindgen src/api/pg_helper.h --no-recursive-whitelist --whitelist-type "ByteSlice" \
# 	--whitelist-function "read_from_pg|copy_to_pg" \
#	-o src/api/pg_helper.rs

# Compile 'helper' module - Carries a few Postgre types and functions, mostly for
# variable-length SQL types (text, bytea, etc).
# -fPIC is used for dll - not static libs
gcc -c src/api/pg_helper.c -o target/c/pg_helper.o -I`pg_config --includedir-server`
ar rcs target/c/pg_helper.a target/c/pg_helper.o

cbindgen src/api/mod.rs -o src/api/mod_temp.c --lang C
sed '1s;^;#include "postgres.h"\n#include "fmgr.h"\n\nPG_MODULE_MAGIC\;\n\n;' src/api/mod_temp.c > src/api/mod.c
rm src/api/mod_temp.c
gcc -c src/api/mod.c -fPIC -o target/c/bayes.o -I`pg_config --includedir-server` -ltarget/release/bayes.a
gcc target/c/bayes.o -shared -o target/c/bayes.so

