#include "postgres.h"
#include "fmgr.h"

PG_MODULE_MAGIC;

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/*typedef struct Text Text;

typedef struct {
  char _0[0];
} __IncompleteArrayField_c_char;

typedef struct {
  char vl_len_[4];
  __IncompleteArrayField_c_char vl_dat;
} varlena;

typedef struct {
  const uint8_t *data;
  uintptr_t len;
} ByteSlice;

extern uintptr_t bytes_len(const varlena *t);

extern const uint8_t *bytes_ptr(const varlena *t);

extern const varlena *copy_to_pg(ByteSlice s);
*/

/*
Datum add_one(PG_FUNCTION_ARGS) {
  int32   arg = PG_GETARG_INT32(0);
  HeapTupleHeader  t = PG_GETARG_HEAPTUPLEHEADER(0);
  isnull = PG_ARGISNULL(0);
  salary = GetAttributeByNum(t, 1, &isnull);
  PG_RETURN_INT32(arg + 1);
  PG_RETURN_TEXT_P(new_t);
}
*/

PG_FUNCTION_INFO_V1(log_prob);
// double log_prob(text* distr_txt) {
//}

// extern const varlena *palloc_varlena(uintptr_t sz);
// extern ByteSlice read_from_pg(const varlena *arg);
