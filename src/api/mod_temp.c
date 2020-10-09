#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Text Text;

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

double log_prob(const Text *distr_txt);

extern const varlena *palloc_varlena(uintptr_t sz);

extern ByteSlice read_from_pg(const varlena *arg);
