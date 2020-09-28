#include "pg_helper.h"
#include <string.h>

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

ByteSlice read_from_pg(text* arg) {
  ByteSlice s;
  s.len = VARSIZE(arg);
  s.data = VARDATA(arg);
  return s;
}

// Here is how to deliver text data to PostgreSQL. char* is not necessarily
// nul-terminated.
text* copy_to_pg(ByteSlice s) {
  text *dst = (text *) palloc(VARHDRSZ + s.len);
  SET_VARSIZE(dst, VARHDRSZ + s.len);
  memcpy((void*) VARDATA(dst), (void*) s.data, s.len);
  return dst;
}

char* text_ptr(text* t) {
  return (char*) VARDATA(t);
}

size_t text_len(text* t) {
  return VARSIZE(t);
}

/*text * copytext(text *t)
{
    text *new_t = (text *) palloc(VARSIZE(t));
    VARATT_SIZEP(new_t) = VARSIZE(t);
    memcpy((void *) VARDATA(new_t),
           (void *) VARDATA(t),
           VARSIZE(t) - VARHDRSZ);
    return new_t;
}

text * concat_text(text *arg1, text *arg2)
{
    int32 new_text_size = VARSIZE(arg1) + VARSIZE(arg2) - VARHDRSZ;
    text *new_text = (text *) palloc(new_text_size);

    VARATT_SIZEP(new_text) = new_text_size;
    memcpy(VARDATA(new_text), VARDATA(arg1), VARSIZE(arg1) - VARHDRSZ);
    memcpy(VARDATA(new_text) + (VARSIZE(arg1) - VARHDRSZ),
           VARDATA(arg2), VARSIZE(arg2) - VARHDRSZ);
    return new_text;
}*/

