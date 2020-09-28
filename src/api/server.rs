use std::slice;
use std::os::raw::c_char;

// Bindgen-generated code to reproduce a small subset of the PG types here.

#[repr(C)]
#[derive(Default)]
pub struct __IncompleteArrayField<T>(::std::marker::PhantomData<T>, [T; 0]);
impl<T> __IncompleteArrayField<T> {
    #[inline]
    pub const fn new() -> Self {
        __IncompleteArrayField(::std::marker::PhantomData, [])
    }
    #[inline]
    pub unsafe fn as_ptr(&self) -> *const T {
        ::std::mem::transmute(self)
    }
    #[inline]
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        ::std::mem::transmute(self)
    }
    #[inline]
    pub unsafe fn as_slice(&self, len: usize) -> &[T] {
        ::std::slice::from_raw_parts(self.as_ptr(), len)
    }
    #[inline]
    pub unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
        ::std::slice::from_raw_parts_mut(self.as_mut_ptr(), len)
    }
}
impl<T> ::std::fmt::Debug for __IncompleteArrayField<T> {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        fmt.write_str("__IncompleteArrayField")
    }
}
impl<T> ::std::clone::Clone for __IncompleteArrayField<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new()
    }
}

#[repr(C)]
pub struct __BindgenUnionField<T>(::std::marker::PhantomData<T>);
impl<T> __BindgenUnionField<T> {
    #[inline]
    pub const fn new() -> Self {
        __BindgenUnionField(::std::marker::PhantomData)
    }
    #[inline]
    pub unsafe fn as_ref(&self) -> &T {
        ::std::mem::transmute(self)
    }
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut T {
        ::std::mem::transmute(self)
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct varlena {
    pub vl_len_: [::std::os::raw::c_char; 4usize],
    pub vl_dat: __IncompleteArrayField<::std::os::raw::c_char>,
}

pub type bytea = varlena;

pub type text = varlena;

pub type BpChar = varlena;

pub type VarChar = varlena;

#[repr(C)]
struct ByteSlice  {
    data : *const u8,
    len : usize
}

// Available at link time from the pg_helper.c module.
#[link(name = "pg_helper", kind="static")]
extern "C" {

    fn read_from_pg(arg : *const text) -> ByteSlice;

    fn copy_to_pg(s : ByteSlice) -> *const text;

    fn text_ptr(t : *const text) -> *const u8;

    fn text_len(t : *const text) -> usize;

}

pub fn utf8_to_str<'a>(txt : *const text) -> &'a str {
    unsafe{ std::str::from_utf8(slice::from_raw_parts(text_ptr(txt), text_len(txt))).unwrap() }
}

/// Copies data from s into a buffer allocated via palloc, returning the
/// newly-allocated data.
fn copy_string_to_pg(s : String) -> *const text {
    let (data, len, cap) = s.into_raw_parts();
    let bs = ByteSlice{ data, len };

    unsafe {
        let txt_ptr = copy_to_pg(bs);

        // Re-build to drop the original content
        String::from_raw_parts(data, len, cap);

        txt_ptr
    }
}



