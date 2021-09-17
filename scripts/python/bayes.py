from ctypes import cdll, c_double, c_void_p, pointer, Structure, c_int, c_ulong, cast, POINTER

lib = cdll.LoadLibrary("target/debug/libbayes.so")
lib.normal_prior.argtypes = [c_double, c_double]
lib.normal_prior.restype = c_void_p

lib.predict.argtypes = [c_void_p]
lib.predict.restype = c_double

# n = lib.normal_prior(c_double(0.0), c_double(1.0))
# lib.predict(n)

def contiguous_access(i, ptr, len):
    if i >= 0:
        if i < len:
            return ptr[i]
        else:
            raise IndexError(f"Index {i} outside range of slice of size {len}")
    else:
        if abs(i) < len:
            return ptr[len - abs(i)]
        else:
            raise IndexError(f"Reverse index {i} outside range of slice of size {len}")

class Vec(Structure):
    _fields_ = [("ptr", POINTER(c_double)), ("capacity", c_ulong), ("len", c_ulong)]

    def __init__(self, cap):
        lib.with_capacity_double.argtypes = [c_ulong]
        lib.with_capacity_double.restype = Vec
        if cap is None:
            self = lib.with_capacity_double(c_ulong(0))
        else:
            self = lib.with_capacity_double(c_ulong(cap))

    def __getitem__(self, i):
        return contiguous_access(i, self.ptr, self.len)

    def push(self, d):
        lib.push_double.argtypes = [POINTER(Vec), c_double]
        lib.push_double.restype = None
        lib.push_double(pointer(self), c_double(d))

    def as_slice(self):
        lib.into_double_boxed.argtypes = [Vec]
        lib.into_double_boxed.restype = Slice
        return lib.into_double_boxed(self)

    # Vecs should never be freed at del because they are just used to build
    # slices internally with the as_slice method call. Only the Boxed slice
    # drop should happen.
    # def __del__(self):
        # lib.free_double_vec.argtypes = [Vec]
        # lib.free_double_vec.restype = None
        #lib.free_double_vec(self)

class SliceIter:

    def __init__(self, slice):
        self.slice = slice
        self.index = -1

    def __next__(self):
        self.index = self.index + 1
        if self.index == self.slice.len:
            raise StopIteration
        return self.slice[self.index]

# Might export that to PyPI as module named "slices". Then at Python, import slices.Slice, and use that
# to work with Rust/Python FFI.
class Slice(Structure):
    _fields_ = [("ptr", POINTER(c_double)), ("len", c_ulong)]

    def __repr__(self):
        s = "Slice(["
        for i in range(self.len - 1):
            s += f"{self[i]}, "
        s += f"{self[self.len-1]}])"
        return s

    def len(self):
        return self.len

    def __init__(self, iterable=[]):
        v = Vec(None)

        for a in iterable:
            v.push(a)

        # lib.into_double_boxed.argtypes = [Vec]
        # lib.into_double_boxed.restype = Slice
        # self = lib.into_double_boxed(v)
        self.ptr = v.ptr
        self.len = v.len

    def __getitem__(self, i):
        return contiguous_access(i, self.ptr, self.len)

    def __iter__(self):
        return SliceIter(self)

    def __del__(self):
        lib.free_double_box.argtypes = [Slice]
        lib.free_double_box.restype = None
        lib.free_double_box(self)

# l = [ctypes.pointer(ctypes.c_int(a)) for a in [1, 2]]
# ctypes.pointer(l[0])
# s = lib.export_slice()
# ptr = cast(s.ptr, POINTER(c_double))

# Access with ptr[0]; ptr[1]; etc.
# Pass pointer to primitive t where t is c_double, c_int, etc.
# bytef(t)
#
# Get content from a pointer type:
# i = c_int(42); pi = pointer(i); pi.contents
#
# Write content using a pointer
# pi.contents = i
# class Normal:
#    def __init__(self, mean, var):
#        self.__ptr = normal_prior(c_double(mean), c_double(var))

