from ctypes import cdll, c_double, c_void_p, pointer, Structure, c_int, cast, POINTER

lib = cdll.LoadLibrary("target/debug/libbayes.so")
lib.normal_prior.argtypes = [c_double, c_double]
lib.normal_prior.restype = c_void_p

lib.predict.argtypes = [c_void_p]
lib.predict.restype = c_double

n = lib.normal_prior(c_double(0.0), c_double(1.0))
lib.predict(n)

lib.export_slice.argtypes = []
lib.export_slice.restype = Slice

class Slice(Structure):
    _fields_ = [("ptr", POINTER(c_double)), ("len", c_int)]

    def __getitem__(self, i):
        if i >= 0:
            if i < self.len:
                return self.ptr[i]
            else:
                raise IndexError(f"Index {i} outside range of slice of size {self.len}")
        else:
            if abs(i) < self.len:
                return self[self.len - abs(i)]
            else:
                raise IndexError(f"Reverse index {i} outside range of slice of size {self.len}")

s = lib.export_slice()

ptr = cast(s.ptr, POINTER(c_double))

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

