import cffi
import numpy as np

ffi = cffi.FFI()
ffi.cdef("double mnp_bernoulli();")
ffi.cdef("double mnp_deep_contig_w(int64_t n, double* contiguous_s_weights, double* m_weights, int64_t* layer_sizes, int64_t n_layers);")
C = ffi.dlopen("./build/libpymnp.dylib")
    
def mnp_deep_contig_w(n, submodular_w, modular_w, layers, n_layers):
    p_s_w = ffi.cast("double *", ffi.from_buffer(submodular_w))
    p_m_w = ffi.cast("double *", ffi.from_buffer(modular_w))
    p_l = ffi.cast("int64_t *", ffi.from_buffer(layers))

    return C.mnp_deep_contig_w(ffi.cast("int64_t", n), p_s_w, p_m_w, p_l, ffi.cast("int64_t", n_layers))

def mnp_bernoulli():
    return C.mnp_bernoulli()

# Rules: 
#   1. Every matrix is column major
#   2. Every pointer is a numpy ndarray
#   3. Every numpy object has its datatype declared explicitly

def mnp():
    n = 100
    n_layers = 3
    p = 0.2

    layers = np.zeros(n_layers, dtype=np.int64)
    layers.fill(10)
    print(layers)

    tot_weights = 0
    for i in range(n_layers):
        layer_h = layers[i]
        layer_w = n if i == 0 else layers[i-1]
        tot_weights += layer_h * layer_w
   
    submodular_w = np.zeros(tot_weights, dtype=np.float64)
    for i in range(tot_weights):
        submodular_w[i] = 1.0 if np.random.random() < p else 0.0

    modular_w = np.zeros(n, dtype=np.float64)
    for i in range(n):
        modular_w[i] = 2.0 * (np.random.random() - 0.5)

    return mnp_deep_contig_w(n, submodular_w, modular_w, layers, n_layers)

if __name__ == "__main__":
    #f_a = mnp_bernoulli()
    #print(f_a)

    f_a = mnp()
    print(f_a)

