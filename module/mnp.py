import math
import cffi
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# C FFI Stuff
ffi = cffi.FFI()
ffi.cdef("double mnp_bernoulli();")
ffi.cdef("double mnp_deep_contig_w(int64_t n, double* contiguous_s_weights, double* m_weights, int64_t* layer_sizes, int64_t n_layers, double* y);")
ffi.cdef("double dsf_eval(int64_t n, double* contiguous_s_weights, double* m_weights, int64_t* layer_sizes, int64_t n_layers, bool* A);")
ffi.cdef("double watts_strogatz_graph_cut_yprime(int64_t n, int64_t k, double beta, double* y);")
C = ffi.dlopen("./build/libpymnp.dylib")
    
def mnp_deep_contig_w(n, layer_sizes, submodular_w, modular_w):
    p_s_w = ffi.cast("double *", ffi.from_buffer(submodular_w))
    p_m_w = ffi.cast("double *", ffi.from_buffer(modular_w))
    p_l   = ffi.cast("int64_t *", ffi.from_buffer(layer_sizes))

    y = np.zeros(n, dtype=np.float64)
    p_y   = ffi.cast("double *", ffi.from_buffer(y))
    F_A_star = C.mnp_deep_contig_w(ffi.cast("int64_t", n), p_s_w, p_m_w, p_l, ffi.cast("int64_t", len(layer_sizes)), p_y)
    return F_A_star, y

def dsf_eval(n, layer_sizes, submodular_w, modular_w, A):
    p_s_w = ffi.cast("double *", ffi.from_buffer(submodular_w))
    p_m_w = ffi.cast("double *", ffi.from_buffer(modular_w))
    p_l   = ffi.cast("int64_t *", ffi.from_buffer(layer_sizes))
    p_A   = ffi.cast("bool *", ffi.from_buffer(A))

    F_A = C.dsf_eval(ffi.cast("int64_t", n), p_s_w, p_m_w, p_l, ffi.cast("int64_t", len(layer_sizes)), p_A)
    return F_A


def watts_strogatz_graph_cut_yprime(n, k, beta):
    y = np.zeros(n, dtype=np.float64)
    p_y   = ffi.cast("double *", ffi.from_buffer(y))
    F_A_star = C.watts_strogatz_graph_cut_yprime(ffi.cast("int64_t", n), ffi.cast("int64_t", k), ffi.cast("double", beta), p_y)
    return F_A_star, y


def mnp_bernoulli():
    return C.mnp_bernoulli()

# Rules: 
#   1. Every matrix is row major
#   2. Every pointer is a numpy ndarray
#   3. Every numpy object has its datatype declared explicitly
#This is a test function for calling mnp from python
def mnp():
    n = 100
    n_layers = 3
    p = 0.2

    layers = np.zeros(n_layers, dtype=np.int64)
    layers.fill(10)

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
    
    F_A_star, y  = mnp_deep_contig_w(n, layers, submodular_w, modular_w)
    return F_A_star

def assemble_dsf(n, layer_sizes, sweights):
    layers = []
    offset = 0
    layer_n = n 
    for i in range(len(layer_sizes)):
        #Determine m and n
        layer_m = layer_sizes[i]
       
        #Append layer
        layer_W = sweights[offset:offset + layer_m * layer_n].reshape(layer_m, layer_n)
        layer_W = Variable(layer_W, requires_grad = True)
        layers.append(layer_W)

        offset += layer_m * layer_n
        layer_n = layer_m
    return layers

def disassemble_dsf_grads(layers, n_weights):
    offset = 0
    sgrad = torch.zeros(n_weights, dtype=torch.float64)
    for layer in layers:
        m, n = layer.shape
        sgrad[offset:offset + m*n] = layer.grad.detach().reshape(m*n)
        offset += m * n
    return sgrad 

def eval_dsf(layers, x):
    y = x;
    for W in layers:
        y = torch.mv(W, y)
        y = 2.0 * torch.sigmoid(y) - torch.ones_like(y)
    return y


class DeepSubmodular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s_weights, m_weights, n, layer_sizes=10*np.ones(3)):
        F_A_star, y_prime = mnp_deep_contig_w(n, layer_sizes,
                s_weights.detach().numpy(), m_weights.detach().numpy())
        y_prime = torch.tensor(y_prime)
        
        ctx.save_for_backward(y_prime, s_weights, m_weights)
        ctx.layer_sizes = layer_sizes
        ctx.n = n
        return y_prime
    
    @staticmethod
    def backward(ctx, grad_output):
        yprime, weights_submodular, weights_modular = ctx.saved_tensors
        n = ctx.n
        layer_sizes = ctx.layer_sizes

        #Create the DSF layers
        layers = assemble_dsf(n, layer_sizes, weights_submodular)
        
        #Get groups to form Delta y
        grad_x = torch.zeros(n, dtype=torch.float64)
        values = np.unique(yprime.detach().numpy())
        groups = [(yprime.detach().numpy() == v).reshape(yprime.size()) for v in values] #TODO: this really is terrible inefficient
        for group in groups:
            grad_x.numpy()[group] = np.mean(grad_output.numpy()[group])

        #Get the gradient of each F(A + sigma_i) - F(A)
        sigma = yprime.argsort(descending=True)
        grad_weights_submodular = torch.zeros_like(weights_submodular)
        dFAold_dW = torch.zeros_like(weights_submodular)
        x = torch.zeros(n, dtype=torch.float64)
        dummy_optimizer = torch.optim.SGD(layers, lr=0.0)
        for i in range(n):
            with torch.enable_grad():
                dummy_optimizer.zero_grad()

                x[sigma[i]] = 1.0
                y = eval_dsf(layers, x)
                fa = torch.sum(y)
                fa.backward()
                
                dFA_dW = disassemble_dsf_grads(layers, weights_submodular.shape)
                grad_gain = dFA_dW - dFAold_dW
                dFAold_dW = dFA_dW

                grad_weights_submodular += grad_x[sigma[i]].item() * grad_gain

        grad_weights_modular = grad_x

        return grad_weights_submodular, grad_weights_modular, None, None

def gen_deep_submodular_bernoulli(n, layers, p=0.2):
    n_layers = len(layers)
   
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
    
    return submodular_w, modular_w

class LogQ(nn.Module):
    def __init__(self, A_gt):
        super(LogQ, self).__init__()
        self.A_gt = A_gt
        self.Ac_gt = torch.ones_like(A_gt, dtype=torch.float64) - A_gt

    def forward(self, y):
        part1 = torch.log(1.0 + torch.exp(y))
        part2 = torch.log(1.0 + torch.exp(-y))
        return part1.dot(self.A_gt) + part2.dot(self.Ac_gt)

if __name__ == "__main__":
