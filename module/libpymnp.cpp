#include <iostream>

#include "../set_fn/deep.h"
#include "../set_fn/plus_modular.h"
#include "../minimizers/mnp.h"
#include "../la/vector.h"
#include "../la/matrix.h"

//#include "ndarray.h"

class User
{
    std::string name;
    public:
        User(char *name):name(name) {}
        User(std::string &name):name(name) {}

        std::string greet() { return "hello, " + name; }
};

void hello(char *name)
{
    User user(name);
    std::cout << user.greet() << std::endl;
}

double mnp_random_bernoulli() 
{
    int64_t n = 100;

    std::vector<int64_t> layers;
    layers.push_back(10);
    layers.push_back(10);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.2);

    Deep<double> deep(n, layers);
    deep.init_weights(gen, dist);
    deep.rectify = [](double x){ return std::min(x, 1.0); };
    PlusModular<double, Deep<double>> problem(n, std::move(deep), dist);
    auto mnp_A = mnp(problem, 1e-5, 1e-10);
    return problem.eval(mnp_A);
}

int main()
{
    std::cout << mnp_random_bernoulli() << std::endl;
    return 0;
}

extern "C"
{
    extern double mnp_bernoulli()
    {
        return mnp_random_bernoulli();
    }

    // The buffer flat_s_weights stores all weights for the deep submodular function (monotone submodular component) in a flat array.
    // Each layer is stored as a contiguous row-major matrix
    // Layers are stored contiguously from first layer to last
    extern double mnp_deep_contig_w(int64_t n, double* contiguous_s_weights_in, double* m_weights_in, int64_t* layer_sizes_in, int64_t n_layers, double* p_y)
    {
        //Create Deep Submodular Function.
        std::vector<int64_t> layer_sizes(n_layers);
        for(int64_t i = 0; i < n_layers; i++) {
            layer_sizes[i] = layer_sizes_in[i];
        }

        //Initialize submodular function weights
        std::vector<Matrix<double>> s_weights;
        int64_t offset = 0;
        for(int64_t i = 0; i < n_layers; i++) {
            int64_t layer_height = layer_sizes[i];
            int64_t layer_width = (i == 0) ? n : layer_sizes[i-1];

            //Creates a shallow copy, doesn't manage own memory
            //Row major matrix
            s_weights.emplace_back(contiguous_s_weights_in + offset, layer_height, layer_width, layer_width, 1); 
            offset += layer_width * layer_height;
        }

        Deep<double> deep(n, layer_sizes, std::move(s_weights));
//      deep.rectify = [](double x){ return std::min(x, 1.0); };
        deep.rectify = [](double x){ return 1.0 / (1.0 + exp(-x)) ; };

        //Initialize modular function weights
        Vector<double> m_weights(m_weights_in, n);
        PlusModular<double, Deep<double>> problem(n, std::move(deep), std::move(m_weights));

        Vector<double> y(p_y, n);
        auto A_star = mnp(problem, y, 1e-10, 1e-10);
        return problem.eval(A_star);
    }
}
