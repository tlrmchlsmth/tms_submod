#ifndef DEEP_H
#define DEEP_H

#include "submodular.h"
#include "../la/vector.h"
#include "../la/matrix.h"

template<class DT>
DT rectify_sqrt(DT x) {
    assert(x >= 0);
    return sqrt(x);
}

template<class DT>
DT rectify_min_1_x(DT x) {
    assert(x >= 0);
    return std::min(1.0, x);
}

template<class DT>
class Deep : public SubmodularFunction<DT> {
public:
    int64_t n;
    DT (*rectify)(DT);

    std::vector<Matrix<DT>> layers;
    Vector<DT> final_layer;
    std::vector<Vector<DT>> inputs;

    void init_layers(const std::vector<int64_t>& layer_sizes) {
        layers.reserve(layer_sizes.size());
        inputs.reserve(layer_sizes.size()+1);

        int64_t previous_layer_size = n;
        for(int layer = 0; layer < layer_sizes.size(); layer++) {
            layers.emplace_back(layer_sizes[layer], previous_layer_size);
            previous_layer_size = layer_sizes[layer];
        }

        //Create workspace for the model
        inputs.emplace_back(n);
        for(int layer = 0; layer < layer_sizes.size(); layer++) {
            inputs.emplace_back(layer_sizes[layer]);
        }
    }

    template<class RNG, class DIST>
    void init_weights(RNG &gen, DIST &dist) {
        for(int layer = 0; layer < layers.size(); layer++) {
            layers[layer].fill_rand(gen, dist);
        }
        final_layer.set_all(1.0);
    }

    //Generate default function
    Deep(int64_t n_in) : SubmodularFunction<DT>(n_in), 
        n(n_in), final_layer(4), layers(), inputs(), rectify(rectify_sqrt)
    {
        std::vector<int64_t> layer_sizes;
        layer_sizes.push_back(16);
        layer_sizes.push_back(4);
        init_layers(layer_sizes);
        
        std::random_device rd; 
        std::mt19937 gen{rd()};
        std::bernoulli_distribution dist(0.3);
        init_weights(gen, dist);
    }
    
    Deep(int64_t n_in, const std::vector<int64_t>& layer_sizes) : SubmodularFunction<DT>(n_in), 
        n(n_in), final_layer(layer_sizes.back()), layers(), inputs(), rectify(rectify_sqrt)
    {
        init_layers(layer_sizes);
    }

    //Constructor taking weights using move semantics
    Deep(int64_t n_in, const std::vector<int64_t>& layer_sizes, std::vector<Matrix<DT>>&& weights) : SubmodularFunction<DT>(n_in), 
        n(n_in), final_layer(layer_sizes.back()), rectify(rectify_min_1_x), layers(weights)
    {
        //Create workspace for the model
        inputs.emplace_back(n);
        for(int layer = 0; layer < layer_sizes.size(); layer++) {
            inputs.emplace_back(layer_sizes[layer]);
        }

        final_layer.set_all(1.0);
    }

    //Constructor taking weights using copy semantics
    Deep(int64_t n_in, const std::vector<int64_t>& layer_sizes, const std::vector<Matrix<DT>>& weights) : SubmodularFunction<DT>(n_in), 
        n(n_in), final_layer(layer_sizes.back()), rectify(rectify_min_1_x), layers(weights)
    {
        //Create workspace for the model
        inputs.emplace_back(n);
        for(int layer = 0; layer < layer_sizes.size(); layer++) {
            inputs.emplace_back(layer_sizes[layer]);
        }

        final_layer.set_all(1.0);
    }
    
    DT eval(const std::vector<bool>& A) 
    {
        assert(A.size() == n);

        //Initialize first layer's input
        inputs[0].set_all(0.0);
        for(int i = 0; i < n; i++) {
            if(A[i]) { 
                inputs[0](i) = 1.0; 
            }
        }

        //Middle layers
        for(int layer = 0; layer < layers.size(); layer++) {
            layers[layer].mvm(1.0, inputs[layer], 0.0, inputs[layer+1]);
            rectify_vec(inputs[layer+1]);
        }

        //Final layer
        return final_layer.dot(inputs.back());
    }

    //Only limited savings from gains fn
    //We can compute 1st layer (pre rectification) once,
    //also we can compute modular component once
    virtual void gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        x.set_all(0.0);

        Vector<DT> layer1_ws = Vector<DT>(inputs[1].length());
        layer1_ws.set_all(0.0);

        DT FA_old = 0.0;
        for(int i = 0; i < n; i++) {
            layer1_ws.axpy(1.0, layers[0].subcol(perm[i]));
            rectify_vec(inputs[1], layer1_ws);

            for(int layer = 1; layer < layers.size(); layer++) {
                layers[layer].mvm(1.0, inputs[layer], 0.0, inputs[layer+1]);
                rectify_vec(inputs[layer+1]);
            }
            DT FA = final_layer.dot(inputs.back());
            x(perm[i]) += FA - FA_old;
            FA_old = FA;
        }
    }
    
    //Helper rectifiers
    template<class DT>
    void rectify_vec(Vector<DT>& y, const Vector<DT>& x) {
        assert(y.length() == x.length());
        _Pragma("omp parallel for")
        for(int64_t i = 0; i < x.length(); i++) {
            assert(x(i) >= 0);
            y(i) = rectify(x(i));
        }
    }

    template<class DT>
    void rectify_vec(Vector<DT>& x) {
        //_Pragma("omp parallel for")
        for(int64_t i = 0; i < x.length(); i++) {
            assert(x(i) >= 0);
            x(i) = rectify(x(i));
        }
    }

};

#endif
