#ifndef DEEP_H
#define DEEP_H

#include "submodular.h"
#include "../la/vector.h"
#include "../la/matrix.h"

template<class DT>
DT rectify(DT x) {
    assert(x >= 0);
    return sqrt(x);
}

template<class DT>
void rectify(Vector<DT>& y, const Vector<DT>& x) {
    assert(y.length() == x.length());
    _Pragma("omp parallel for")
    for(int64_t i = 0; i < x.length(); i++) {
        assert(x(i) >= 0);
        y(i) = rectify(x(i));
    }
}

template<class DT>
void rectify(Vector<DT>& x) {
    _Pragma("omp parallel for")
    for(int64_t i = 0; i < x.length(); i++) {
        assert(x(i) >= 0);
        x(i) = rectify(x(i));
    }
}


template<class DT>
class Deep : public SubmodularFunction<DT> {
public:
    int64_t n;

    //Monotone Submodular Component
    std::vector<Matrix<DT>> layers;
    Vector<DT> final_layer; //(row vector)

    //Modular Component
    Vector<DT> modular;
    
    std::vector<Vector<DT>> inputs;

    void constructor_helper(std::vector<int64_t>& layer_sizes) {
        std::random_device rd; 
        std::mt19937 gen{rd()};
        std::uniform_real_distribution<DT> uniform(0.0, 1.1);
        
        layers.reserve(layer_sizes.size());
        inputs.reserve(layer_sizes.size()+1);

        int64_t previous_layer_size = n;
        for(int layer = 0; layer < layer_sizes.size(); layer++) {
            layers.emplace_back(layer_sizes[layer], previous_layer_size);
            previous_layer_size = layer_sizes[layer];
        }

        //Generate monotone submodular model with random weights
        //First layer
        for(int layer = 0; layer < layer_sizes.size(); layer++) {
            layers[layer].fill_rand(gen, uniform);
        }
        final_layer.fill_rand(gen, uniform);


        //Create workspace for the model
        inputs.emplace_back(n);
        for(int layer = 0; layer < layer_sizes.size(); layer++) {
            inputs.emplace_back(layer_sizes[layer]);
        }
        
        //Random nonmonotone modular component
        std::normal_distribution<DT> normal(0.0, 0.05);
        modular.fill_rand(gen, normal);
    }

    Deep(int64_t n_in) : SubmodularFunction<DT>(n_in), 
        n(n_in), final_layer(3), layers(), inputs(), modular(n) 
    {
        std::vector<int64_t> layer_sizes;
        layer_sizes.push_back(9);
        layer_sizes.push_back(3);
        constructor_helper(layer_sizes);

    }
    
    Deep(int64_t n_in, std::vector<int64_t>& layer_sizes) : SubmodularFunction<DT>(n), 
        n(n_in), final_layer(layer_sizes.back()), layers(), inputs(), modular(n)
    {
        constructor_helper(layer_sizes);
    }


    DT eval(const std::vector<bool>& A) 
    {
        //Initialize first layer's input
        //Also compute modular component while we're at it
        DT modular_component = 0.0;
        inputs[0].set_all(0.0);
        for(int i = 0; i < n; i++) {
            if(A[i]) { 
                inputs[0](i) = 1.0; 
                modular_component += modular(i);
            }
        }

        //Middle layers
        for(int layer = 0; layer < layers.size(); layer++) {
            layers[layer].mvm(1.0, inputs[layer], 0.0, inputs[layer+1]);
            rectify(inputs[layer+1]);
        }

        //Final layer
        DT submodular_component = final_layer.dot(inputs.back());
        
        return rectify(submodular_component) + modular_component;
    }

    //Only limited savings from gains fn
    //We can compute 1st layer (pre rectification) once,
    //also we can compute modular component once
    virtual void gains(const std::vector<int64_t>& perm, Vector<DT>& x) 
    {
        x.copy(modular);

        Vector<DT> layer1_ws = Vector<DT>(inputs[1].length());
        layer1_ws.set_all(0.0);

        DT FA_old = 0.0;
        for(int i = 0; i < n; i++) {
            layer1_ws.axpy(1.0, layers[0].subcol(perm[i]));
            rectify(inputs[1], layer1_ws);

            for(int layer = 1; layer < layers.size(); layer++) {
                layers[layer].mvm(1.0, inputs[layer], 0.0, inputs[layer+1]);
                rectify(inputs[layer+1]);
            }

            DT FA = rectify(final_layer.dot(inputs.back()));
            x(perm[i]) += FA - FA_old;
            FA_old = FA;
        }
    }

};

#endif
