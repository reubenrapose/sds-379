#include <algorithm>
#include <cmath>
#include <vector>

/*
This file's sole purpose is to calculate the output of a network.
The size must be adjust here.
*/

// Declare size of network
int nodes_per_layer = 6;
int layers = 2;

// Output of network for input size 1
double getFP(double z, std::vector<double> p){
    double output = 0.0;
    int pind = 0; // param index
    
    // construct first layer
    std::vector<double> prevNodes;
    for(int i = 0; i < nodes_per_layer; i++){
        double node = std::tanh(p[pind] * z + p[pind+1]);
        prevNodes.push_back(node);
        pind += 2;        
    }

    // middle layers
    for(int lind = 0; lind < layers-1; lind++){
        std::vector<double> newNodes; // nodes in the following layer

        // nnind --> new node index, pnind --> previous node index
        for(int nnind = 0; nnind < nodes_per_layer; nnind++){
            // weight[pnind,nnind] --> weight between node[pnind] of layer[lind-1] and node[nnind] of layer[lind]
            // the to-be calculated sum(previous node[pnind] * weight[pnind,nnind])
            double weighted = 0.0;
            for(int pnind = 0; pnind < nodes_per_layer; pnind++){
                weighted += p[pind] * prevNodes[pnind];
                pind += 1;
            }
            newNodes.push_back(std::tanh(weighted + p[pind]));
            pind += 1;
        }
        prevNodes = newNodes; // reset to most recently constructed layer
    }

    // final layer
    for(int nind = 0; nind < nodes_per_layer; nind++){
        output += p[pind] * prevNodes[nind];
        pind += 1;
    }
    output += p[pind];

    return output;
}

// Output of network for input size z.size()
double getFPVec(std::vector<double> z, std::vector<double> p){
    double output = 0.0;
    int pind = 0; // param index
    
    // construct first layer
    std::vector<double> prevNodes;
    for(int nnind = 0; nnind < nodes_per_layer; nnind++){ // nnind --> new node index
        double weighted = 0.0;
        for(int zind = 0; zind < z.size(); zind++){
            weighted += z[zind] * p[pind];
            pind += 1;
        }
        prevNodes.push_back(std::tanh(weighted + p[pind]));
        pind += 1;
    }

    // middle layers
    for(int lind = 0; lind < layers-1; lind++){
        std::vector<double> newNodes; // nodes in the following layer

        // nnind --> new node index, pnind --> previous node index
        for(int nnind = 0; nnind < nodes_per_layer; nnind++){
            // weight[pnind,nnind] --> weight between node[pnind] of layer[lind-1] and node[nnind] of layer[lind]
            // the to-be calculated sum(previous node[pnind] * weight[pnind,nnind])
            double weighted = 0.0;
            for(int pnind = 0; pnind < nodes_per_layer; pnind++){
                weighted += prevNodes[pnind] * p[pind];
                pind += 1;
            }
            newNodes.push_back(std::tanh(weighted + p[pind]));
            pind += 1;
        }
        prevNodes = newNodes; // reset to most recently constructed layer
    }

    // final layer
    for(int nind = 0; nind < nodes_per_layer; nind++){
        output += prevNodes[nind] * p[pind];
        pind += 1;
    }
    output += p[pind];

    return output;
}