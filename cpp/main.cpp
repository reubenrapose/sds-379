#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sstream>

#ifndef LOSS_FUNCTIONS
#define LOSS_FUNCTIONS
    #include "loss_functions.cpp"
#endif
#include "gradients.cpp"


double PI = std::atan(1)*4;
int lossRisingMax = 10000;

std::string printVec(std::vector<double> v){
    std::string output = "[";
    for(int i = 0; i < v.size()-1; i++){
        output += std::to_string(v[i]) + ", ";
    }
    output += std::to_string(v[v.size()-1]) + "]";
    return output;
}

std::string printVec(std::vector<std::vector<double>> v){
    std::string output;
    for(int i = 0; i < v.size()-1; i++){
        std::string miniOutput = "[";
        for(int j = 0; j < v[i].size()-1; j++){
            miniOutput += std::to_string(v[i][j]) + ", ";
        }
        miniOutput += std::to_string(v[i][v[i].size()-1]) + "]";
        output += miniOutput + ", ";
    }
    std::string miniOutput = "[";
    for(int j = 0; j < v[v.size()-1].size()-1; j++){
        miniOutput += std::to_string(v[v.size()-1][j]) + ", ";
    }
    miniOutput += std::to_string(v[v.size()-1][v[v.size()-1].size()-1]) + "]";
    output += miniOutput;
    return output;
}

// 1 dimensional mesh
std::vector<double> getMesh(int size, double xlow, double xup){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(xlow, xup);

    std::vector<double> mesh;
    for(int i = 0; i < size; i++){
        mesh.push_back(dist(gen));
    }
    return mesh;
}

// 2 dimensional mesh
std::vector<std::vector<double>> getMesh(int size, double xlow, double xup, double tlow, double tup){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distx(xlow, xup);
    std::uniform_real_distribution<double> distt(tlow, tup);

    std::vector<std::vector<double>> mesh;
    for(int i = 0; i < size; i++){
        mesh.push_back({distx(gen), distt(gen)});
    }
    return mesh;
}

// 2-d mesh for an initial condition
std::vector<std::vector<double>> getMeshIC(int size, double xlow, double xup){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distx(xlow, xup);

    std::vector<std::vector<double>> mesh;
    for(int i = 0; i < size; i++){
        mesh.push_back({distx(gen), 0});
    }
    return mesh;
}

// 2-d mesh for boundary condition at xval = boundary
std::vector<std::vector<double>> getMeshBC(int size, double tlow, double tup, double xval){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distt(tlow, tup);

    std::vector<std::vector<double>> mesh;
    for(int i = 0; i < size; i++){
        mesh.push_back({xval, distt(gen)});
    }
    return mesh;
}


void trainGradientDescent(std::string introMessage,
                            std::string fileName,
                            std::vector<double> z, std::vector<double> p,
                            double lr, int maxIters, double tol){
    std::cout << std::setprecision(10) << std::fixed;

    // used for rerandomizing mesh
    auto minmax = std::minmax_element(z.begin(), z.end());
    double xlow = *minmax.first;
    double xup = *minmax.second;

    int iter = 1;
    double loss;
    double prevLoss;
    std::vector<double> gradCopy;
    int lossRisingCount = 0; // end algorithm early if loss function is increasing

    std::ofstream MyFile(fileName);
    MyFile << introMessage << "\n\n";
    std::string message;

    auto start = std::chrono::steady_clock::now();
    do {
        auto startIter = std::chrono::steady_clock::now();
        bool print = iter == 1 || iter % 5000 == 0; // decide how often to print (progress tracker)

        prevLoss = loss;

        // chosen out of {lossFunctionApprox, lossODE}}
        loss = lossFunctionApprox1(z, p); 
        
        // chosen out of {gradientFuncApprox, gradientODE}
        std::vector<double> grad = gradientFuncApprox1(z, p);

        if(print){
            message = "Iter: " + std::to_string(iter) + " || Fitting Error: " + std::to_string(loss);
            message += "\ngrad: " + printVec(grad);
            MyFile << message << "\n";
            std::cout << message << "\n";
        }
        std::vector<double> gradCopy = grad;

        // gradient descent formula: pnew = pold - lr * grad(loss wrt pold)
        std::transform(grad.begin(), grad.end(), grad.begin(), [lr](double x){return x*lr;});
        std::transform(p.begin(), p.end(), grad.begin(), p.begin(), std::minus<double>());
        
        if(print){
            message = "p: " + printVec(p);
            MyFile << message << "\n\n";
            std::cout << message << "\n\n";
        }

        if(loss > prevLoss){
            lossRisingCount++;
        }
        else {
            lossRisingCount = 0; // reset if loss decreases
        }
        if(lossRisingCount > lossRisingMax){
            message = "FITTING ERROR RISING!!!\n";
            message += "Iter: " + std::to_string(iter) + " || Fitting Error: " + std::to_string(loss) + "\n";
            message += "grad: " + printVec(gradCopy) + "\n";
            message += "p: " + printVec(p) + "\n";
            MyFile << message << "\n";
            std::cout << message << "\n";
            break;
        }
        iter++;
        auto stopIter = std::chrono::steady_clock::now();
        if(print){
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopIter - startIter);
            message = "Single Iteration Execution Time: " + std::to_string(duration.count() * 1e-6) + " seconds";
            MyFile << message << "\n\n";
            std::cout << message << "\n\n";
        }

        if(print){ // optional rerandomize
            std::vector<double> z = getMesh(z.size()-3, xlow, xup);
            z.push_back(xlow);
            z.push_back(xup);
            z.push_back(1/6);
        }
    }
    while(iter <= maxIters && loss > tol);

    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    message = "Execution time: " + std::to_string(duration.count() * 1e-6) + " seconds\n";
    MyFile << message << "\n";
    std::cout << message << "\n";
    if(iter > maxIters){
        message = "Max Iterations Reached.";
        MyFile << message << "\n";
        std::cout << message << "\n";
    }

    message = "Final Iter: " + std::to_string(iter) + " || Fitting Error: " + std::to_string(loss);
            message += "\ngrad: " + printVec(gradCopy);
            MyFile << message << "\n";
            std::cout << message << "\n";
    message += "p: " + printVec(p);
            MyFile << message << "\n\n";
            std::cout << message << "\n\n";

    MyFile << message;
    std::cout << message << std::endl;
    
    MyFile.close();

    std::cout << "Complete." << std::endl;
}

// runs same as trainGradientDescent except handles more inputs and PDE function calls
void trainGradientDescentPDE(std::string introMessage,
                            std::string fileName,
                            std::vector<std::vector<double>> zpde, 
                            std::vector<std::vector<double>> zic,
                            std::vector<std::vector<double>> zbc1,
                            std::vector<std::vector<double>> zbc2, 
                            std::vector<double> p,
                            double lr, int maxIters, double tol){

    std::cout << std::setprecision(10) << std::fixed;

    int iter = 1;
    double loss;
    double prevLoss;
    std::vector<double> gradCopy;
    int lossRisingCount = 0;

    std::ofstream MyFile(fileName);
    MyFile << introMessage << "\n\n";
    std::string message;

    auto start = std::chrono::steady_clock::now();
    do {
        auto startIter = std::chrono::steady_clock::now();
        bool print = iter == 1 || iter % 10000 == 0;

        prevLoss = loss;

        loss = lossPDE2(zpde, zic, zbc1, zbc2, p);
        
        std::vector<double> grad = gradientPDE(zpde, zic, zbc1, zbc2, p);
        if(print){
            message = "Iter: " + std::to_string(iter) + " || Fitting Error: " + std::to_string(loss);
            message += "\ngrad: " + printVec(grad);
            MyFile << message << "\n";
            std::cout << message << "\n";
        }

        std::vector<double> gradCopy = grad;

        std::transform(grad.begin(), grad.end(), grad.begin(), [lr](double x){return x*lr;});
        std::transform(p.begin(), p.end(), grad.begin(), p.begin(), std::minus<double>());

        if(print){
            message = "p: " + printVec(p);
            MyFile << message << "\n\n";
            std::cout << message << "\n\n";
        }

        if(loss > prevLoss){
            lossRisingCount++;
        }
        else {
            lossRisingCount = 0;
        }
        if(lossRisingCount > lossRisingMax){
            message = "FITTING ERROR RISING!!!\n";
            message += "Iter: " + std::to_string(iter) + " || Fitting Error: " + std::to_string(loss) + "\n";
            message += "grad: " + printVec(gradCopy);
            MyFile << message << "\n";
            std::cout << message << "\n";
            break;
        }
        iter++;
        auto stopIter = std::chrono::steady_clock::now();
        if(print){
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopIter - startIter);
            message = "Single Iteration Execution Time: " + std::to_string(duration.count() * 1e-6) + " seconds";
            MyFile << message << "\n\n";
            std::cout << message << "\n\n";
        }
    }
    while(iter <= maxIters && loss > tol);

    auto stop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    message = "Execution time: " + std::to_string(duration.count() * 1e-6) + " seconds\n";
    MyFile << message << "\n";
    std::cout << message << "\n";
    if(iter > maxIters){
        message = "Max Iterations Reached.";
        MyFile << message << "\n";
        std::cout << message << "\n";
    }

    message = "Final Iter: " + std::to_string(iter) + " || Fitting Error: " + std::to_string(loss);
            message += "\ngrad: " + printVec(gradCopy);
            MyFile << message << "\n";
            std::cout << message << "\n";
    message += "p: " + printVec(p);
            MyFile << message << "\n\n";
            std::cout << message << "\n\n";

    MyFile << message;
    std::cout << message << std::endl;

    MyFile.close();

    std::cout << "Complete." << std::endl;
}

std::vector<double> arange(double start, double end, double step) {
    std::vector<double> result;
    for (double value = start; value < end; value += step) {
        result.push_back(value);
    }
    return result;
}

/*
The following two functions are for predicting the model based on parameter vector p.
However, this only works using the custom getFP and getFPVec NN evaluators,
and cannot be translated to other models.
It is also important to ensure the network size matches that of the file in which p was found.
*/
void modelPredict1(std::vector<double> p){
    // Create first sequence
    std::vector<double> z = arange(0, 1.0, 1.0/102);

    // Insert true function to approximate into final argument of transform call
    std::vector<double> fz(z.size());
    std::transform(z.begin(), z.end(), fz.begin(), F4);

    // Concatenate the two output sequences
    std::vector<double> ybar(z.size());
    std::transform(z.begin(), z.end(), ybar.begin(), [p](double x){return getFP(x, p);});

    // Print concatenated sequence
    std::cout << "z: " << printVec(z) << std::endl;
    std::cout << "fz: " << printVec(fz) << std::endl;
    std::cout << "ybar: " << printVec(ybar) << std::endl;
}

void modelPredict2(std::vector<double> p){
    // Create x sequence
    std::vector<double> x = arange(0, 1.0, .02);

    // Create t sequence
    std::vector<double> t = arange(0, 1.0, .02);
    
    std::vector<std::vector<double>> z;
    for(int i = 0; i < x.size(); i++){
        z.push_back({x[i], t[i]});
    }

    // Create new vectors for transformed sequences
    std::vector<double> fz;
    std::vector<double> ybar;

    // The following matches Matlab's meshgrid construction for plotting
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 50; ++j) {
            double value = exp(-4*PI * t[j]) * sin(2*PI * x[i]);
            fz.push_back(value);
            ybar.push_back(getFPVec({x[i], t[j]}, p));
            
        }
    }

    std::cout << "z: " << printVec(z) << std::endl;
    std::cout << "fz: " << printVec(fz) << std::endl;
    std::cout << "ybar: " << printVec(ybar) << std::endl;
}

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distp(-1.0, 1.0); // parameters, do not change

    // Set domain
    double xlow = 0;
    double xup = 1;
    double tlow = 0;
    double tup = 1;

    // 1-d mesh, adjust as needed
    std::vector<double> z = getMesh(50, xlow, xup);
    z.push_back(xlow);
    z.push_back(xup);
    z.push_back(1/6);
    
    // 2-d mesh for PDE
    std::vector<std::vector<double>> zpde = getMesh(25, xlow, xup, tlow, tup);
    std::vector<std::vector<double>> zic = getMeshIC(10, xlow, xup);
    std::vector<std::vector<double>> zbc1 = getMeshBC(10, tlow, tup, xlow);
    std::vector<std::vector<double>> zbc2 = getMeshBC(10, tlow, tup, xup);

    int inputSize = 1;
    
    int num_params = nodes_per_layer*inputSize + nodes_per_layer // input and output layer weights
                        + pow(nodes_per_layer, 2) * (layers-1) // middle layer weights
                        + nodes_per_layer*layers + 1; // biases

    // create initial weights
    std::vector<double> p;
    for(int i = 0; i < num_params; i++){
        p.push_back(distp(gen));
    }

    // tunable training parameters for standard gradient descent
    double learningRate = .01;
    int maxIters = 800000;
    double tol = .00001;

    /* 
    The following is for recording summary of training for storage.
    It is meant to follow a loose format, but should contain any information that is deemed necessary. 
    */

    std::stringstream introMessage;
    // introMessage << "Approximate soln to u_t - 1/pi * u_{xx} = 0,\n"
    //                 << "u(x,0) = sin(2*pi*x), u(0,t) = u(1,t) = 0\n"
    //                 << "Domain: x in [0, 1]x[0,1]\n"
    //                 << "Mesh Size: " << zpde.size() << "\n\n"
    //                 << "Network Size: "
    //                 << nodes_per_layer << "x" << layers << "\n"
    //                 << "Learning Rate: " << double(learningRate) << "\n"
    //                 << "Maximum Iterations: " << maxIters << " || Tolerance: " << tol << "\n\n";

    introMessage << "Approximate f(x) = -3x+1 on [0, 1/6], sin(pi*x) on [1/6, 1]\n"
                    << "Domain: [0,1]\n"
                    << "Mesh Size (remesh, bdy pts, x=1/6): " << z.size() << "\n\n"
                    << "Network Size: "
                    << nodes_per_layer << "x" << layers << "\n"
                    << "Learning Rate: " << double(learningRate) << "\n"
                    << "Maximum Iterations: " << maxIters << " || Tolerance: " << tol << "\n\n";
                                
    // Should be updated every run to avoid overwriting files
    std::string fileName = "function_approximation20.txt";

    trainGradientDescent(introMessage.str(), fileName, z, p, learningRate, maxIters, tol);
    // trainGradientDescentPDE(introMessage.str(), fileName, zpde, zic, zbc1, zbc2, p, learningRate, maxIters, tol);


    return 0;
}

