#include <cmath>
#include <vector>

#include "nn_functions.cpp"

/*
Each loss function is some version of the Mean Squared Error
Each function is numbered according to the example in its respective chapter.
*/

double F1(double z){
    return std::exp(std::sin(M_1_PI*z));
}

double F2(double z){
    return .1*std::pow(z, 5) + 2*std::pow(z, 4) - std::pow(z, 3) - 2*std::pow(z, 2) + z;
}

double F3(double z){
    return std::sin(std::pow(z, 3)) * std::pow(z, -1.1);
}

double F4(double z){
    if(z <= 1/6){
        return -3*z+1;
    }
    else{
        return std::sin(M_1_PI*z);
    }
}

double lossFunctionApprox1(std::vector<double> z, std::vector<double> p){
    double output = 0;
    for(int i = 0; i < z.size(); i++){
        output += pow(getFP(z[i], p) - F1(z[i]), 2);
    }
    return output;
}

double lossFunctionApprox2(std::vector<double> z, std::vector<double> p){
    double output = 0;
    for(int i = 0; i < z.size(); i++){
        output += pow(getFP(z[i], p) - F2(z[i]), 2);
    }
    return output;
}

double lossFunctionApprox3(std::vector<double> z, std::vector<double> p){
    double output = 0;
    for(int i = 0; i < z.size(); i++){
        output += pow(getFP(z[i], p) - F3(z[i]), 2);
    }
    return output;
}

double lossFunctionApprox4(std::vector<double> z, std::vector<double> p){
    double output = 0;
    for(int i = 0; i < z.size(); i++){
        output += pow(getFP(z[i], p) - F4(z[i]), 2);
    }
    return output;
}

// ODE: dydx + yx = 0
double getFPODE1(double z, std::vector<double> p){
    double fpz = getFP(z, p);

    double h = 1e-5;
    double dfpz = (getFP(z+h, p) - getFP(z-h, p))/(2*h);

    double fpOde = dfpz + z * fpz; 
    return fpOde;
}

// bcterm: y(0) = 1
double lossODE1(std::vector<double> z, std::vector<double> p){
    double output = 0;

    double bcTerm = pow(getFP(0.0, p)-1, 2);
    for(int i = 0; i < z.size(); i++){
        output += pow(getFPODE1(z[i], p), 2) + bcTerm;
    }
    return output;
}

// PDE: u_t - 1/PI * u_xx = 0
double getFPPDE2(std::vector<double> z, std::vector<double> p){

    double h = 1e-5;
    double d2fpdxx = (getFPVec({z[0]+h, z[1]}, p) - 2*getFPVec(z, p) + getFPVec({z[0]-h, z[1]}, p))/(pow(h,2));
    double dfpdt = (getFPVec({z[0], z[1]+h}, p) - getFPVec({z[0], z[1]-h}, p))/(2*h);

    return dfpdt - 1/M_1_PI * d2fpdxx;
}

// icTerm: u(x,0) = sin(2*pi*x), bcTerm1 = bcTerm2: u(0,t) = u(1,t) = 0
double lossPDE2(std::vector<std::vector<double>> zpde,
                    std::vector<std::vector<double>> zic,
                    std::vector<std::vector<double>> zbc1,
                    std::vector<std::vector<double>> zbc2, 
                    std::vector<double> p){
    double output = 0;
    double pde, icTerm, bcTerm1, bcTerm2;
    for(int i = 0; i < zpde.size(); i++){
        pde += pow(getFPPDE2(zpde[i], p), 2);
    }
    for(int i = 0; i < zic.size(); i++){
        icTerm += pow(getFPVec(zic[i], p) - std::sin(2*M_1_PI*zic[i][0]), 2);
    }
    for(int i = 0; i < zbc1.size(); i++){
        bcTerm1 += pow(getFPVec(zbc1[i], p), 2);
    }
    for(int i = 0; i < zbc2.size(); i++){
        bcTerm2 += pow(getFPVec(zbc2[i], p), 2);
    }
    
    output = pde/zpde.size() + icTerm/zic.size() + bcTerm1/zbc1.size() + bcTerm2/zbc2.size();

    return output;
}