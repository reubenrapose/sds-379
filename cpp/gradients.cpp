#include <cmath>
#include <vector>

#ifndef LOSS_FUNCTIONS
#define LOSS_FUNCTIONS
    #include "loss_functions.cpp"
#endif

// 2pt central finite difference
std::vector<double> gradientFuncApprox1(std::vector<double> z, std::vector<double> p){
    double h = 1e-5;
    std::vector<double> gradient;
    for(int i = 0; i < p.size(); i++) {
        p[i] = p[i] + h;
        double hp = lossFunctionApprox1(z, p);
        p[i] = p[i] - 2*h;
        double hn = lossFunctionApprox1(z, p);
        p[i] = p[i] + h;
        double pdv = (hp - hn)/(2*h);
        gradient.push_back(pdv);
    }
    
    return gradient;
}

std::vector<double> gradientFuncApprox2(std::vector<double> z, std::vector<double> p){
    double h = 1e-5;
    std::vector<double> gradient;
    for(int i = 0; i < p.size(); i++) {
        p[i] = p[i] + h;
        double hp = lossFunctionApprox2(z, p);
        p[i] = p[i] - 2*h;
        double hn = lossFunctionApprox2(z, p);
        p[i] = p[i] + h;
        double pdv = (hp - hn)/(2*h);
        gradient.push_back(pdv);
    }
    
    return gradient;
}

std::vector<double> gradientFuncApprox3(std::vector<double> z, std::vector<double> p){
    double h = 1e-5;
    std::vector<double> gradient;
    for(int i = 0; i < p.size(); i++) {
        p[i] = p[i] + h;
        double hp = lossFunctionApprox3(z, p);
        p[i] = p[i] - 2*h;
        double hn = lossFunctionApprox3(z, p);
        p[i] = p[i] + h;
        double pdv = (hp - hn)/(2*h);
        gradient.push_back(pdv);
    }
    
    return gradient;
}

// 2pt central finite difference
std::vector<double> gradientODE1(std::vector<double> z, std::vector<double> p){
    double h = 1e-5;
    std::vector<double> gradient;

    for(int i = 0; i < p.size(); i++) {
        p[i] += h;
        double hp = lossODE1(z, p);
        p[i] -= 2*h;
        double hn = lossODE1(z, p);
        p[i] += h;
        double pdv = (hp - hn)/(2*h);
        gradient.push_back(pdv);
    }

    return gradient;
}

// partially analytic gradient, partially finite difference
// specifically for u(x,0) = sin(pi*x)
std::vector<double> gradientPDE(std::vector<std::vector<double>> zpde,
                                std::vector<std::vector<double>> zic,
                                std::vector<std::vector<double>> zbc1,
                                std::vector<std::vector<double>> zbc2,
                                std::vector<double> p){
    double h = 1e-5;
    std::vector<double> gradient;
    for(int pind = 0; pind < p.size(); pind++){
        double dRdpi = 0;
        double dPDE, dIC, dBC1, dBC2;
        for(int i = 0; i < zpde.size(); i++){
            p[pind] += h;
            double hp = getFPPDE(zpde[i], p);
            p[pind] -= 2*h;
            double hn = getFPPDE(zpde[i], p);
            p[pind] += h;
            double dPDEdpi = (hp-hn)/(2*h);

            dPDE += getFPPDE(zpde[i], p) * dPDEdpi;
        }
        dPDE = dPDE * 2/zpde.size();

        for(int i = 0; i < zic.size(); i++){
            p[pind] += h;
            double hp = getFPVec(zic[i], p);
            p[pind] -= 2*h;
            double hn = getFPVec(zic[i], p);
            p[pind] += h;
            double dICdpi = (hp-hn)/(2*h);

            dIC += (getFPVec(zic[i], p) - std::sin(M_PI*zic[i][0])) * (dICdpi - std::cos(M_PI*zic[i][0]));
        }
        dIC = dIC * 2/zic.size();

        for(int i = 0; i < zbc1.size(); i++){
            p[pind] += h;
            double hp = getFPVec(zbc1[i], p);
            p[pind] -= 2*h;
            double hn = getFPVec(zbc1[i], p);
            p[pind] += h;
            double dBC1dpi = (hp-hn)/(2*h);

            dBC1 += getFPVec(zbc1[i], p) * dBC1dpi;
        }
        dBC1 = dBC1 * 2/zbc1.size();

        for(int i = 0; i < zbc2.size(); i++){
            p[pind] += h;
            double hp = getFPVec(zbc2[i], p);
            p[pind] -= 2*h;
            double hn = getFPVec(zbc2[i], p);
            p[pind] += h;
            double dBC2dpi = (hp-hn)/(2*h);

            dBC2 += getFPVec(zbc2[i], p) * dBC2dpi;
        }
        dBC2 = dBC2 * 2/zbc2.size();

        dRdpi = dPDE + dIC + dBC1 + dBC2;

        gradient.push_back(dRdpi);
    }
    return gradient;
}