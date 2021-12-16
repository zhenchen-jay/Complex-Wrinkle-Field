#pragma once
#include <Eigen/Dense>

struct QuadraturePoints
{
    double u;
    double v;
    double weight;
};

std::vector<QuadraturePoints> buildQuadraturePoints(int order); // this is based one the paper: http://lsec.cc.ac.cn/~tcui/myinfo/paper/quad.pdf and the corresponding source codes: http://lsec.cc.ac.cn/phg/download.htm (quad.c)

Eigen::Vector3d computeHatWeight(double u, double v);

Eigen::MatrixXd SPDProjection(Eigen::MatrixXd A);