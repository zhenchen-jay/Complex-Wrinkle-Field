#pragma once
#include "CommonTools.h"

bool mapPlane2Cylinder(Eigen::MatrixXd planeV, Eigen::MatrixXi planeF, Eigen::MatrixXd& cylinderV, Eigen::MatrixXi& cylinderF, Eigen::VectorXi* rest2cylinder);
void generateCylinder(double radius, double height, double triarea);

void generateCylinderWaves(const Eigen::MatrixXd& restV, const MeshConnectivity& restMesh, const Eigen::MatrixXd& cylinderV, const MeshConnectivity& cylinderMesh, double numWaves, double ampMag, Eigen::VectorXd& amp, Eigen::VectorXd& edgeOmega, std::vector<std::complex<double>>& zvals);
void generateCylinderWaves(double radius, double height, double triarea, double numWaves, double ampMag, Eigen::MatrixXd& cylinderV, Eigen::MatrixXi& cylinderF, Eigen::VectorXd& amp, Eigen::VectorXd& dphi, std::vector<std::complex<double>>& zvals);