#pragma once
#include "CommonTools.h"

bool mapPlane2Cylinder(Eigen::MatrixXd planeV, Eigen::MatrixXi planeF, Eigen::MatrixXd& cylinderV, Eigen::MatrixXi& cylinderF, Eigen::VectorXi* rest2cylinder);
void generateCylinder(double radius, double height, double triarea);
void generateRectangle(double len, double width, double triarea, Eigen::MatrixXd& triV, Eigen::MatrixXi& triF);

void generateCylinderWaves(const Eigen::MatrixXd& restV, const MeshConnectivity& restMesh, const Eigen::MatrixXd& cylinderV, const MeshConnectivity& cylinderMesh, double numWaves, double ampMag, Eigen::VectorXd& amp, Eigen::VectorXd& edgeOmega, std::vector<std::complex<double>>& zvals, Eigen::VectorXd *restAmp = NULL, Eigen::VectorXd* restEdgeOmega = NULL, std::vector<std::complex<double>>* restZvals = NULL);
void generateCylinderWaves(double radius, double height, double triarea, double numWaves, double ampMag, Eigen::MatrixXd& cylinderV, Eigen::MatrixXi& cylinderF, Eigen::VectorXd& amp, Eigen::VectorXd& dphi, std::vector<std::complex<double>>& zvals, Eigen::MatrixXd* restV = NULL, Eigen::MatrixXi* restF = NULL, Eigen::VectorXd* restAmp = NULL, Eigen::VectorXd* restEdgeOmega = NULL, std::vector<std::complex<double>>* restZvals = NULL);

void generateWhirlPool(const Eigen::MatrixXd& triV, double centerx, double centery, Eigen::MatrixXd& w, std::vector<std::complex<double>>& z, int pow = 1, std::vector<Eigen::Vector2cd> *gradZ = NULL);
void generatePlaneWave(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, const Eigen::Vector2d& w, Eigen::VectorXd& edgeOmega, std::vector<std::complex<double>>& vertZvals, std::vector<Eigen::Vector2cd>* gradVertZvals = NULL);