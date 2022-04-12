#pragma once
#include "CommonTools.h"

double curlFreeEnergyPerface(const MeshConnectivity& mesh, const Eigen::MatrixXd& w, int faceId);

double amplitudeEnergyWithGivenOmegaPerface(const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::VectorXd& w, int fid, Eigen::Vector3d* deriv, Eigen::Matrix3d* hess);

double amplitudeEnergyWithGivenOmega(const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::VectorXd& w,
	Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT);

void ampExtraction(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& w, const std::vector<std::pair<int, double>>& clampedAmp, Eigen::VectorXd& amp);