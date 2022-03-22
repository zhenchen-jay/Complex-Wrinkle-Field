#pragma once

#include"../CommonTools.h"
#include<Eigen/Sparse>
#include<Eigen/Dense>

class WrinkleGluingProcess
{
public:
	WrinkleGluingProcess() {}
	WrinkleGluingProcess(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXi& faceFlag, const std::vector<std::vector<Eigen::VectorXd>>& refAmpList, std::vector<std::vector<Eigen::MatrixXd>>& refOmegaList, int quadOrd);

private:
	void computeCombinedRefAmpList(const std::vector<std::vector<Eigen::VectorXd>>& refAmpList);
	void computeCombinedRefOmegaList(const std::vector<std::vector<Eigen::MatrixXd>>& refOmegaList);
	double curlFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL);
	double curlFreeEnergyPerface(const Eigen::MatrixXd& w, int faceId, Eigen::Matrix<double, 6, 1>* deriv = NULL, Eigen::Matrix<double, 6, 6>* hess = NULL);

private:
	Eigen::MatrixXd _pos;
	MeshConnectivity _mesh;
	Eigen::VectorXi _faceFlag;
	Eigen::VectorXi _vertFlag;
	Eigen::VectorXi _edgeFlag;

	Eigen::MatrixXd _cotMatrixEntries;
	Eigen::VectorXd _faceArea;
	std::vector<Eigen::VectorXd> _combinedRefAmpList;
	std::vector<Eigen::MatrixXd> _combinedRefOmegaList;

};