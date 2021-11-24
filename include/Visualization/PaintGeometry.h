#pragma once

#include <Eigen/Core>
#include "../MeshLib/MeshConnectivity.h"

class PaintGeometry
{
public:
	PaintGeometry()
		: truncateRatio(0), isNormalize(true)
	{}
	~PaintGeometry() {}

	void setTrancateRatio(double r) { truncateRatio = r; }
	void setNormalization(bool flag) { isNormalize = flag; }
	
	Eigen::MatrixXd paintAmplitude(const Eigen::VectorXd& amplitude);
	Eigen::MatrixXd paintPhi(const Eigen::VectorXd& phi);

private:
	double truncateRatio;
	bool isNormalize;
};

