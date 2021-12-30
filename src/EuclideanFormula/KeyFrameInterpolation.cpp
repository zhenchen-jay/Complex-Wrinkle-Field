#include "../../include/EuclideanFormula/KeyFrameInterpolation.h"
#include "../../include/json.hpp"
#include <iostream>
#include <fstream>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/doublearea.h>

using namespace EuclideanFormula;

void KeyFrameInterpolation::convertList2Variable(Eigen::VectorXd& x)
{
	int nverts = _zList[0].size();

	int numFrames = _zList.size() - 2;

	int DOFsPerframe = (2 * nverts + 3 * nverts);

	int DOFs = numFrames * DOFsPerframe;

	x.setZero(DOFs);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			x(i * DOFsPerframe + 2 * j) = _zList[i + 1][j].real();
			x(i * DOFsPerframe + 2 * j + 1) = _zList[i + 1][j].imag();

			x(i * DOFsPerframe + 2 * nverts + 3 * j) = _wList[i + 1](j, 0);
			x(i * DOFsPerframe + 2 * nverts + 3 * j + 1) = _wList[i + 1](j, 1);
			x(i * DOFsPerframe + 2 * nverts + 3 * j + 2) = _wList[i + 1](j, 2);
		}
	}
}

void KeyFrameInterpolation::convertVariable2List(const Eigen::VectorXd& x)
{
	int nverts = _zList[0].size();

	int numFrames = _zList.size() - 2;
	int DOFsPerframe = (2 * nverts + 3 * nverts);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			_zList[i + 1][j] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
			_wList[i + 1](j, 0) = x(i * DOFsPerframe + 2 * nverts + 3 * j);
			_wList[i + 1](j, 1) = x(i * DOFsPerframe + 2 * nverts + 3 * j + 1);
			_wList[i + 1](j, 2) = x(i * DOFsPerframe + 2 * nverts + 3 * j + 2);
		}
	}
}

double KeyFrameInterpolation::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nverts = _zList[0].size();
	
	int numFrames = _zList.size() - 2;
	int DOFsPerframe = (2 * nverts + 3 * nverts);
	int DOFs = numFrames * DOFsPerframe;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}
	for (int i = 0; i < _zList.size() - 1; i++)
	{
		energy += _zdotModel.computeZdotIntegration(_zList[i], _wList[i], _zList[i + 1], _wList[i + 1], deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);

		if (deriv)
		{
			if (i == 0)
				deriv->segment(0, DOFsPerframe) += curDeriv.segment(DOFsPerframe, DOFsPerframe);
			else if (i == _zList.size() - 2)
				deriv->segment((i - 1) * DOFsPerframe, DOFsPerframe) += curDeriv.segment(0, DOFsPerframe);
			else
			{
				deriv->segment((i - 1) * DOFsPerframe, 2 * DOFsPerframe) += curDeriv;
			}

		}

		if (hess)
		{
			for (auto& it : curT)
			{
				if (i == 0)
				{
					if (it.row() >= DOFsPerframe && it.col() >= DOFsPerframe)
						T.push_back({ it.row() - DOFsPerframe, it.col() - DOFsPerframe, it.value() });
				}
				else if (i == _zList.size() - 2)
				{
					if (it.row() < DOFsPerframe && it.col() < DOFsPerframe)
						T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
				}
				else
				{
					T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
				}

			}
			curT.clear();
		}
	}
	if (hess)
	{
		//std::cout << "num of triplets: " << T.size() << std::endl;
		hess->resize(DOFs, DOFs);
		hess->setFromTriplets(T.begin(), T.end());
	}
	return energy;
}

bool KeyFrameInterpolation::save(const std::string& fileName, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
	using json = nlohmann::json;
	json jval;
	jval["mesh_name"] = "mesh.obj";
	jval["num_frame"] = _zList.size() - 2;
	jval["quad_order"] = _quadOrd;

	std::string filePath = fileName;
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	std::string workingFolder = filePath.substr(0, id + 1);

	igl::writeOBJ(workingFolder + "mesh.obj", V, F);
	for (int i = 0; i < _zList.size(); i++)
	{
		std::ofstream zfs(workingFolder + "vertZvals_" + std::to_string(i) + ".txt");
		std::ofstream wfs(workingFolder + "vertOmega_" + std::to_string(i) + ".txt");
		wfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << _wList[i] << std::endl;
		for (int j = 0; j < _zList[i].size(); j++)
		{
			zfs << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << _zList[i][j].real() << " " << _zList[i][j].imag() << std::endl;
		}
	}
	std::ofstream o(workingFolder + "data.json");
	o << std::setw(4) << jval << std::endl;
	std::cout << "save file in: " << workingFolder + "data.json" << std::endl;
	return true;
}

bool KeyFrameInterpolation::load(const std::string& fileName, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	std::cout << "load file in: " << fileName << std::endl;
	using json = nlohmann::json;
	std::ifstream inputJson(fileName);
	if (!inputJson)
	{
		std::cerr << "missing json file in " << fileName << std::endl;
		return false;
	}

	std::string filePath = fileName;
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	std::string workingFolder = filePath.substr(0, id + 1);

	json jval;
	inputJson >> jval;

	std::string meshFile = jval["mesh_name"];
	meshFile = workingFolder + meshFile;
	igl::readOBJ(meshFile, V, F);

	Eigen::VectorXd faceArea;
	igl::doublearea(V, F, faceArea);
	faceArea /= 2;

	int quadOrder = jval["quad_order"];
	int numFrames = jval["num_frame"];

	_wList.resize(numFrames + 2);
	_zList.resize(numFrames + 2);

	MeshConnectivity mesh(F);

	auto loadZandOmega = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			//std::cout << i << std::endl;
			std::ifstream zfs(workingFolder + "vertZvals_" + std::to_string(i) + ".txt");
			_zList[i].resize(V.rows());
			for (int j = 0; j < _zList[i].size(); j++)
			{
				//std::cout << j << std::endl;
				std::string line;
				std::getline(zfs, line);
				//std::cout << line << std::endl;
				std::stringstream ss(line);
				std::string x, y;
				ss >> x;
				ss >> y;
				//std::cout << x << " " << y << std::endl;
				_zList[i][j] = std::complex<double>(std::stod(x), std::stod(y));
			}

			std::ifstream wfs(workingFolder + "vertOmega_" + std::to_string(i) + ".txt");

			_wList[i].resize(V.rows(), 3);
			for (int j = 0; j < _wList[i].rows(); j++)
			{
				std::string line;
				std::getline(wfs, line);
				std::stringstream ss(line);
				std::string x, y, z;
				ss >> x;
				ss >> y;
				ss >> z;
				_wList[i].row(j) << std::stod(x), std::stod(y), std::stod(z);
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)(numFrames + 2), GRAIN_SIZE);
	tbb::parallel_for(rangex, loadZandOmega);

	//for (int i = 0; i < numFrames + 2; i++)
	//{
	//	//std::cout << i << std::endl;
	//	std::ifstream zfs(workingFolder + "zvals_" + std::to_string(i) + ".txt");
	//	_zList[i].resize(V.rows());
	//	for (int j = 0; j < _zList[i].size(); j++)
	//	{
	//		//std::cout << j << std::endl;
	//		std::string line;
	//		std::getline(zfs, line);
	//		//std::cout << line << std::endl;
	//		std::stringstream ss(line);
	//		std::string x, y;
	//		ss >> x;
	//		ss >> y;
	//		//std::cout << x << " " << y << std::endl;
	//		_zList[i][j] = std::complex<double>(std::stod(x), std::stod(y));
	//	}
	//	
	//	std::ifstream wfs(workingFolder + "halfEdgeOmega_" + std::to_string(i) + ".txt");
	//	
	//	_wList[i].resize(mesh.nEdges(), 2);
	//	for (int j = 0; j < _wList[i].rows(); j++)
	//	{
	//		std::string line;
	//		std::getline(wfs, line);
	//		std::stringstream ss(line);
	//		std::string x, y;
	//		ss >> x;
	//		ss >> y;
	//		_wList[i].row(j) << std::stod(x), std::stod(y);
	//	}
	//}
	return true;

}

void KeyFrameInterpolation::testEnergy(Eigen::VectorXd x)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double e = computeEnergy(x, &deriv, &hess, false);
	std::cout << "energy: " << e << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::VectorXd deriv1;
		double e1 = computeEnergy(x + eps * dir, &deriv1, NULL, false);

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}