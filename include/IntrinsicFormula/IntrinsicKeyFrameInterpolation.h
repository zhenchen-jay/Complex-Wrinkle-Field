#pragma once
#include "ComputeZdotFromEdgeOmega.h"

namespace IntrinsicFormula
{
	class IntrinsicKeyFrameInterploation
	{
	public:
		IntrinsicKeyFrameInterploation() {}
		IntrinsicKeyFrameInterploation(const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const int numFrames, const int quadOrder, const std::vector<std::complex<double>>& startZvals, const Eigen::VectorXd& startOmega, const std::vector<std::complex<double>>& endZvals, const Eigen::VectorXd& endOmega)
		{
			double dt = 1.0 / (numFrames + 1);
			_zdotModel = ComputeZdotFromEdgeOmega(mesh, faceArea, quadOrder, dt);
			
			_mesh = mesh;

			_wList.resize(numFrames + 2);
			_zList.resize(numFrames + 2);

			_wList[0] = startOmega;
			_wList[numFrames + 1] = endOmega;

			_zList[0] = startZvals;
			_zList[numFrames + 1] = endZvals;

			// linear interpolate in between
			for (int i = 1; i <= numFrames; i++)
			{
				double t = dt * i;
				_wList[i] = (1 - t) * startOmega + t * endOmega;
				_zList[i] = startZvals;

				for (int j = 0; j < _zList[i].size(); j++)
					_zList[i][j] = (1 - t) * startZvals[j] + t * endZvals[j];
			}

		}

		void convertVariable2List(const Eigen::VectorXd& x);
		void convertList2Variable(Eigen::VectorXd& x);

		std::vector<Eigen::VectorXd> getWList() { return _wList; }
		std::vector<std::vector<std::complex<double>>> getVertValsList() { return _zList; }

		void getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm)
		{
			int nverts = _zList[0].size();
			int nedges = _wList[0].rows();

			int numFrames = _zList.size() - 2;

			znorm = 0;
			wnorm = 0;

			for (int i = 0; i < numFrames; i++)
			{
				for (int j = 0; j < nverts; j++)
				{
					znorm = std::max(znorm, std::abs(x(i * (2 * nverts + nedges) + 2 * j)));
					znorm = std::max(znorm, std::abs(x(i * (2 * nverts + nedges) + 2 * j + 1)));
				}
				for (int j = 0; j < nedges; j++)
				{
					wnorm = std::max(wnorm, std::abs(x(i * (2 * nverts + nedges) + 2 * nverts + j)));
				}
			}
		}
		void setwzLists(const std::vector<std::vector<std::complex<double>>>& zList, const std::vector<Eigen::VectorXd>& wList)
		{
			_zList = zList;
			_wList = wList;
		}


		double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);
		void testEnergy(Eigen::VectorXd x);

	public:
		ComputeZdotFromEdgeOmega _zdotModel;
	private:
		
		MeshConnectivity _mesh;
		std::vector<std::vector<std::complex<double>>> _zList;
		std::vector<Eigen::VectorXd> _wList;
	};
}