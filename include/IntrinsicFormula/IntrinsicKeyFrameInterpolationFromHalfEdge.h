#pragma once
#include "ComputeZdotFromHalfEdgeOmega.h"

namespace IntrinsicFormula
{
	class IntrinsicKeyFrameInterpolationFromHalfEdge
	{
	public:
		IntrinsicKeyFrameInterpolationFromHalfEdge() {}
		IntrinsicKeyFrameInterpolationFromHalfEdge(const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const int numFrames, const int quadOrder, const std::vector<std::complex<double>>& startZvals, const Eigen::MatrixXd& startOmega, const std::vector<std::complex<double>>& endZvals, const Eigen::MatrixXd& endOmega)
		{
			_quadOrd = quadOrder;
			double dt = 1.0 / (numFrames + 1);
			_zdotModel = ComputeZdotFromHalfEdgeOmega(mesh, faceArea, quadOrder, dt);
			
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
        IntrinsicKeyFrameInterpolationFromHalfEdge(const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, const int numInterFrames, const int quadOrder, const std::vector<std::vector<std::complex<double>>>& zvalsList, const std::vector<Eigen::MatrixXd>& omegaList)
        {
            _quadOrd = quadOrder;
            _mesh = mesh;
            for(int i = 0; i < zvalsList.size() - 1; i++)
            {
                std::vector<std::complex<double>> startZvals, endZvals;
                Eigen::MatrixXd startOmega, endOmega;

                startZvals = zvalsList[i];
                endZvals = zvalsList[i + 1];

                startOmega = omegaList[i];
                endOmega = omegaList[i + 1];

                double dt = 1.0 / (numInterFrames + 1);
                for(int j = 0; j <= numInterFrames; j++)
                {
                    double t = dt * j;
                    std::vector<std::complex<double>> curZvals = startZvals;
                    Eigen::MatrixXd curOmega = (1 - t) * startOmega + t * endOmega;

                    for(int k = 0; k < startZvals.size(); k++)
                    {
                        curZvals[k] = (1 - t) * startZvals[k] + t * endZvals[k];
                    }

                    _zList.push_back(curZvals);
                    _wList.push_back(curOmega);
                }
            }

            _zList.push_back(zvalsList[zvalsList.size() - 1]);
            _wList.push_back(omegaList[omegaList.size() - 1]);

            int numFrames = _zList.size();
            double dt = 1.0 / (numFrames - 1);
            _zdotModel = ComputeZdotFromHalfEdgeOmega(mesh, faceArea, quadOrder, dt);

        }
		void convertVariable2List(const Eigen::VectorXd& x);
		void convertList2Variable(Eigen::VectorXd& x);

		std::vector<Eigen::MatrixXd> getWList() { return _wList; }
		std::vector<std::vector<std::complex<double>>> getVertValsList() { return _zList; }

        void setBaseMesh(const Eigen::MatrixXd& V) {_triV = V;}
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
					znorm = std::max(znorm, std::abs(x(i * (2 * nverts + 2 * nedges) + 2 * j)));
					znorm = std::max(znorm, std::abs(x(i * (2 * nverts + 2 * nedges) + 2 * j + 1)));
				}
				for (int j = 0; j < nedges; j++)
				{
					wnorm = std::max(wnorm, std::abs(x(i * (2 * nverts + 2 * nedges) + 2 * nverts + 2 * j)));
					wnorm = std::max(wnorm, std::abs(x(i * (2 * nverts + 2 * nedges) + 2 * nverts + 2 * j + 1)));
				}
			}
		}
		void setwzLists(const std::vector<std::vector<std::complex<double>>>& zList, const std::vector<Eigen::MatrixXd>& wList)
		{
			_zList = zList;
			_wList = wList;
		}

        void postProcess(Eigen::VectorXd& x);
        void rescaleZvals(const Eigen::MatrixXd& edgeW, std::vector<std::complex<double>>& zvals);

		double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);
		void testEnergy(Eigen::VectorXd x);

		bool save(const std::string& fileName, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
		bool load(const std::string& fileName, Eigen::MatrixXd& V, Eigen::MatrixXi& F);

	public:	// should be private, when publishing
		ComputeZdotFromHalfEdgeOmega _zdotModel;
	private:
        Eigen::MatrixXd _triV;
		int _quadOrd;
		MeshConnectivity _mesh;
		std::vector<std::vector<std::complex<double>>> _zList;
		std::vector<Eigen::MatrixXd> _wList;
	};
}