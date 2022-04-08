#pragma once
#include "../CommonTools.h"
#include "KnoppelStripePatternEdgeOmega.h"
#include "ComputeZdotFromEdgeOmega.h"

namespace IntrinsicFormula {
    class KnoppelDrivenEdgeModel {
    public:
        KnoppelDrivenEdgeModel() {}
        KnoppelDrivenEdgeModel(const MeshConnectivity mesh, const Eigen::VectorXd& faceArea, const Eigen::MatrixXd& cotEntries, const std::vector<Eigen::VectorXd> &refEdgeOmegaList, const std::vector<Eigen::VectorXd> &refAmpList, const std::vector<std::complex<double>> &initZvals, const std::vector<std::complex<double>> &tarZvals, const Eigen::VectorXd &initEdgeOmega, const Eigen::VectorXd &tarEdgeOmega, int numFrames = 20, double spatialRatio = 1.0, double quadOrder = 4) : _numFrames(numFrames), _spatialRatio(spatialRatio)
        {
            _mesh = mesh;
            _refAmpList = refAmpList;
            _refEdgeOmegaList = refEdgeOmegaList;
            
            _edgeOmegaList = _refEdgeOmegaList;
            _zvalsList.resize(_numFrames + 2);

            _zvalsList[0] = initZvals;
            _zvalsList[_numFrames + 1] = tarZvals;

            double dt = 1.0 / (_numFrames + 1);

            for(int i = 1; i <= _numFrames; i++)
            {
                double t = i * dt;

                _zvalsList[i] = tarZvals;

                for(int j = 0; j < tarZvals.size(); j++)
                {
                    _zvalsList[i][j] = (1 - t) * initZvals[j] + t * tarZvals[j];
                }
            }
            _zdotModel = ComputeZdotFromEdgeOmega(mesh, faceArea, quadOrder, dt);
            _quadOrder = quadOrder;
            _faceArea = faceArea;
            _cotEntries = cotEntries;

        }

        void convertVariable2List(const Eigen::VectorXd& x);
        void convertList2Variable(Eigen::VectorXd& x);

        std::vector<Eigen::VectorXd> getWList() { return _edgeOmegaList; }
        std::vector<std::vector<std::complex<double>>> getVertValsList() { return _zvalsList; }

        void setwzLists(std::vector<std::vector<std::complex<double>>> &zList, std::vector<Eigen::VectorXd> &wList)
        {
            _zvalsList = zList;
            _edgeOmegaList = wList;
        }

        void getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm)
        {
            int nverts = _zvalsList[0].size();
            int nedges = _edgeOmegaList[0].rows();

            int numFrames = _zvalsList.size() - 2;

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

        double computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv = NULL, Eigen::SparseMatrix<double>* hess = NULL, bool isProj = false);
        void testEnergy(Eigen::VectorXd x);

    public:	// should be private, when publishing
        ComputeZdotFromEdgeOmega _zdotModel;

    private:
        std::vector<Eigen::VectorXd> _refEdgeOmegaList;
        std::vector<Eigen::VectorXd> _refAmpList;
        MeshConnectivity _mesh;

        std::vector<Eigen::VectorXd> _edgeOmegaList;
        std::vector<std::vector<std::complex<double>>> _zvalsList;
        int _numFrames;
        double _spatialRatio;

        Eigen::VectorXd _faceArea;
        Eigen::MatrixXd _cotEntries;
        int _quadOrder;
    };
}