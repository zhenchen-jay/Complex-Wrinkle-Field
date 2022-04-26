#pragma once
#include "InterpolateZvals.h"

namespace IntrinsicFormula
{
	class ComputeZdotFromEdgeOmega
			{
			public:
				ComputeZdotFromEdgeOmega() {}
				ComputeZdotFromEdgeOmega(const MeshConnectivity& mesh, const Eigen::VectorXd& faceArea, int quadOrder, double dt = 1)
				{
					_mesh = mesh;
					_faceArea = faceArea;
					_quadpts = buildQuadraturePoints(quadOrder);
					_dt = dt;
				}

				void resetQuadpts(int quadOrder)
				{
					_quadpts = buildQuadraturePoints(quadOrder);
				}

				double computeZdotIntegration(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj = false);

				// test functions (for debug usage)
				void testZdotIntegration(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw);
				void testZdotIntegrationFromQuad(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw,int fid, int qid);
				void testZdotIntegrationPerface(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw, int fid);

				double computeZdotIntegrationFromQuad(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw,int fid, int qid, Eigen::Matrix<double, 18, 1>* deriv, Eigen::Matrix<double, 18, 18>* hess);

				double computeZdotIntegrationPerface(const std::vector<std::complex<double>>& curZvals, const Eigen::VectorXd& curw, const std::vector<std::complex<double>>& nextZvals, const Eigen::VectorXd& nextw, int fid, Eigen::Matrix<double, 18, 1>* deriv, Eigen::Matrix<double, 18, 18>* hess, bool isProj = false);

			private:
				MeshConnectivity _mesh;
				Eigen::VectorXd _faceArea;
				double _dt;
				std::vector<QuadraturePoints> _quadpts;
			};
}
