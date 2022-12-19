#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <set>
/*
 * I implemented the rounding up algorithm mention in the paper "Fine Wrinkling on Coarsely-Meshed Thin Shells". (https://github.com/zhenchen-jay/WrinkledTensionFields)
 */

namespace TFWAlg
{
	void firstFoundForms(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, std::vector<Eigen::Matrix2d> &abars);
	void findCuts(const Eigen::MatrixXi &F, std::vector<std::vector<int> > &cuts);
	void cutMesh(const Eigen::MatrixXi &F,
			// list of cuts, each of which is a list (in order) of vertex indices of one cut.
			// Cuts can be closed loops (in which case the last vertex index should equal the
			// first) or open (in which case the two endpoint vertices should be distinct).
			// Multiple cuts can cross but there may be strange behavior if cuts share endpoint
			// vertices, or are non-edge-disjoint.
				 const std::vector<std::vector<int> > &cuts,
			// new vertices and faces
			// **DO NOT ALIAS V OR F!**
				 Eigen::MatrixXi &newF
	);
	void reindex(Eigen::MatrixXi &F);

	void ComisoWrapper(const Eigen::SparseMatrix<double> &constraints,
					   const Eigen::SparseMatrix<double> &A,
					   Eigen::VectorXd &result,
					   const Eigen::VectorXd &rhs,
					   const Eigen::VectorXi &toRound,
					   double reg);

	void findSharpCorners(const Eigen::MatrixXd& uncutV, const Eigen::MatrixXi& uncutF, std::set<int>& cornerVerts);

	void roundPhiFromEdgeOmega(const Eigen::MatrixXd& V,
							   const Eigen::MatrixXi& F,
							   const std::vector<Eigen::Matrix2d>& abars,
							   const Eigen::VectorXd& amp,
							   const Eigen::VectorXd& dPhi, // |E| vector of phi jump
							   Eigen::VectorXd& phi,
							   Eigen::MatrixXd& seamedV,
							   Eigen::MatrixXi& seamedF,
							   Eigen::VectorXd& seamedAmp,
							   Eigen::VectorXd& seamedPhi,
							   std::set<int> &problemFaces);

	void wrinkledMeshUpsamplingUncut(const Eigen::MatrixXd &uncutV, const Eigen::MatrixXi &uncutF,
									 const Eigen::MatrixXd &restV, const Eigen::MatrixXi &restF,
									 const Eigen::MatrixXd &cutV, const Eigen::MatrixXi &cutF,
									 const Eigen::VectorXd &cutAmplitude, const Eigen::VectorXd &cutPhi,
									 const std::set<int> &noPhiFaces,
									 const std::set<int> &clampedVerts,
									 Eigen::MatrixXd *wrinkledV, Eigen::MatrixXi *wrinkledF,
									 Eigen::MatrixXd *upsampledTFTV, Eigen::MatrixXi *upsampledTFTF,
									 Eigen::MatrixXd *soupPhiV, Eigen::MatrixXi *soupPhiF,
									 Eigen::MatrixXd *soupProblemV, Eigen::MatrixXi *soupProblemF,
									 Eigen::VectorXd *upsampledAmp, Eigen::VectorXd *soupPhi, Eigen::VectorXd *upsampledPhi,
									 int numSubdivs = 0,
									 double ampScaling = 1.0,
									 bool isUseV2Term = true);

	void getTFWSurfacePerframe(const Eigen::MatrixXd& baseV, const Eigen::MatrixXi& baseF,
							   const Eigen::VectorXd& amp, const Eigen::VectorXd& omega,
							   Eigen::MatrixXd& wrinkledV, Eigen::MatrixXi& wrinkledF,
							   Eigen::MatrixXd* upsampledV, Eigen::MatrixXi* upsampledF,
							   Eigen::MatrixXd* soupPhiV, Eigen::MatrixXi* soupPhiF,
							   Eigen::MatrixXd* soupProblemV, Eigen::MatrixXi* soupProblemF,
							   Eigen::VectorXd& upsampledAmp, Eigen::VectorXd* soupPhi, Eigen::VectorXd& upsampledPhi,
							   int numSubdivs = 0, double ampScaling = 1.0, bool isUseV2Term = true, bool isFixedBnd = false
							   );

	void getTFWSurfaceSequence(const Eigen::MatrixXd& baseV, const Eigen::MatrixXi& baseF,
						const std::vector<Eigen::VectorXd>& ampList, const std::vector<Eigen::VectorXd>& omegaList,
						std::vector<Eigen::MatrixXd>& wrinkledVList, std::vector<Eigen::MatrixXi>& wrinkledFList,
						std::vector<Eigen::MatrixXd>& upsampledVList, std::vector<Eigen::MatrixXi>& upsampledFList,
						std::vector<Eigen::MatrixXd>* soupPhiVList, std::vector<Eigen::MatrixXi>* soupPhiFList,
						std::vector<Eigen::MatrixXd>* soupProblemVList, std::vector<Eigen::MatrixXi>* soupProblemFList,
						std::vector<Eigen::VectorXd>& upsampledAmpList, std::vector<Eigen::VectorXd>* soupPhiList, std::vector<Eigen::VectorXd>& upsampledPhiList,
						int numSubdivs = 0, double ampScaling = 1.0, bool isUseV2Term = true, bool isFixedBnd = false
						);
}