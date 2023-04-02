#include "../../include/IntrinsicFormula/WrinkleEditingBaseMeshSeqCWF.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>
#include <unordered_set>

using namespace IntrinsicFormula;

WrinkleEditingBaseMeshSeqCWF::WrinkleEditingBaseMeshSeqCWF(const std::vector<Eigen::MatrixXd>& pos, const MeshConnectivity& mesh, int quadOrd, double spatialAmpRatio, double spatialKnoppelRatio)
{
	_posList = pos;
	_mesh = mesh;
	_quadOrd = quadOrd;
	_spatialAmpRatio = spatialAmpRatio;
	_spatialKnoppelRatio = spatialKnoppelRatio;

	std::cout << "number of frames: " << _posList.size() << std::endl;

	if (_posList.size() < 2)
	{
		std::cerr << "At least provide two key frames!" << std::endl;
		exit(EXIT_FAILURE);
	}

	int nframes = _posList.size();

	int nverts = pos[0].rows();
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	buildVertexNeighboringInfo(_mesh, nverts, _vertNeiEdges, _vertNeiFaces);

	_vertAreaList.resize(nframes);
	_edgeAreaList.resize(nframes);
	_faceAreaList.resize(nframes);
	_edgeCotCoeffsList.resize(nframes);
	_faceVertMetricsList.resize(nframes);

	for (int n = 0; n < nframes; n++)
	{
		_vertAreaList[n] = getVertArea(_posList[n], _mesh);
		_edgeAreaList[n] = getEdgeArea(_posList[n], _mesh);
		_faceAreaList[n] = getFaceArea(_posList[n], _mesh);

		Eigen::MatrixXd cotMatrixEntries;

		igl::cotmatrix_entries(_posList[n], mesh.faces(), cotMatrixEntries);
		_edgeCotCoeffsList[n].setZero(nedges);

		for (int i = 0; i < nfaces; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				int eid = _mesh.faceEdge(i, j);
				int vid = _mesh.faceVertex(i, j);
				_edgeCotCoeffsList[n](eid) += cotMatrixEntries(i, j);
			}
		}

		_faceVertMetricsList[n].resize(nfaces);
		for (int i = 0; i < nfaces; i++)
		{
			_faceVertMetricsList[n][i].resize(3);
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(i, j);
				int vidj = _mesh.faceVertex(i, (j + 1) % 3);
				int vidk = _mesh.faceVertex(i, (j + 2) % 3);

				Eigen::Vector3d e0 = _posList[n].row(vidj) - _posList[n].row(vid);
				Eigen::Vector3d e1 = _posList[n].row(vidk) - _posList[n].row(vid);

				Eigen::Matrix2d I;
				I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
				_faceVertMetricsList[n][i][j] = I.inverse();
			}
		}
	}
	
}

void WrinkleEditingBaseMeshSeqCWF::adjustOmegaForConsistency(const std::vector<std::complex<double>>& zvals, const Eigen::VectorXd& omega, Eigen::VectorXd& newOmega, Eigen::VectorXd& deltaOmega, Eigen::VectorXi* edgeFlags)
{
	int nedges = _mesh.nEdges();
	deltaOmega.setZero(nedges);
	newOmega = omega;
	for (int i = 0; i < nedges; i++)
	{
		if (edgeFlags)
		{
			if ((*edgeFlags)(i) == 1)	// fixed omega
				continue;
		}
		int v0 = _mesh.edgeVertex(i, 0);
		int v1 = _mesh.edgeVertex(i, 1);

		double theta1 = std::arg(zvals[v1]);
		double theta0 = std::arg(zvals[v0]);

		double dtheta = theta1 - theta0;
		double c = (omega(i) - dtheta) / 2.0 / M_PI;
		int k = std::floor(c + 0.5);
		newOmega(i) = dtheta + 2 * k * M_PI;
		deltaOmega(i) = omega(i) - newOmega(i);
	}
}

void WrinkleEditingBaseMeshSeqCWF::vecFieldLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag)
{
	vecList.resize(numFrames + 2);
	vecList[0] = initVec;
	vecList[numFrames + 1] = tarVec;

	double dt = 1. / (numFrames + 1);

	for (int j = 1; j < numFrames + 1; j++)
	{
		double t = dt * j;
		vecList[j] = (1 - t) * initVec + t * tarVec;

		if (edgeFlag)
		{
			for (int v = 0; v < initVec.rows(); v++)
			{
				if ((*edgeFlag)(v))
					vecList[j][v] = initVec[v];
			}
		}
	}
}

void WrinkleEditingBaseMeshSeqCWF::vecFieldSLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag)
{
	vecList.resize(numFrames + 2);
	vecList[0] = initVec;
	vecList[numFrames + 1] = tarVec;

	Eigen::MatrixXd initFaceVec, tarFaceVec;
	initFaceVec = intrinsicEdgeVec2FaceVec(initVec, _posList[0], _mesh);
	tarFaceVec = intrinsicEdgeVec2FaceVec(tarVec, _posList[numFrames + 1], _mesh);
	int nfaces = _mesh.nFaces();
	double dt = 1. / (numFrames + 1);

	for (int j = 1; j < numFrames + 1; j++)
	{
		double t = dt * j;
		Eigen::MatrixXd faceVec = initFaceVec;


		for (int f = 0; f < faceVec.rows(); f++)
		{
			Eigen::RowVector3d vec = Eigen::RowVector3d::Zero();

			if (initFaceVec.row(f).norm() == 0)
				vec = t * tarFaceVec.row(f);
			else if (tarFaceVec.row(f).norm() == 0)
				vec = (1 - t) * initFaceVec.row(f);
			else
			{
				int vid0 = _mesh.faceVertex(f, 0);
				int vid1 = _mesh.faceVertex(f, 1);
				int vid2 = _mesh.faceVertex(f, 2);


				Eigen::RowVector3d e0 = _posList[j].row(vid1) - _posList[j].row(vid0);
				Eigen::RowVector3d e1 = _posList[j].row(vid2) - _posList[j].row(vid0);
				Eigen::RowVector3d faceNormal = e0.cross(e1);
				if (faceNormal.norm() < 1e-12)
				{
					vec = (1 - t) * initFaceVec.row(f) + t * tarFaceVec.row(f);
					continue;
				}

				Eigen::RowVector3d initE0 = _posList[0].row(vid1) - _posList[0].row(vid0);
				Eigen::RowVector3d tarE0 = _posList[numFrames + 1].row(vid1) - _posList[numFrames + 1].row(vid0);

				double cos0 = initFaceVec.row(f).segment<3>(0).dot(initE0) / initFaceVec.row(f).norm() / initE0.norm();
				cos0 = std::clamp(cos0, -1., 1.);   // avoid numerical issues
				double phi0 = std::acos(cos0);

				double cos1 = tarFaceVec.row(f).segment<3>(0).dot(tarE0) / tarFaceVec.row(f).norm() / tarE0.norm();
				cos1 = std::clamp(cos1, -1., 1.);   // avoid numerical issues
				double phi1 = std::acos(cos1);

				double phi = (1 - t) * phi0 + t * phi1;

				faceNormal.normalize();
				vec = rotateSingleVector(e0.transpose(), faceNormal.transpose(), phi).transpose();

				vec = vec / vec.norm();

				// lerp on mag square
				double mag = (1 - t) * initFaceVec.row(f).norm() + t * tarFaceVec.row(f).norm();
				vec *= mag > 0 ? mag : 0;
			}
			faceVec.row(f) = vec;

		}
		vecList[j] = faceVec2IntrinsicEdgeVec(faceVec, _posList[j], _mesh);

		if (edgeFlag)
		{
			for (int e = 0; e < initVec.rows(); e++)
			{
				if ((*edgeFlag)(e))
					vecList[j][e] = initVec[e];
			}
		}

	}
}

void WrinkleEditingBaseMeshSeqCWF::ampFieldLERP(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, std::vector<Eigen::VectorXd>& ampList, int numFrames, Eigen::VectorXi* vertFlag)
{
	ampList.resize(numFrames + 2);
	ampList[0] = initAmp;
	ampList[numFrames + 1] = tarAmp;

	double dt = 1. / (numFrames + 1);

	for (int j = 1; j < numFrames + 1; j++)
	{
		double t = dt * j;
		ampList[j] = (1 - t) * initAmp + t * tarAmp;

		if (vertFlag)
		{
			for (int v = 0; v < initAmp.rows(); v++)
			{
				if ((*vertFlag)(v))
					ampList[j][v] = initAmp[v];
			}
		}
	}
}

void WrinkleEditingBaseMeshSeqCWF::initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, const std::vector<std::complex<double>>& tarZvals, const Eigen::VectorXd& tarOmega, bool applyAdj)
{
	int numFrames = _posList.size() - 2;
	int nverts = _posList[0].rows();

	// we use our new formula to initialize everything
	_combinedRefAmpList.resize(numFrames + 2);
	_combinedRefOmegaList.resize(numFrames + 2);
	_edgeOmegaList.resize(numFrames + 2);
	_deltaOmegaList.resize(numFrames + 2, Eigen::VectorXd::Zero(_mesh.nEdges()));
	_zvalsList.resize(numFrames + 2);
	_unitZvalsList.resize(numFrames + 2);
	_ampTimesOmegaSq.resize(numFrames + 2);
	_ampTimesDeltaOmegaSq.resize(numFrames + 2, Eigen::VectorXd::Zero(_mesh.nEdges()));

	Eigen::VectorXd initAmp, tarAmp;
	initAmp.setZero(nverts);
	tarAmp.setZero(nverts);

	_edgeOmegaList[0] = initOmega;
	_edgeOmegaList[numFrames + 1] = tarOmega;


	_zvalsList[0] = initZvals;
	_zvalsList[numFrames + 1] = tarZvals;

	_unitZvalsList[0] = _zvalsList[0];
	_unitZvalsList[numFrames + 1] = _zvalsList[numFrames + 1];

	for (int i = 0; i < initAmp.rows(); i++)
	{
		initAmp(i) = std::abs(initZvals[i]);
		tarAmp(i) = std::abs(tarZvals[i]);

		_unitZvalsList[0][i] = initAmp[i] != 0 ? _unitZvalsList[0][i] / initAmp[i] : _zvalsList[0][i];
		_unitZvalsList[numFrames + 1][i] = tarAmp[i] != 0 ? _unitZvalsList[numFrames + 1][i] / tarAmp[i] : _zvalsList[numFrames + 1][i];
	}

	_combinedRefAmpList[0] = initAmp;
	_combinedRefAmpList[numFrames + 1] = tarAmp;

	double knoppel0 = KnoppelEdgeEnergy(_mesh, _edgeOmegaList[0], _edgeAreaList[0], initZvals, NULL, NULL);
	double knoppel1 = KnoppelEdgeEnergy(_mesh, _edgeOmegaList[numFrames + 1], _edgeAreaList[numFrames + 1], tarZvals, NULL, NULL);

	std::cout << "init knoppel: " << knoppel0 << ", tar knoppel: " << knoppel1 << std::endl;

	if (applyAdj)
	{
		// we first adjust the input, to make sure that (z, w) are consistent
		adjustOmegaForConsistency(initZvals, initOmega, _edgeOmegaList[0], _deltaOmegaList[0]);
		adjustOmegaForConsistency(tarZvals, tarOmega, _edgeOmegaList[numFrames + 1], _deltaOmegaList[numFrames + 1]);
	}

	std::cout << "after adjust, knoppel energy is: " << std::endl;
	knoppel0 = KnoppelEdgeEnergy(_mesh, _edgeOmegaList[0], _edgeAreaList[0], initZvals, NULL, NULL);
	knoppel1 = KnoppelEdgeEnergy(_mesh, _edgeOmegaList[numFrames + 1], _edgeAreaList[numFrames + 1], tarZvals, NULL, NULL);

	std::cout << "init knoppel: " << knoppel0 << ", tar knoppel: " << knoppel1 << std::endl;

	computeAmpOmegaSq(_posList[0], _mesh, _combinedRefAmpList[0], _edgeOmegaList[0], _ampTimesOmegaSq[0]);
	computeAmpOmegaSq(_posList[numFrames + 1], _mesh, _combinedRefAmpList[numFrames + 1], _edgeOmegaList[numFrames + 1], _ampTimesOmegaSq[numFrames + 1]);

	computeAmpOmegaSq(_posList[0], _mesh, _combinedRefAmpList[0], _deltaOmegaList[0], _ampTimesDeltaOmegaSq[0]);
	computeAmpOmegaSq(_posList[numFrames + 1], _mesh, _combinedRefAmpList[numFrames + 1], _deltaOmegaList[numFrames + 1], _ampTimesDeltaOmegaSq[numFrames + 1]);

	double dt = 1.0 / (numFrames + 1);


	for (int i = 1; i < numFrames + 1; i++)
	{
		double t = i * dt;
		_ampTimesOmegaSq[i] = ampTimeOmegaSqInitialization(_ampTimesOmegaSq[0], _ampTimesOmegaSq[numFrames + 1], t);
		_combinedRefAmpList[i] = ampInitialization(_combinedRefAmpList[0], _combinedRefAmpList[numFrames + 1], t);
		_edgeOmegaList[i] = omegaInitialization(_edgeOmegaList[0], _edgeOmegaList[numFrames + 1], _combinedRefAmpList[0], _combinedRefAmpList[numFrames + 1], i);

		_deltaOmegaList[i] = (1 - t) * _deltaOmegaList[0] + t * _deltaOmegaList[numFrames + 1];

		// zvals
		_zvalsList[i] = tarZvals;
		_unitZvalsList[i] = _unitZvalsList[numFrames + 1];
		for (int j = 0; j < tarZvals.size(); j++)
		{
			_zvalsList[i][j] = (1 - t) * initZvals[j] + t * tarZvals[j];
			_unitZvalsList[i][j] = (1 - t) * _unitZvalsList[0][j] + t * _unitZvalsList[numFrames + 1][j];
		}
	}

	std::cout << "omega list initialization finished!" << std::endl;

	std::vector<Eigen::VectorXd> _ampTimesCombinedOmegaSq(numFrames + 2);
	std::vector<Eigen::VectorXd> _ampTest(numFrames + 2);
	_combinedRefOmegaList[0] = initOmega;
	_combinedRefOmegaList[numFrames + 1] = tarOmega;
	_ampTest = _combinedRefAmpList;

	computeAmpOmegaSq(_posList[0], _mesh, _ampTest[0], _combinedRefOmegaList[0], _ampTimesCombinedOmegaSq[0]);
	computeAmpOmegaSq(_posList[numFrames + 1], _mesh, _ampTest[numFrames + 1], _combinedRefOmegaList[numFrames + 1], _ampTimesCombinedOmegaSq[numFrames + 1]);

	for (int i = 0; i < numFrames + 1; i++)
	{
		double t = i * dt;
		_ampTimesCombinedOmegaSq[i] = ampTimeOmegaSqInitialization(_ampTimesCombinedOmegaSq[0], _ampTimesCombinedOmegaSq[numFrames + 1], t);
		_ampTest[i] = ampInitialization(_ampTest[0], _ampTest[numFrames + 1], t);
		_combinedRefOmegaList[i] = omegaInitialization(_combinedRefOmegaList[0], _combinedRefOmegaList[numFrames + 1], _combinedRefAmpList[0], _combinedRefAmpList[numFrames + 1], t);
	}
}

void WrinkleEditingBaseMeshSeqCWF::initialization(const std::vector<std::vector<std::complex<double>>>& zList, const std::vector<Eigen::VectorXd>& omegaList, const std::vector<Eigen::VectorXd>& refAmpList, const std::vector<Eigen::VectorXd>& refOmegaList, bool applyAdj)
{
	_zvalsList = zList;
	_edgeOmegaList = omegaList;
	_combinedRefAmpList = refAmpList;
	_combinedRefOmegaList = refOmegaList;
}

Eigen::VectorXd WrinkleEditingBaseMeshSeqCWF::ampTimeOmegaSqInitialization(const Eigen::VectorXd& initAmpOmegaSq, const Eigen::VectorXd& tarAmpOmegaSq, double t)
{
return (1 - t) * initAmpOmegaSq + t * tarAmpOmegaSq;
}

Eigen::VectorXd WrinkleEditingBaseMeshSeqCWF::ampInitialization(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, double t)
{
	return (1 - t) * initAmp + t * tarAmp;
}

Eigen::VectorXd WrinkleEditingBaseMeshSeqCWF::omegaInitialization(const Eigen::VectorXd& initOmega, const Eigen::VectorXd& tarOmega, const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, int frameId)
{
	int nfaces = _mesh.nFaces();
	int nedges = _mesh.nEdges();
	int totalFrames = _posList.size();
	double t = frameId * 1.0 / (totalFrames - 1);

	Eigen::VectorXd curOmega = Eigen::VectorXd::Zero(nedges);

	for (int fid = 0; fid < nfaces; fid++)
	{
		for (int vInF = 0; vInF < 3; vInF++)
		{
			int vid = _mesh.faceVertex(fid, vInF);

			Eigen::Vector3d faceOmega;
			int eid0 = _mesh.faceEdge(fid, (vInF + 1) % 3);
			int eid1 = _mesh.faceEdge(fid, (vInF + 2) % 3);


			Eigen::RowVector3d r0 = _posList[frameId].row(_mesh.faceVertex(fid, (vInF + 2) % 3)) - _posList[frameId].row(vid);
			Eigen::RowVector3d r1 = _posList[frameId].row(_mesh.faceVertex(fid, (vInF + 1) % 3)) - _posList[frameId].row(vid);

			Eigen::RowVector3d initR0 = _posList[0].row(_mesh.faceVertex(fid, (vInF + 2) % 3)) - _posList[0].row(vid);
			Eigen::RowVector3d initR1 = _posList[0].row(_mesh.faceVertex(fid, (vInF + 1) % 3)) - _posList[0].row(vid);

			Eigen::RowVector3d tarR0 = _posList[totalFrames - 1].row(_mesh.faceVertex(fid, (vInF + 2) % 3)) - _posList[totalFrames - 1].row(vid);
			Eigen::RowVector3d tarR1 = _posList[totalFrames - 1].row(_mesh.faceVertex(fid, (vInF + 1) % 3)) - _posList[totalFrames - 1].row(vid);


			Eigen::RowVector3d initFaceNormal = initR0.cross(initR1);
			Eigen::RowVector3d tarFaceNormal = tarR0.cross(tarR1);

			int flag0 = 1, flag1 = 1;

			if (_mesh.edgeVertex(eid0, 0) == vid)
			{
				flag0 = 1;
			}
			else
			{
				flag0 = -1;
			}


			if (_mesh.edgeVertex(eid1, 0) == vid)
			{
				flag1 = 1;
			}
			else
			{
				flag1 = -1;
			}

			Eigen::Matrix2d initI, tarI;
			initI << initR0.dot(initR0), initR0.dot(initR1), initR1.dot(initR0), initR1.dot(initR1);
			tarI << tarR0.dot(tarR0), tarR0.dot(tarR1), tarR1.dot(tarR0), tarR1.dot(tarR1);

			Eigen::Vector2d rhs0, rhs1;
			rhs0 << flag0 * initOmega(eid0), flag1* initOmega(eid1);
			rhs1 << flag0 * tarOmega(eid0), flag1* tarOmega(eid1);

			Eigen::Vector2d sol0 = initI.inverse() * rhs0;
			Eigen::Vector2d sol1 = tarI.inverse() * rhs1;

			Eigen::Vector3d initFaceOmega = sol0[0] * initR0 + sol0[1] * initR1;
			Eigen::Vector3d tarFaceOmega = sol1[0] * tarR0 + sol1[1] * tarR1;

			if ((r0.cross(r1)).norm() < 1e-10 || initFaceOmega.norm() < 1e-10 || tarFaceOmega.norm() < 1e-10) // to skinny triangles, or 0-cases
			{
				faceOmega = (1 - t) * initFaceOmega + t * tarFaceOmega;
			}
			else
			{
				double w0Sq = initFaceOmega.squaredNorm();
				double w1Sq = tarFaceOmega.squaredNorm();

				double a0 = initAmp[vid];
				double a1 = tarAmp[vid];
				double wSq = 0;

				if (a0 < 1e-10 || a1 < 1e-10)
					wSq = (1 - t) * w0Sq + t * w1Sq;
				else
					wSq = (1 - t) * a0 / ((1 - t) * a0 + t * a1) * w0Sq + t * a1 / ((1 - t) * a0 + t * a1) * w1Sq;

				Eigen::RowVector3d rotAxis = r0.cross(r1);

				Eigen::RowVector3d refEdgeInit = initR0, refEdgeTar = tarR0, refEdge = r0;
				if (std::abs(initFaceOmega.cross(initR0).norm()) < 1e-10)		// r0 is not for init
				{
					if (std::abs(initFaceOmega.cross(initR1).norm()) < 1e-10)	// r1 is also not for init: skinny triangle
					{
						faceOmega = (1 - t) * initFaceOmega + t * tarFaceOmega;
						continue;
					}
					else
					{
						if (std::abs(tarFaceOmega.cross(tarR1).norm()) < 1e-10)	// r1 is good for init, but not for tar
						{
							refEdgeInit = (initR0 + initR1) / 2;
							refEdgeTar = (tarR0 + tarR1) / 2;
							refEdge = (r0 + r1) / 2;
						}
						else							// r1 is a good choice
						{
							refEdgeInit = initR1;
							refEdgeTar = tarR1;
							refEdge = r1;
						}
					}
				}
				else	// r0 is good for init
				{
					if (std::abs(tarFaceOmega.cross(tarR0).norm()) < 1e-10)	// r0 is not good for tar
					{
						if (std::abs(tarFaceOmega.cross(tarR1).norm()) < 1e-10) // r1 is not good for tar: skinny triangle
						{
							faceOmega = (1 - t) * initFaceOmega + t * tarFaceOmega;
							continue;
						}
						else // r1 is good for tar, r0 is not good for tar, and triangle is not skinny. In this case (r0 + r1) / 2 is always good
						{
							if (std::abs(initFaceOmega.cross(initR1).norm()) < 1e-10)	// r1 is not good for init, and triangle is not skinny. In this case (r0 + r1) / 2 is always good
							{
								refEdgeInit = (initR0 + initR1) / 2;
								refEdgeTar = (tarR0 + tarR1) / 2;
								refEdge = (r0 + r1) / 2;
							}
							else
							{
								refEdgeInit = initR1;
								refEdgeTar = tarR1;
								refEdge = r1;
							}
						}
					}
					else
					{
						refEdgeInit = initR0;
						refEdgeTar = tarR0;
						refEdge = r0;
					}
				}
				
				double cos0 = initFaceOmega.dot(refEdgeInit) / initFaceOmega.norm() / refEdgeInit.norm();
				cos0 = std::clamp(cos0, -1., 1.);   // avoid numerical issues
				double phi0 = std::acos(cos0);
				int sign0 = 1;
				if (refEdgeInit.cross(initFaceOmega).dot(initFaceNormal) < 0)
					sign0 = -1;

				double cos1 = tarFaceOmega.dot(refEdgeTar) / tarFaceOmega.norm() / refEdgeTar.norm();
				cos1 = std::clamp(cos1, -1., 1.);   // avoid numerical issues
				double phi1 = std::acos(cos1);
				int sign1 = 1;
				if (refEdgeTar.cross(tarFaceOmega).dot(tarFaceNormal) < 0)
					sign1 *= -1;

				if (std::abs(sign1 * phi1 - sign0 * phi0) <= M_PI)		// valid rotations
				{
					phi1 *= sign1;
					phi0 *= sign0;
					double f0 = a0 * w0Sq;
					double f1 = a1 * w1Sq;

					double phi = (1 - t) * phi0 + t * phi1;
					faceOmega = rotateSingleVector(refEdge.transpose(), rotAxis.transpose(), phi);
				}
				else
				{
					phi0 = -sign0 * (M_PI - phi0);
					phi1 = -sign1 * (M_PI - phi1);

					double phi = (1 - t) * phi0 + t * phi1;
					faceOmega = rotateSingleVector(-refEdge.transpose(), rotAxis.transpose(), phi);
				}
				faceOmega = faceOmega / faceOmega.norm() * std::sqrt(wSq);

			}
			//faceOmega = (1 - t) * initFaceOmega + t * tarFaceOmega;
			double div0 = _mesh.edgeFace(eid0, 0) == -1 || _mesh.edgeFace(eid0, 1) == -1 ? 1 : 2; // whethe an edge is boundary edge
			double div1 = _mesh.edgeFace(eid1, 0) == -1 || _mesh.edgeFace(eid1, 1) == -1 ? 1 : 2;

			curOmega[eid0] += flag0 * faceOmega.dot(r0) / div0 / 2; // #div of edge faces, and two edge vertices
			curOmega[eid1] += flag1 * faceOmega.dot(r1) / div1 / 2;
		}
	}
	return curOmega;
}

void WrinkleEditingBaseMeshSeqCWF::computeAmpOmegaSq(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::VectorXd& omega, Eigen::VectorXd& ampOmegaSq)
{
	int nfaces = mesh.nFaces();
	int nverts = pos.rows();
	ampOmegaSq.setZero(nverts);
	Eigen::VectorXd vNeis(nverts);
	vNeis.setZero();

	for (int fid = 0; fid < nfaces; fid++)
	{
		for (int vInF = 0; vInF < 3; vInF++)
		{
			int vid = mesh.faceVertex(fid, vInF);

			int eid0 = mesh.faceEdge(fid, (vInF + 1) % 3);
			int eid1 = mesh.faceEdge(fid, (vInF + 2) % 3);
			Eigen::RowVector3d r0 = pos.row(mesh.edgeVertex(eid0, 1)) - pos.row(mesh.edgeVertex(eid0, 0));
			Eigen::RowVector3d r1 = pos.row(mesh.edgeVertex(eid1, 1)) - pos.row(mesh.edgeVertex(eid1, 0));

			Eigen::Matrix2d Iinv, I;
			I << r0.dot(r0), r0.dot(r1), r1.dot(r0), r1.dot(r1);
			Iinv = I.inverse();

			Eigen::Vector2d rhs0;
			rhs0 << omega[eid0], omega[eid1];

			Eigen::Vector2d sol0 = Iinv * rhs0;
			Eigen::Vector3d faceOmega = sol0[0] * r0 + sol0[1] * r1;

			ampOmegaSq[vid] += faceOmega.squaredNorm() * amp[vid];
			vNeis[vid] += 1;
		}
	}
	for (int i = 0; i < nverts; i++)
	{
		ampOmegaSq[i] /= vNeis[i];
	}
}

void WrinkleEditingBaseMeshSeqCWF::computeAmpSqOmegaQuaticAverage()
{
	int numFrames = _unitZvalsList.size();
	_ampSqOmegaQauticAverageList.setZero(numFrames);
	_ampSqOmegaQuaticAverage = 0;
	
	for (int i = 0; i < numFrames; i++)
	{
		double activeArea = _vertAreaList[i].sum();
		int nverts = _posList[i].rows();

		for (int j = 0; j < nverts; j++)
		{
			_ampSqOmegaQauticAverageList[i] +=
				_ampTimesOmegaSq[i][j] * _ampTimesOmegaSq[i][j] * _vertAreaList[i](j);
			if (i == 0)
				activeArea += _vertAreaList[i](j);
		}
		_ampSqOmegaQauticAverageList[i] /= activeArea;
		_ampSqOmegaQuaticAverage += _ampSqOmegaQauticAverageList[i];
	}
	_ampSqOmegaQuaticAverage /= numFrames;
}

void WrinkleEditingBaseMeshSeqCWF::getComponentNorm(const Eigen::VectorXd& x, double& znorm, double& wnorm)
{
	int nverts = _posList[0].rows();

	int numFrames = _unitZvalsList.size() - 2;
	int nDOFs = 2 * nverts;

	znorm = 0;
	wnorm = 0;

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			znorm = std::max(znorm, std::abs(x(i * nDOFs + 2 * j)));
			znorm = std::max(znorm, std::abs(x(i * nDOFs + 2 * j + 1)));
		}
	}
}

void WrinkleEditingBaseMeshSeqCWF::save(const Eigen::VectorXd& x0, std::string* workingFolder)
{
	convertVariable2List(x0);
	std::string tmpFolder;
	if (workingFolder)
		tmpFolder = (*workingFolder) + "/tmpRes/";
	else
		tmpFolder = _savingFolder + "tmpRes/";
	mkdir(tmpFolder);

	std::string outputFolder = tmpFolder + "/optZvals/";
	mkdir(outputFolder);

	std::string omegaOutputFolder = tmpFolder + "/optOmega/";
	mkdir(omegaOutputFolder);

	std::string refOmegaOutputFolder = tmpFolder + "/refOmega/";
	mkdir(refOmegaOutputFolder);

	// save reference
	std::string refAmpOutputFolder = tmpFolder + "/refAmp/";
	mkdir(refAmpOutputFolder);

	int nframes = _zvalsList.size();
	auto savePerFrame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			saveVertexZvals(outputFolder + "unitZvals_" + std::to_string(i) + ".txt", _unitZvalsList[i]);
			saveVertexZvals(outputFolder + "zvals_" + std::to_string(i) + ".txt", _zvalsList[i]);
			saveEdgeOmega(omegaOutputFolder + "omega_" + std::to_string(i) + ".txt", _edgeOmegaList[i]);
			saveVertexAmp(refAmpOutputFolder + "amp_" + std::to_string(i) + ".txt", _combinedRefAmpList[i]);
			saveEdgeOmega(refOmegaOutputFolder + "omega_" + std::to_string(i) + ".txt", _combinedRefOmegaList[i]);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes, GRAIN_SIZE);
	tbb::parallel_for(rangex, savePerFrame);
}

void WrinkleEditingBaseMeshSeqCWF::convertList2Variable(Eigen::VectorXd& x)
{
	int nverts = _posList[0].rows();
	int nedges = _mesh.nEdges();

	int numFrames = _unitZvalsList.size() - 2;

	int DOFsPerframe = 2 * nverts;

	int DOFs = numFrames * DOFsPerframe;

	x.setZero(DOFs);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			x(i * DOFsPerframe + 2 * j) = _unitZvalsList[i + 1][j].real();
			x(i * DOFsPerframe + 2 * j + 1) = _unitZvalsList[i + 1][j].imag();
		}
	}
}

void WrinkleEditingBaseMeshSeqCWF::convertVariable2List(const Eigen::VectorXd& x)
{
	int nverts = _posList[0].rows();

	int numFrames = _unitZvalsList.size() - 2;

	int DOFsPerframe = 2 * nverts;

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			_unitZvalsList[i + 1][j] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
		}
	}
}

double WrinkleEditingBaseMeshSeqCWF::temporalAmpDifference(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nverts = _posList[frameId].rows();
	double energy = 0;

	if (deriv)
		deriv->setZero(2 * nverts);
	if (hessT)
		hessT->clear();

	double dt = 1. / (_unitZvalsList.size() - 1);

	for (int vid = 0; vid < nverts; vid++)
	{
		double ampSq = _unitZvalsList[frameId][vid].real() * _unitZvalsList[frameId][vid].real() +
			_unitZvalsList[frameId][vid].imag() * _unitZvalsList[frameId][vid].imag();
		double refAmpSq = 1;
		/*double cf = (_ampTimesOmegaSq[0][vid] * _ampTimesOmegaSq[0][vid] + _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid] * _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid]) / 2;*/
		double cf = 1;
		double ca = _spatialAmpRatio * _vertAreaList[frameId](vid) * dt * cf;

		energy += ca * (ampSq - refAmpSq) * (ampSq - refAmpSq);

		if (deriv)
		{
			(*deriv)(2 * vid) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _unitZvalsList[frameId][vid].real());
			(*deriv)(2 * vid + 1) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _unitZvalsList[frameId][vid].imag());
		}

		if (hessT)
		{
			Eigen::Matrix2d tmpHess;
			tmpHess <<
				2.0 * _unitZvalsList[frameId][vid].real() * 2.0 * _unitZvalsList[frameId][vid].real(),
				2.0 * _unitZvalsList[frameId][vid].real() * 2.0 * _unitZvalsList[frameId][vid].imag(),
				2.0 * _unitZvalsList[frameId][vid].real() * 2.0 * _unitZvalsList[frameId][vid].imag(),
				2.0 * _unitZvalsList[frameId][vid].imag() * 2.0 * _unitZvalsList[frameId][vid].imag();

			tmpHess *= 2.0 * ca;
			tmpHess += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * Eigen::Matrix2d::Identity());


			if (isProj)
				tmpHess = SPDProjection(tmpHess);

			for (int k = 0; k < 2; k++)
				for (int l = 0; l < 2; l++)
					hessT->push_back({ 2 * vid + k, 2 * vid + l, tmpHess(k, l) });
		}
	}
	return energy;
}


double WrinkleEditingBaseMeshSeqCWF::spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	double energy = 0;
	int nedges = _mesh.nEdges();
	int nverts = _posList[frameId].rows();
	std::vector<Eigen::Triplet<double>> AT;
	AT.clear();
	double dt = 1. / (_unitZvalsList.size() - 1);
	for (int eid = 0; eid < nedges; eid++)
	{
		int vid0 = _mesh.edgeVertex(eid, 0);
		int vid1 = _mesh.edgeVertex(eid, 1);

		double r0 = 1;
		double r1 = 1;

		std::complex<double> expw0 = std::complex<double>(std::cos(_edgeOmegaList[frameId](eid)), std::sin(_edgeOmegaList[frameId](eid)));

		std::complex<double> z0 = _unitZvalsList[frameId][vid0];
		std::complex<double> z1 = _unitZvalsList[frameId][vid1];

		/*double cf = (_ampTimesOmegaSq[0][vid0] * _ampTimesOmegaSq[0][vid0] + _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid0] * _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid0]) / 2;
		cf += (_ampTimesOmegaSq[0][vid1] * _ampTimesOmegaSq[0][vid1] + _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid1] * _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid1]) / 2;*/

		double cf = 2;
		double ce = _spatialKnoppelRatio * _edgeAreaList[frameId](eid) * dt * cf / 2;

		energy += 0.5 * norm((r1 * z0 * expw0 - r0 * z1)) * ce;

		if (deriv || hessT)
		{
			AT.push_back({ 2 * vid0, 2 * vid0, r1 * r1 * ce });
			AT.push_back({ 2 * vid0 + 1, 2 * vid0 + 1, r1 * r1 * ce });

			AT.push_back({ 2 * vid1, 2 * vid1, r0 * r0 * ce });
			AT.push_back({ 2 * vid1 + 1, 2 * vid1 + 1, r0 * r0 * ce });


			AT.push_back({ 2 * vid0, 2 * vid1, -ce * (expw0.real()) * r0 * r1 });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1, -ce * (-expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid0, 2 * vid1 + 1, -ce * (expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid0 + 1, 2 * vid1 + 1, -ce * (expw0.real()) * r0 * r1 });

			AT.push_back({ 2 * vid1, 2 * vid0, -ce * (expw0.real()) * r0 * r1 });
			AT.push_back({ 2 * vid1, 2 * vid0 + 1, -ce * (-expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0, -ce * (expw0.imag()) * r0 * r1 });
			AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -ce * (expw0.real()) * r0 * r1 });
		}
	}

	if (deriv || hessT)
	{
		Eigen::SparseMatrix<double> A;

		A.resize(2 * nverts, 2 * nverts);
		A.setFromTriplets(AT.begin(), AT.end());

		// check whether A is PD


		if (deriv)
		{
			Eigen::VectorXd fvals(2 * nverts);
			for (int i = 0; i < nverts; i++)
			{
				fvals(2 * i) = _unitZvalsList[frameId][i].real();
				fvals(2 * i + 1) = _unitZvalsList[frameId][i].imag();
			}
			(*deriv) = A * fvals;
		}

		if (hessT)
			(*hessT) = AT;
	}

	return energy;
}

double WrinkleEditingBaseMeshSeqCWF::kineticEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
	int nverts = _posList[frameId].rows();
	double dt = 1. / (_unitZvalsList.size() - 1);
	double energy = 0;

	int DOFsPerframe = 2 * nverts;

	if (deriv)
		deriv->setZero(4 * nverts);

	for (int vid = 0; vid < nverts; vid++)
	{
		Eigen::Vector2d diff;
		double coeff = (_ampTimesOmegaSq[0][vid] * _ampTimesOmegaSq[0][vid] + _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid] * _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid]) / 2;
		coeff /= _ampSqOmegaQuaticAverage;
		coeff *= 1.0 / (dt * dt) * _vertAreaList[frameId][vid] * dt;

		diff << (_unitZvalsList[frameId + 1][vid] - _unitZvalsList[frameId][vid]).real(), (_unitZvalsList[frameId + 1][vid] - _unitZvalsList[frameId][vid]).imag();
		energy += 0.5 * coeff * diff.squaredNorm();

		if (deriv)
		{
			deriv->segment<2>(2 * vid) += -coeff * diff;
			deriv->segment<2>(2 * vid + DOFsPerframe) += coeff * diff;
		}

		if (hessT)
		{
			hessT->push_back({ 2 * vid, 2 * vid, coeff });
			hessT->push_back({ 2 * vid, DOFsPerframe + 2 * vid, -coeff });

			hessT->push_back({ 2 * vid + 1, 2 * vid + 1, coeff });
			hessT->push_back({ 2 * vid + 1, DOFsPerframe + 2 * vid + 1, -coeff });

			hessT->push_back({ DOFsPerframe + 2 * vid, DOFsPerframe + 2 * vid, coeff });
			hessT->push_back({ DOFsPerframe + 2 * vid, 2 * vid, -coeff });

			hessT->push_back({ DOFsPerframe + 2 * vid + 1, DOFsPerframe + 2 * vid + 1, coeff });
			hessT->push_back({ DOFsPerframe + 2 * vid + 1, 2 * vid + 1, -coeff });

		}
	}
	return energy;
}

double WrinkleEditingBaseMeshSeqCWF::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
	int nverts = _posList[0].rows();

	int numFrames = _unitZvalsList.size() - 2;

	int DOFsPerframe = 2 * nverts;

	int DOFs = numFrames * DOFsPerframe;

	convertVariable2List(x);

	Eigen::VectorXd curDeriv;
	std::vector<Eigen::Triplet<double>> T, curT;

	double energy = 0;
	if (deriv)
	{
		deriv->setZero(DOFs);
	}

	std::vector<Eigen::VectorXd> curKDerivList(numFrames + 1);
	std::vector<std::vector<Eigen::Triplet<double>>> curKTList(numFrames + 1);
	std::vector<double> keList(numFrames + 1);

	auto kineticEnergyPerframe = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			keList[i] = kineticEnergy(i, deriv ? &curKDerivList[i] : NULL, hess ? &curKTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames + 1, GRAIN_SIZE);
	tbb::parallel_for(rangex, kineticEnergyPerframe);

	double ke = 0;
	for (int i = 0; i < _unitZvalsList.size() - 1; i++)
	{
		ke += keList[i];

		if (deriv)
		{
			if (i == 0)
				deriv->segment(0, DOFsPerframe) += curKDerivList[i].segment(DOFsPerframe, DOFsPerframe);
			else if (i == _unitZvalsList.size() - 2)
				deriv->segment((i - 1) * DOFsPerframe, DOFsPerframe) += curKDerivList[i].segment(0, DOFsPerframe);
			else
			{
				deriv->segment((i - 1) * DOFsPerframe, 2 * DOFsPerframe) += curKDerivList[i];
			}
		}

		if (hess)
		{
			for (auto& it : curKTList[i])
			{

				if (i == 0)
				{
					if (it.row() >= DOFsPerframe && it.col() >= DOFsPerframe)
						T.push_back({ it.row() - DOFsPerframe, it.col() - DOFsPerframe, it.value() });
				}
				else if (i == _unitZvalsList.size() - 2)
				{
					if (it.row() < DOFsPerframe && it.col() < DOFsPerframe)
						T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
				}
				else
				{
					T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
				}


			}
		}
	}
	energy += ke;


	std::vector<Eigen::VectorXd> ampDerivList(numFrames), knoppelDerivList(numFrames);
	std::vector<std::vector<Eigen::Triplet<double>>> ampTList(numFrames), knoppelTList(numFrames);
	std::vector<double> ampEnergyList(numFrames), knoppelEnergyList(numFrames);

	auto otherEnergiesPerframe = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			ampEnergyList[i] = temporalAmpDifference(i + 1, deriv ? &ampDerivList[i] : NULL, hess ? &ampTList[i] : NULL, isProj);
			knoppelEnergyList[i] = spatialKnoppelEnergy(i + 1, deriv ? &knoppelDerivList[i] : NULL, hess ? &knoppelTList[i] : NULL, isProj);
		}
	};

	tbb::blocked_range<uint32_t> rangex1(0u, (uint32_t)numFrames, GRAIN_SIZE);
	tbb::parallel_for(rangex1, otherEnergiesPerframe);

	double ampE = 0, knoppelE = 0;

	for (int i = 0; i < numFrames; i++)
	{
		ampE += ampEnergyList[i];
		knoppelE += knoppelEnergyList[i];

		if (deriv)
		{
			deriv->segment(i * DOFsPerframe, DOFsPerframe) += knoppelDerivList[i];
			deriv->segment(i * DOFsPerframe, 2 * nverts) += ampDerivList[i];
		}

		if (hess)
		{
			for (auto& it : ampTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
			}
			for (auto& it : knoppelTList[i])
			{
				T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
			}
		}
	}

	energy += ampE + knoppelE;


	if (hess)
	{
		//std::cout << "num of triplets: " << T.size() << std::endl;
		hess->resize(DOFs, DOFs);
		hess->setFromTriplets(T.begin(), T.end());
		std::cout << "kinetic energy: " << ke << ", amp energy: " << ampE << ", knoppel energy: " << knoppelE << std::endl;
	}
	return energy;
}

void WrinkleEditingBaseMeshSeqCWF::solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol, double xTol, double fTol, bool isdisplayInfo, std::string workingFolder)
{
	std::cout << "CWF model with new formula" << std::endl;
	computeAmpSqOmegaQuaticAverage();
	std::cout << "a^2 * |w|^4 = " << _ampSqOmegaQuaticAverage << std::endl;
	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
		Eigen::VectorXd deriv;
		Eigen::SparseMatrix<double> H;
		double E = computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

		if (grad)
		{
			(*grad) = deriv;
		}

		if (hess)
		{
			(*hess) = H;
		}

		return E;
	};
	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	auto getVecNorm = [&](const Eigen::VectorXd& x, double& znorm, double& wnorm) {
		getComponentNorm(x, znorm, wnorm);
	};
	auto saveTmpRes = [&](const Eigen::VectorXd& x, std::string* folder)
	{
		save(x, folder);
	};

	OptSolver::testFuncGradHessian(funVal, x);
	auto x0 = x;
	Eigen::VectorXd grad;
	Eigen::SparseMatrix<double> hess;
	double f0 = funVal(x0, &grad, &hess, false);
	std::cout << "initial f: " << f0 << ", grad norm: " << grad.norm() << ", hess norm: " << hess.norm() << std::endl;

	if (std::isnan(grad.norm()) || std::isnan(hess.norm()))
	{
		std::cerr << "get nan error in hessian or gradient computation!" << std::endl;
		exit(EXIT_FAILURE);
	}
	OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, std::max(1e-16, xTol), std::max(1e-16, fTol), true, getVecNorm, &workingFolder, saveTmpRes);
	std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;

	// get zvals
	convertVariable2List(x);
	for (int i = 1; i < _zvalsList.size() - 1; i++)
	{
		for (int j = 0; j < _zvalsList[i].size(); j++)
		{
			_zvalsList[i][j] = _unitZvalsList[i][j] * _combinedRefAmpList[i][j];
		}
	}

	std::cout << "solve finished." << std::endl;

	convertVariable2List(x);

	for (int i = 0; i < _zvalsList.size() - 2; i++)
	{
		std::cout << "frame: " << i << std::endl;
		double kinetic = kineticEnergy(i, NULL, NULL);
		double ampEnergy = temporalAmpDifference(i + 1, NULL, NULL);
		double knoppelEnergy = spatialKnoppelEnergy(i + 1, NULL, NULL);
		std::cout << "kinetic: " << kinetic << ", amp: " << ampEnergy << ", knoppel: " << knoppelEnergy << std::endl;
	}
	std::cout << "frame: " << _zvalsList.size() - 2 << std::endl;
	double kinetic = kineticEnergy(_zvalsList.size() - 2, NULL, NULL);
	std::cout << "kinetic: " << kinetic << ", amp: " << 0 << ", knoppel: " << 0 << std::endl;
	return;
}