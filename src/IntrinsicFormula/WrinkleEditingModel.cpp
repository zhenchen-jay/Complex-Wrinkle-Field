#include "../../include/IntrinsicFormula/WrinkleEditingModel.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>
#include <igl/heat_geodesics.h>
#include <igl/avg_edge_length.h>

using namespace IntrinsicFormula;

WrinkleEditingModel::WrinkleEditingModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio, int effectivedistFactor)
{
	_pos = pos;
	_mesh = mesh;

	_quadOrd = quadOrd;
	_spatialAmpRatio = spatialAmpRatio;
	_spatialEdgeRatio = spatialEdgeRatio;
	_spatialKnoppelRatio = spatialKnoppelRatio;
	_effectiveFactor = effectivedistFactor;

	_vertexOpts = vertexOpts;

	int nverts = pos.rows();
	int nfaces = mesh.nFaces();
	int nedges = mesh.nEdges();

	_edgeCotCoeffs.setZero(nedges);

	buildVertexNeighboringInfo(_mesh, _pos.rows(), _vertNeiEdges, _vertNeiFaces);
	_vertArea = getVertArea(_pos, _mesh);
	_edgeArea = getEdgeArea(_pos, _mesh);
	_faceArea = getFaceArea(_pos, _mesh);
	

	std::vector<int> bnds;
	igl::boundary_loop(_mesh.faces(), bnds);

	_nInterfaces = 0;

	Eigen::MatrixXd cotMatrixEntries;

	igl::cotmatrix_entries(pos, mesh.faces(), cotMatrixEntries);

	for (int i = 0; i < nfaces; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int eid = _mesh.faceEdge(i, j);
			int vid = _mesh.faceVertex(i, j);
			_edgeCotCoeffs(eid) += cotMatrixEntries(i, j);
		}
	}

	_faceVertMetrics.resize(nfaces);
	for (int i = 0; i < nfaces; i++)
	{
		_faceVertMetrics[i].resize(3);
		for (int j = 0; j < 3; j++)
		{
			int vid = _mesh.faceVertex(i, j);
			int vidj = _mesh.faceVertex(i, (j + 1) % 3);
			int vidk = _mesh.faceVertex(i, (j + 2) % 3);

			Eigen::Vector3d e0 = _pos.row(vidj) - _pos.row(vid);
			Eigen::Vector3d e1 = _pos.row(vidk) - _pos.row(vid);

			Eigen::Matrix2d I;
			I << e0.dot(e0), e0.dot(e1), e1.dot(e0), e1.dot(e1);
			_faceVertMetrics[i][j] = I.inverse();
		}
	}
	faceFlagsSetup(faceFlag);
	_nInterfaces = _interfaceFids.size();
	_savingFolder = "";

	std::cout << "number of interfaces: " << _nInterfaces << std::endl;
	std::cout << "min edge area: " << _edgeArea.minCoeff() << ", min vertex area: " << _vertArea.minCoeff() << std::endl;
	std::cout << "max edge area: " << _edgeArea.maxCoeff() << ", max vertex area: " << _vertArea.maxCoeff() << std::endl;

}

void WrinkleEditingModel::faceFlagsSetup(const Eigen::VectorXi& faceFlags)
{
	int nfaces = _mesh.nFaces();
	for (int i = 0; i < nfaces; i++)
	{
		if (faceFlags(i) == 1)
		{
			_selectedFids.push_back(i);
		}
		else if (faceFlags(i) == -1)
		{
			_interfaceFids.push_back(i);
		}
		else
			_unselectedFids.push_back(i);
	}
	// selected edges and verts
	Eigen::VectorXi selectedEdgeFlags, selectedVertFlags;
	getVertIdinGivenDomain(_selectedFids, selectedVertFlags);
	getEdgeIdinGivenDomain(_selectedFids, selectedEdgeFlags);

	// unselected edges and verts
	Eigen::VectorXi unselectedEdgeFlags, unselectedVertFlags;
	getVertIdinGivenDomain(_unselectedFids, unselectedVertFlags);
	getEdgeIdinGivenDomain(_unselectedFids, unselectedEdgeFlags);

	// interface edges and verts
	Eigen::VectorXi interfaceEdgeFlags, interfaceVertFlags;
	getVertIdinGivenDomain(_interfaceFids, interfaceVertFlags);
	getEdgeIdinGivenDomain(_interfaceFids, interfaceEdgeFlags);

	// build the list
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

	for (int i = 0; i < nverts; i++)
	{
		if (selectedVertFlags(i))
			_selectedVids.push_back(i);
		else if (unselectedVertFlags(i))
			_unselectedVids.push_back(i);
		else
			_interfaceVids.push_back(i);
	}

	for (int i = 0; i < nedges; i++)
	{
		if (selectedEdgeFlags(i))
			_selectedEids.push_back(i);
		else if (unselectedEdgeFlags(i))
			_unselectedEids.push_back(i);
		else
			_interfaceEids.push_back(i);
	}

}

void WrinkleEditingModel::getVertIdinGivenDomain(const std::vector<int> faceList, Eigen::VectorXi& vertFlags)
{
	int nverts = _pos.rows();
	vertFlags.setZero(nverts);

	for (auto& fid : faceList)
	{
		for (int i = 0; i < 3; i++)
		{
			int vid = _mesh.faceVertex(fid, i);
			vertFlags(vid) = 1;
		}
	}
}

void WrinkleEditingModel::getEdgeIdinGivenDomain(const std::vector<int> faceList, Eigen::VectorXi& edgeFlags)
{
	int nedges = _mesh.nEdges();
	edgeFlags.setZero(nedges);

	for (auto& fid : faceList)
	{
		for (int i = 0; i < 3; i++)
		{
			int eid = _mesh.faceEdge(fid, i);
			edgeFlags(eid) = 1;
		}
	}
}
void WrinkleEditingModel::buildWeights()
{
	int nverts = _pos.rows();
	int nfaces = _mesh.nFaces();

	_faceWeight.setZero(nfaces);
	_vertWeight.setZero(nverts);

	if (_selectedVids.size() && _effectiveFactor > 0)
	{

		Eigen::VectorXi selectedEdgeFlags, selectedVertFlags;
		getVertIdinGivenDomain(_selectedFids, selectedVertFlags);
		getEdgeIdinGivenDomain(_selectedFids, selectedEdgeFlags);

		// interface edges and verts
		Eigen::VectorXi interfaceEdgeFlags, interfaceVertFlags;
		getVertIdinGivenDomain(_interfaceFids, interfaceVertFlags);
		getEdgeIdinGivenDomain(_interfaceFids, interfaceEdgeFlags);

		std::vector<int> sourceVerts;
		for (int i = 0; i < nverts; i++)
		{
			if (selectedVertFlags(i))
				sourceVerts.push_back(i);
			else if (interfaceVertFlags(i))
				sourceVerts.push_back(i);
		}
		if (sourceVerts.size() == nverts)
		{
			_faceWeight.setConstant(1.0);
			_vertWeight.setConstant(1.0);
		}
		else
		{
			// build geodesics
			// Precomputation
			igl::HeatGeodesicsData<double> data;
			double t = std::pow(igl::avg_edge_length(_pos, _mesh.faces()), 2);
			const auto precompute = [&]()
			{
				if (!igl::heat_geodesics_precompute(_pos, _mesh.faces(), t, data))
				{
					std::cerr << "Error: heat_geodesics_precompute failed." << std::endl;
					exit(EXIT_FAILURE);
				};
			};
			precompute();

			Eigen::VectorXi gamma(sourceVerts.size());
			for (int i = 0; i < gamma.rows(); i++)
				gamma(i) = sourceVerts[i];

			Eigen::VectorXd dis;
			igl::heat_geodesics_solve(data, gamma, dis);

			for (int i = 0; i < nverts; i++)
			{
				if (selectedVertFlags(i) || interfaceVertFlags(i))
					dis(i) = 0;
			}

			double min = dis.minCoeff();
			double max = dis.maxCoeff();

			std::cout << "min geo: " << min << ", max geo: " << max << std::endl;

			double mu = 0;
			double sigma = (max - min) / _effectiveFactor;

			for (int i = 0; i < nfaces; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					int vid = _mesh.faceVertex(i, j);
					double weight = std::min(expGrowth(dis(vid), mu, sigma), 1e10);
					_vertWeight(vid) = weight;
					_faceWeight(i) += weight / 3;
				}
			}
			std::cout << "face weight min: " << _faceWeight.minCoeff() << ", face weight max: " << _faceWeight.maxCoeff() << std::endl;
		}

	}
	else
	{
		_faceWeight.setConstant(1.0);
		_vertWeight.setConstant(1.0);
	}
}


void WrinkleEditingModel::adjustOmegaForConsistency(const std::vector<std::complex<double>>& zvals, const Eigen::VectorXd& amp, Eigen::VectorXd& omega, Eigen::VectorXi* edgeFlag)
{
	int nedges = _mesh.nEdges();
	Eigen::VectorXd deltaOmega = Eigen::VectorXd::Zero(nedges);

	for (int i = 0; i < nedges; i++)
	{
		if (edgeFlag)
		{
			if ((*edgeFlag)(i) == 1)	// fixed omega
				continue;
		}
		int v0 = _mesh.edgeVertex(i, 0);
		int v1 = _mesh.edgeVertex(i, 1);

        if(amp(v0) == 0 || amp(v1) == 0)
            continue;

		double theta1 = std::arg(amp(v0)*zvals[v1]);
		double theta0 = std::arg(amp(v1)*zvals[v0]);

		double dtheta = theta1 - theta0;
		double c = (omega(i) - dtheta) / 2.0 / M_PI;
		int k = std::floor(c + 0.5);
		double newEdgeOmega = dtheta + 2 * k * M_PI;
		omega(i) = dtheta + 2 * k * M_PI;

	}

}

void WrinkleEditingModel::vecFieldLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag)
{
    vecList.resize(numFrames + 2);
    vecList[0] = initVec;
    vecList[numFrames + 1] = tarVec;

    Eigen::MatrixXd initFaceVec, tarFaceVec;
    initFaceVec = intrinsicEdgeVec2FaceVec(initVec, _pos, _mesh);
    tarFaceVec = intrinsicEdgeVec2FaceVec(tarVec, _pos, _mesh);

    Eigen::VectorXd theta(initFaceVec.rows());

    Eigen::MatrixXd faceNormals;
    igl::per_face_normals(_pos, _mesh.faces(), faceNormals);

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

void WrinkleEditingModel::vecFieldSLERP(const Eigen::VectorXd& initVec, const Eigen::VectorXd& tarVec, std::vector<Eigen::VectorXd>& vecList, int numFrames, Eigen::VectorXi* edgeFlag)
{
	vecList.resize(numFrames + 2);
	vecList[0] = initVec;
	vecList[numFrames + 1] = tarVec;

	Eigen::MatrixXd initFaceVec, tarFaceVec;
	initFaceVec = intrinsicEdgeVec2FaceVec(initVec, _pos, _mesh);
	tarFaceVec = intrinsicEdgeVec2FaceVec(tarVec, _pos, _mesh);

	Eigen::VectorXd theta(initFaceVec.rows());

	Eigen::MatrixXd faceNormals;
	igl::per_face_normals(_pos, _mesh.faces(), faceNormals);


	for (int i = 0; i < theta.rows(); i++)
	{
		theta(i) = initFaceVec.row(i).dot(tarFaceVec.row(i));
		if (initFaceVec.row(i).norm() && tarFaceVec.row(i).norm())
		{
			Eigen::Vector3d e0 = initFaceVec.row(i);
			Eigen::Vector3d e1 = tarFaceVec.row(i);
			double cos = e0.dot(e1) / (e0.norm() * e1.norm());

			if (std::abs(cos - 1) < 1e-10)	 // almost parallel
			{
				theta(i) = 0;
				continue;
			}
				
			double angle = std::acos(cos);
            theta(i) = angle;
		}
			
		else
			theta(i) = 0;

	}
	double dt = 1. / (numFrames + 1);

	for (int j = 1; j < numFrames + 1; j++)
	{
		double t = dt * j;
		Eigen::MatrixXd faceVec = initFaceVec;

		
		for (int f = 0; f < faceVec.rows(); f++)
		{
			Eigen::Vector3d axis = faceNormals.row(f);
			Eigen::Vector3d vec = Eigen::Vector3d::Zero();

			if (initFaceVec.row(f).norm() == 0)
				vec = t * tarFaceVec.row(f);
			else if (tarFaceVec.row(f).norm() == 0)
				vec = (1 - t) * initFaceVec.row(f);
			else
			{
				double angle = t * theta(f);
				// slerp on theta
				double ux = axis(0) / axis.norm(), uy = axis(1) / axis.norm(), uz = axis(2) / axis.norm();
				Eigen::Matrix3d rotMat;

				double c = std::cos(angle);
				double s = std::sin(angle);;
				rotMat << c + ux * ux * (1 - c), ux* uy* (1 - c) - uz * s, ux* uz* (1 - c) + uy * s,
					uy* ux* (1 - c) + uz * s, c + uy * uy * (1 - c), uy* uz* (1 - c) - ux * s,
					uz* ux* (1 - c) - uy * s, uz* uy* (1 - c) + ux * s, c + uz * uz * (1 - c);

				vec = rotMat * (initFaceVec.row(f).transpose());
					
				vec = vec / vec.norm();
				
				// lerp on mag
				double mag = (1 - t) * initFaceVec.row(f).norm() + t * tarFaceVec.row(f).norm();
				vec *= mag;
			}
			faceVec.row(f) = vec;
			
		}
		vecList[j] = faceVec2IntrinsicEdgeVec(faceVec, _pos, _mesh);
		
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

void WrinkleEditingModel::ampFieldLERP(const Eigen::VectorXd& initAmp, const Eigen::VectorXd& tarAmp, std::vector<Eigen::VectorXd>& ampList, int numFrames, Eigen::VectorXi* vertFlag)
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

void WrinkleEditingModel::initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, const std::vector<std::complex<double>>& tarZvals, const Eigen::VectorXd& tarOmega, int numFrames)
{
	_combinedRefAmpList.resize(numFrames + 2);
	_combinedRefOmegaList.resize(numFrames + 2);

	Eigen::VectorXd initAmp, tarAmp;
	initAmp.setZero(_pos.rows());
	tarAmp.setZero(_pos.rows());

	for (int i = 0; i < initAmp.rows(); i++)
	{
		initAmp(i) = std::abs(initZvals[i]);
		tarAmp(i) = std::abs(tarZvals[i]);
	}

	ampFieldLERP(initAmp, tarAmp, _combinedRefAmpList, numFrames);
	vecFieldSLERP(initOmega, tarOmega, _combinedRefOmegaList, numFrames);

    _deltaOmegaList.resize(numFrames + 2, Eigen::VectorXd::Zero(_mesh.nEdges()));
}

void WrinkleEditingModel::initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, double numFrames, InitializationType initType, double zuenkoTau, int zuenkoInner)
{
	Eigen::VectorXd initAmp;
	initAmp.setZero(_pos.rows());

	for (int i = 0; i < initAmp.rows(); i++)
	{
		initAmp(i) = std::abs(initZvals[i]);
	}

	std::vector<Eigen::VectorXd> refOmegaList(numFrames + 2);
	std::vector<Eigen::VectorXd> refAmpList(numFrames + 2);

	refAmpList[0] = initAmp;

	Eigen::VectorXd initOmegaAdj = initOmega;
	//adjustOmegaForConsistency(initZvals, initAmp, initOmegaAdj);

	refOmegaList[0] = initOmegaAdj;

	double dt = 1.0 / (numFrames + 1);
	for (int i = 1; i <= numFrames + 1; i++)
	{
		std::vector<VertexOpInfo> curVertOpts = _vertexOpts;
		for (int j = 0; j < _vertexOpts.size(); j++)
		{
			if (_vertexOpts[j].vecOptType == None)
				continue;
			double offset = _vertexOpts[j].vecOptType != Enlarge ? 0 : 1;
			double A = _vertexOpts[j].vecOptType != Enlarge ? _vertexOpts[j].vecOptValue : _vertexOpts[j].vecOptValue - 1;

			curVertOpts[j].vecOptValue = offset + A * dt * i;
			curVertOpts[j].vecMagValue = 1 + (_vertexOpts[j].vecMagValue - 1) * dt * i;

		}

		WrinkleFieldsEditor::edgeBasedWrinkleEdition(_pos, _mesh, initAmp, initOmegaAdj, curVertOpts, refAmpList[i], refOmegaList[i]);
	}

	

	std::cout << "initialize bnd zvals." << std::endl;
	std::vector<std::complex<double>> tarZvals;
    _deltaOmegaList.resize(numFrames + 2, Eigen::VectorXd::Zero(_mesh.nEdges()));

	if (!_nInterfaces)
	{
		_combinedRefOmegaList = refOmegaList;
		_combinedRefAmpList = refAmpList;
		
		roundZvalsFromEdgeOmegaVertexMag(_mesh, refOmegaList[numFrames + 1], refAmpList[numFrames + 1], _edgeArea, _vertArea, _pos.rows(), tarZvals);

		//adjustOmegaForConsistency(tarZvals, refAmpList[numFrames + 1], _combinedRefOmegaList[numFrames + 1]);
	}
	else
	{
		/*std::cout << "compute reference omega." << std::endl;
		Eigen::VectorXd tarOmega = computeCombinedRefOmega(refOmegaList[numFrames + 1]);
		std::cout << "compute reference amplitude." << std::endl;
		Eigen::VectorXd tarAmp = computeCombinedRefAmp(refAmpList[numFrames + 1], &tarOmega);*/

		std::cout << "compute reference omega." << std::endl;
		computeCombinedRefOmegaList(refOmegaList);
		std::cout << "compute reference amplitude." << std::endl;
		computeCombinedRefAmpList(refAmpList, &_combinedRefOmegaList);

		Eigen::VectorXd tarOmega = _combinedRefOmegaList[numFrames + 1];
		Eigen::VectorXd tarAmp = _combinedRefAmpList[numFrames + 1];

		

		std::cout << "build weight." << std::endl;
		buildWeights();

		Eigen::VectorXi fixedVertsFlag, fixedEdgeFlags, unchangedEdgeFlags;
		fixedVertsFlag.setZero(_pos.rows());
		unchangedEdgeFlags.setZero(_mesh.nEdges());
		fixedEdgeFlags.setZero(_mesh.nEdges());
		
		for (auto& vid : _unselectedVids)
		{
			fixedVertsFlag(vid) = 1;
		}
			

		for (auto& eid : _selectedEids)
		{
			unchangedEdgeFlags(eid) = 1;
		}
		for (auto& eid : _unselectedEids)
		{
			unchangedEdgeFlags(eid) = 1;
			fixedEdgeFlags(eid) = 1;
		}
		
		tarZvals = initZvals;
		
		roundZvalsForSpecificDomainFromEdgeOmegaBndValues(_mesh, tarOmega, fixedVertsFlag, _edgeArea, _vertArea, _pos.rows(), tarZvals, &(tarAmp));

		for (int i = 0; i < tarZvals.size(); i++)
		{
			if (fixedVertsFlag[i] == 0)
			{
				double arg = std::arg(tarZvals[i]);
				tarZvals[i] = tarAmp[i] * std::complex<double>(std::cos(arg), std::sin(arg));
			}

		}
		/*adjustOmegaForConsistency(tarZvals, tarAmp, tarOmega);

		vecFieldSLERP(initOmega, tarOmega, _combinedRefOmegaList, numFrames, &fixedEdgeFlags);
		ampFieldLERP(initAmp, tarAmp, _combinedRefAmpList, numFrames, &fixedVertsFlag);*/

		double knoppel0 = KnoppelEdgeEnergyGivenMag(_mesh, _combinedRefOmegaList[0], _combinedRefAmpList[0], _edgeArea, initZvals, NULL, NULL);
		double knoppel1 = KnoppelEdgeEnergyGivenMag(_mesh, _combinedRefOmegaList[numFrames + 1], _combinedRefAmpList[numFrames + 1], _edgeArea, tarZvals, NULL, NULL);

		std::cout << "init knoppel: " << knoppel0 << ", tar knoppel: " << knoppel1 << std::endl;
	}



	_edgeOmegaList = _combinedRefOmegaList;
	_zvalsList.resize(numFrames + 2);

	_zvalsList[0] = initZvals;
	_zvalsList[numFrames + 1] = tarZvals;

	std::cout << "initialize the intermediate frames." << std::endl;

	if (initType == Linear)
	{
		for (int i = 1; i <= numFrames; i++)
		{
			double t = i * dt;

			_zvalsList[i] = tarZvals;

			for (int j = 0; j < tarZvals.size(); j++)
			{
				_zvalsList[i][j] = (1 - t) * initZvals[j] + t * tarZvals[j];
			}
		}
	}
	else if(initType == SeperateLinear)
	{
//		ZuenkoAlgorithm(initZvals, _combinedRefOmegaList, _zvalsList, zuenkoTau, zuenkoInner);
//
//		for (int i = 0; i <= numFrames + 1; i++)
//		{
//			for (int j = 0; j < _zvalsList[i].size(); j++)
//			{
//				_zvalsList[i][j] *= _combinedRefAmpList[i][j];
//			}
//		}
        std::cout << "new initialization" << std::endl;

        for (int i = 1; i <= numFrames; i++)
        {
            double t = i * dt;

            _zvalsList[i] = tarZvals;

            for (int j = 0; j < tarZvals.size(); j++)
            {
                _zvalsList[i][j] = (1 - t) * initZvals[j] + t * tarZvals[j];
            }
        }

        Eigen::VectorXd deltaOmega0 = initOmega, deltaOmega1 = _combinedRefOmegaList[numFrames + 1];
        adjustOmegaForConsistency(initZvals, initAmp, deltaOmega0);
        deltaOmega0 = deltaOmega0 - initOmega;

        adjustOmegaForConsistency(tarZvals, _combinedRefAmpList[numFrames + 1], deltaOmega1);
        deltaOmega1 = deltaOmega1 - _combinedRefOmegaList[numFrames + 1];

        vecFieldLERP(deltaOmega0, deltaOmega1, _deltaOmegaList, numFrames);

        _edgeOmegaList = _combinedRefOmegaList;

        for(int i = 0; i < _edgeOmegaList.size(); i++)
        {
            _edgeOmegaList[i] += _deltaOmegaList[i];
        }

	}
    else
	{
		for (int i = 1; i <= numFrames; i++)
		{
			if (!_nInterfaces)
			{
				roundZvalsFromEdgeOmegaVertexMag(_mesh, refOmegaList[i], refAmpList[i], _edgeArea, _vertArea, _pos.rows(), _zvalsList[i]);
			}
			else
			{
				_zvalsList[i] = initZvals;
				Eigen::VectorXi fixedVertsFlag;
				fixedVertsFlag.setZero(_pos.rows());

				for (auto& vid : _unselectedVids)
					fixedVertsFlag(vid) = 1;
				roundZvalsForSpecificDomainFromEdgeOmegaBndValues(_mesh, _combinedRefOmegaList[i], fixedVertsFlag, _edgeArea, _vertArea, _pos.rows(), _zvalsList[i], &(_combinedRefAmpList[i]));

				for (int j = 0; j < _zvalsList[i].size(); j++)
				{
					if (fixedVertsFlag[j] == 0)
					{
						double arg = std::arg(_zvalsList[i][j]);
						_zvalsList[i][j] = refAmpList[i][j] * std::complex<double>(std::cos(arg), std::sin(arg));
					}

				}
			}
		}
	}
	

	_zdotModel = ComputeZdotFromEdgeOmega(_mesh, _faceArea, _quadOrd, dt);
	_refAmpAveList.resize(numFrames + 2);

	for (int i = 0; i < _refAmpAveList.size(); i++)
	{
		double ave = 0;
		for (int j = 0; j < _pos.rows(); j++)
		{
			ave += _combinedRefAmpList[i][j];
		}
		ave /= _pos.rows();
		_refAmpAveList[i] = ave;
	}
}

void WrinkleEditingModel::initialization(const std::vector<std::complex<double>>& initZvals, const Eigen::VectorXd& initOmega, const std::vector<std::complex<double>>& tarZvals, const Eigen::VectorXd& tarOmega, const std::vector<Eigen::VectorXd>& refAmpList, const std::vector<Eigen::VectorXd>& refOmegaList, InitializationType initType)
{
    int numFrames = refAmpList.size() - 2;
    _zvalsList.resize(numFrames + 2);
    _edgeOmegaList.resize(numFrames + 2);

    _edgeOmegaList = refOmegaList;
    _combinedRefAmpList = refAmpList;
    _combinedRefOmegaList = refOmegaList;

    _zvalsList[0] = initZvals;
    _zvalsList[numFrames + 1] = tarZvals;
    _deltaOmegaList.resize(numFrames + 2, Eigen::VectorXd::Zero(_mesh.nEdges()));

    double dt = 1.0 / (numFrames + 1);

    if (initType == Linear)
    {
        for (int i = 1; i <= numFrames; i++)
        {
            double t = i * dt;

            _zvalsList[i] = tarZvals;

            for (int j = 0; j < tarZvals.size(); j++)
            {
                _zvalsList[i][j] = (1 - t) * initZvals[j] + t * tarZvals[j];
            }
        }
    }
    else
    {
        for (int i = 1; i <= numFrames; i++)
        {
            roundZvalsFromEdgeOmegaVertexMag(_mesh, refOmegaList[i], refAmpList[i], _edgeArea, _vertArea, _pos.rows(), _zvalsList[i]);
        }
    }


    _zdotModel = ComputeZdotFromEdgeOmega(_mesh, _faceArea, _quadOrd, dt);
    _refAmpAveList.resize(numFrames + 2);

    for (int i = 0; i < _refAmpAveList.size(); i++)
    {
        double ave = 0;
        for (int j = 0; j < _pos.rows(); j++)
        {
            ave += _combinedRefAmpList[i][j];
        }
        ave /= _pos.rows();
        _refAmpAveList[i] = ave;
    }

}

void WrinkleEditingModel::initialization(const std::vector<std::vector<std::complex<double>>>& zList, const std::vector<Eigen::VectorXd>& omegaList, const std::vector<Eigen::VectorXd>& refAmpList, const std::vector<Eigen::VectorXd>& refOmegaList)
{
	_zvalsList = zList;
	_edgeOmegaList = omegaList;
	_combinedRefAmpList = refAmpList;
	_combinedRefOmegaList = refOmegaList;

	int numFrames = zList.size() - 2;
	double dt = 1.0 / (numFrames + 1);

	_zdotModel = ComputeZdotFromEdgeOmega(_mesh, _faceArea, _quadOrd, dt);
	_refAmpAveList.resize(numFrames + 2);

	for (int i = 0; i < _refAmpAveList.size(); i++)
	{
		double ave = 0;
		for (int j = 0; j < _pos.rows(); j++)
		{
			ave += _combinedRefAmpList[i][j];
		}
		ave /= _pos.rows();
		_refAmpAveList[i] = ave;
	}
}


void WrinkleEditingModel::ZuenkoAlgorithm(const std::vector<std::complex<double>>& initZvals, const std::vector<Eigen::VectorXd>& refOmegaList, std::vector<std::vector<std::complex<double>>>& zList, double zuenkoTau, int zuenkoInner)
{
	int nframes = refOmegaList.size();
	if (!nframes)
		return;
	zList.resize(refOmegaList.size());
	zList[0] = initZvals;

	for (int i = 0; i < zList[0].size(); i++)
	{
		double theta = std::arg(zList[0][i]);
		zList[0][i] = std::complex<double>(std::cos(theta), std::sin(theta));
	}

	int nedges = refOmegaList[0].rows();

	// Zuenko's algorithm described in their section 5.1
	for (int i = 1; i < nframes; i++)
	{
		// step 1: apply their equation (10) k times, where k = 5, as they suggested in their paper
		std::vector<std::complex<double>> Phi0 = zList[i - 1];
		std::vector<std::complex<double>> Phi1 = Phi0;
		for (int k = 0; k < zuenkoInner; k++)
		{
			// update the Phi
			for (int v = 0; v < Phi0.size(); v++)
			{
				std::complex<double> tmpZ = 0;
				for (auto& e : _vertNeiEdges[v]) // all neighboring edges
				{
					double deltaTheta = refOmegaList[i][e];
					int vj = _mesh.edgeVertex(e, 0);
					if (_mesh.edgeVertex(e, 0) == v)
					{
						deltaTheta *= -1;			// make sure deltaTheta = theta_i - theta_j
						vj = _mesh.edgeVertex(e, 1);
					}
						

					std::complex<double> deltaZi(std::cos(deltaTheta), std::sin(deltaTheta));
					tmpZ += deltaZi * Phi0[vj];
				}
				double theta = std::arg(tmpZ);
				Phi1[v] = std::complex<double>(std::cos(theta), std::sin(theta));
			}
			Phi0.swap(Phi1);
		}

		// bluring
		zList[i] = zList[i - 1];
		for (int v = 0; v < Phi0.size(); v++)
		{
			std::complex<double> tmpZ = zuenkoTau * zList[i - 1][v] + (1 - zuenkoTau) * Phi0[v];
			double theta = std::arg(tmpZ);
			zList[i][v] = std::complex<double>(std::cos(theta), std::sin(theta));
		}
	}
}

double WrinkleEditingModel::amplitudeEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& w,
	int fid, Eigen::Vector3d* deriv,
	Eigen::Matrix3d* hess)
{
	double energy = 0;

	double curlSq = curlFreeEnergyPerface(w, fid, NULL, NULL);
	Eigen::Vector3d wSq;
	wSq.setZero();


	if (deriv)
		deriv->setZero();
	if (hess)
		hess->setZero();

	for (int i = 0; i < 3; i++)
	{
		int vid = _mesh.faceVertex(fid, i);
		energy += 0.5 * amp(vid) * amp(vid) / 3 * (wSq(i) * _faceArea(fid) + curlSq);

		if (deriv)
			(*deriv)(i) += amp(vid) * (wSq(i) * _faceArea(fid) + curlSq) / 3;
		if (hess)
			(*hess)(i, i) += (wSq(i) * _faceArea(fid) + curlSq) / 3;
	}

	return energy;
}

double WrinkleEditingModel::amplitudeEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::VectorXd& w,
	Eigen::VectorXd* deriv,
	std::vector<Eigen::Triplet<double>>* hessT)
{
	double energy = 0;

	int nverts = _pos.rows();
	int nEffectiveFaces = _interfaceFids.size();

	std::vector<double> energyList(nEffectiveFaces);
	std::vector<Eigen::Vector3d> derivList(nEffectiveFaces);
	std::vector<Eigen::Matrix3d> hessList(nEffectiveFaces);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			int fid = _interfaceFids[i];
			energyList[i] = amplitudeEnergyWithGivenOmegaPerface(amp, w, fid, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nEffectiveFaces, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
		deriv->setZero(nverts);
	if (hessT)
		hessT->clear();

	for (int efid = 0; efid < nEffectiveFaces; efid++)
	{
		energy += energyList[efid];
		int fid = _interfaceFids[efid];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(fid, j);
				(*deriv)(vid) += derivList[efid](j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int vid = _mesh.faceVertex(fid, j);
				for (int k = 0; k < 3; k++)
				{
					int vid1 = _mesh.faceVertex(fid, k);
					hessT->push_back({ vid, vid1, hessList[efid](j, k) });
				}
			}
		}
	}
	return energy;
}

Eigen::VectorXd WrinkleEditingModel::computeCombinedRefAmp(const Eigen::VectorXd& refAmp, Eigen::VectorXd* combinedOmega)
{
	int nverts = _pos.rows();
	int nfaces = _mesh.nFaces();

	std::vector<Eigen::Triplet<double>> T;
	// projection matrix
	std::vector<int> freeVid = _interfaceVids;
	Eigen::VectorXi fixedFlags = Eigen::VectorXi::Ones(nverts);

	for (int i = 0; i < freeVid.size(); i++)
	{
		T.push_back(Eigen::Triplet<double>(i, freeVid[i], 1.0));
		fixedFlags(freeVid[i]) = 0;
	}

	Eigen::SparseMatrix<double> projM(freeVid.size(), nverts);
	projM.setFromTriplets(T.begin(), T.end());

	Eigen::SparseMatrix<double> unProjM = projM.transpose();

	auto projVar = [&]()
	{
		Eigen::MatrixXd fullX = refAmp;
		Eigen::VectorXd x0 = projM * fullX;
		return x0;
	};

	auto unProjVar = [&](const Eigen::VectorXd& x)
	{
		Eigen::VectorXd fullX = unProjM * x;

		for (int i = 0; i < nverts; i++)
		{
			if (fixedFlags(i))
			{
				fullX(i) = refAmp(i);
			}
		}
		return fullX;
	};

	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(_pos, _mesh.faces(), L);

	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
		Eigen::VectorXd deriv, deriv1;
		std::vector<Eigen::Triplet<double>> T;
		Eigen::SparseMatrix<double> H;

		Eigen::VectorXd fullx = unProjVar(x);
		double E = -0.5 * fullx.dot(L * fullx);

		if (combinedOmega)
		{
			E += amplitudeEnergyWithGivenOmega(fullx, (*combinedOmega), grad ? &deriv1 : NULL, hess ? &T : NULL);
		}

		if (grad)
		{
			deriv = -L * fullx;
			if (combinedOmega)
				deriv += deriv1;
		}

		if (hess)
		{
			if (combinedOmega)
			{
				H.resize(fullx.rows(), fullx.rows());
				H.setFromTriplets(T.begin(), T.end());
				(*hess) = projM * (H - L) * unProjM;
			}

			else
				(*hess) = projM * (-L) * unProjM;

		}

		return E;
	};
	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	Eigen::VectorXd x0 = projVar();
	if (_nInterfaces && freeVid.size())
	{
		OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);

		Eigen::VectorXd deriv;
		double E = funVal(x0, &deriv, NULL, false);
		std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
	}

	return unProjVar(x0);
}

void WrinkleEditingModel::computeCombinedRefAmpList(const std::vector<Eigen::VectorXd>& refAmpList, std::vector<Eigen::VectorXd>* combinedOmegaList)
{
	int nverts = _pos.rows();
	int nfaces = _mesh.nFaces();
	int nFrames = refAmpList.size();

	_combinedRefAmpList.resize(nFrames);

	double c = std::min(1.0 / (nFrames * nFrames), 1e-3);

	std::vector<Eigen::Triplet<double>> T;
	// projection matrix
	std::vector<int> freeVid = _interfaceVids;
	Eigen::VectorXi fixedFlags = Eigen::VectorXi::Ones(nverts);

	for (int i = 0; i < freeVid.size(); i++)
	{
		T.push_back(Eigen::Triplet<double>(i, freeVid[i], 1.0));
		fixedFlags(freeVid[i]) = 0;
	}

	Eigen::SparseMatrix<double> projM(freeVid.size(), nverts);
	projM.setFromTriplets(T.begin(), T.end());

	Eigen::SparseMatrix<double> unProjM = projM.transpose();

	auto projVar = [&](const int frameId)
	{
		Eigen::MatrixXd fullX = refAmpList[frameId];
		Eigen::VectorXd x0 = projM * fullX;
		return x0;
	};

	auto unProjVar = [&](const Eigen::VectorXd& x, const int frameId)
	{
		Eigen::VectorXd fullX = unProjM * x;

		for (int i = 0; i < nverts; i++)
		{
			if (fixedFlags(i))
			{
				fullX(i) = refAmpList[frameId](i);
			}
		}
		return fullX;
	};

	Eigen::SparseMatrix<double> L;
	igl::cotmatrix(_pos, _mesh.faces(), L);

	_combinedRefAmpList[0] = refAmpList[0];

	Eigen::VectorXd prevX = refAmpList[0];

	Eigen::SparseMatrix<double> idmat(prevX.rows(), prevX.rows());
	idmat.setIdentity();


	for (int i = 1; i < nFrames; i++)
	{
		std::cout << "Frame " << std::to_string(i) << ": free vertices: " << freeVid.size() << std::endl;;
		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv, deriv1;
			std::vector<Eigen::Triplet<double>> T;
			Eigen::SparseMatrix<double> H;

			Eigen::VectorXd fullx = unProjVar(x, i);
			double E = -0.5 * fullx.dot(L * fullx) + 0.5 * c * (fullx - prevX).squaredNorm();

			if (combinedOmegaList)
			{
				E += amplitudeEnergyWithGivenOmega(fullx, (*combinedOmegaList)[i], grad ? &deriv1 : NULL, hess ? &T : NULL);
			}

			if (grad)
			{
				deriv = -L * fullx + c * (fullx - prevX);
				if (combinedOmegaList)
					deriv += deriv1;
				(*grad) = projM * deriv;
			}

			if (hess)
			{
				if (combinedOmegaList)
				{
					H.resize(fullx.rows(), fullx.rows());
					H.setFromTriplets(T.begin(), T.end());
					(*hess) = projM * (H - L + c * idmat) * unProjM;
				}

				else
					(*hess) = projM * (-L + c * idmat) * unProjM;

			}

			return E;
		};
		auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
			return 1.0;
		};

		Eigen::VectorXd x0 = projVar(i);
		if (_nInterfaces && freeVid.size())
		{
			OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);

			Eigen::VectorXd deriv;
			double E = funVal(x0, &deriv, NULL, false);
			std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
		}

		_combinedRefAmpList[i] = unProjVar(x0, i);
		prevX = _combinedRefAmpList[i];
	}
}

double WrinkleEditingModel::curlFreeEnergyPerface(const Eigen::MatrixXd& w, int faceId, Eigen::Matrix<double, 3, 1>* deriv, Eigen::Matrix<double, 3, 3>* hess)
{
	double E = 0;

	double diff0;
	Eigen::Matrix<double, 3, 1> select0;
	select0.setZero();

	Eigen::Matrix<double, 3, 1> edgews;

	for (int i = 0; i < 3; i++)
	{
		int eid = _mesh.faceEdge(faceId, i);
		edgews(i) = w(eid);

		if (_mesh.faceVertex(faceId, (i + 1) % 3) == _mesh.edgeVertex(eid, 0))
		{
			select0(i) = 1;
		}

		else
		{
			select0(i) = -1;
		}
	}
	diff0 = select0.dot(edgews);

	E = 0.5 * (diff0 * diff0);
	if (deriv)
	{
		*deriv = select0 * diff0;
	}
	if (hess)
	{
		*hess = select0 * select0.transpose();
	}

	return E;
}


double WrinkleEditingModel::curlFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT)
{
	double E = 0;
	int nedges = _mesh.nEdges();
	int nEffectiveFaces = _interfaceFids.size();

	std::vector<double> energyList(nEffectiveFaces);
	std::vector<Eigen::Matrix<double, 3, 1>> derivList(nEffectiveFaces);
	std::vector<Eigen::Matrix<double, 3, 3>> hessList(nEffectiveFaces);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			int fid = _interfaceFids[i];
			energyList[i] = curlFreeEnergyPerface(w, fid, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nEffectiveFaces, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);


	if (deriv)
		deriv->setZero(nedges);
	if (hessT)
		hessT->clear();

	for (int efid = 0; efid < nEffectiveFaces; efid++)
	{
		E += energyList[efid];
		int fid = _interfaceFids[efid];

		if (deriv)
		{
			for (int j = 0; j < 3; j++)
			{
				int eid = _mesh.faceEdge(fid, j);
				(*deriv)(eid) += derivList[efid](j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < 3; j++)
			{
				int eid = _mesh.faceEdge(fid, j);
				for (int k = 0; k < 3; k++)
				{
					int eid1 = _mesh.faceEdge(fid, k);
					hessT->push_back({ eid, eid1, hessList[efid](j, k) });
				}
			}
		}
	}

	return E;
}


double WrinkleEditingModel::divFreeEnergyPervertex(const Eigen::MatrixXd& w, int vertId, Eigen::VectorXd* deriv, Eigen::MatrixXd* hess)
{
	double energy = 0;
	int neiEdges = _vertNeiEdges[vertId].size();


	Eigen::VectorXd selectedVec0;
	selectedVec0.setZero(neiEdges);

	Eigen::VectorXd edgew;
	edgew.setZero(neiEdges);

	for (int i = 0; i < neiEdges; i++)
	{
		int eid = _vertNeiEdges[vertId][i];
		if (_mesh.edgeVertex(eid, 0) == vertId)
		{
			selectedVec0(i) = _edgeCotCoeffs(eid);
		}
		else
		{
			selectedVec0(i) = -_edgeCotCoeffs(eid);
		}

		edgew(i) = w(eid);
	}
	double diff0 = selectedVec0.dot(edgew);

	energy = 0.5 * (diff0 * diff0);
	if (deriv)
	{
		(*deriv) = (diff0 * selectedVec0);
	}
	if (hess)
	{
		(*hess) = (selectedVec0 * selectedVec0.transpose());
	}

	return energy;
}

double WrinkleEditingModel::divFreeEnergy(const Eigen::MatrixXd& w, Eigen::VectorXd* deriv,
	std::vector<Eigen::Triplet<double>>* hessT)
{
	double energy = 0;
	int nedges = _mesh.nEdges();

	Eigen::VectorXi interfaceVertFlags;
	getVertIdinGivenDomain(_interfaceFids, interfaceVertFlags);
	
	std::vector<int> effectiveVids;
	for (int i = 0; i < interfaceVertFlags.rows(); i++)
	{
		if (interfaceVertFlags(i))
			effectiveVids.push_back(i);
	}

	int nEffectiveVerts = effectiveVids.size();

	std::vector<double> energyList(nEffectiveVerts);
	std::vector<Eigen::VectorXd> derivList(nEffectiveVerts);
	std::vector<Eigen::MatrixXd> hessList(nEffectiveVerts);

	auto computeEnergy = [&](const tbb::blocked_range<uint32_t>& range) {
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			int vid = effectiveVids[i];
			energyList[i] = divFreeEnergyPervertex(w, vid, deriv ? &derivList[i] : NULL, hessT ? &hessList[i] : NULL);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nEffectiveVerts, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeEnergy);

	if (deriv)
		deriv->setZero(nedges);
	if (hessT)
		hessT->clear();

	for (int efid = 0; efid < nEffectiveVerts; efid++)
	{
		int vid = effectiveVids[efid];
		energy += energyList[efid];

		if (deriv)
		{
			for (int j = 0; j < _vertNeiEdges[vid].size(); j++)
			{
				int eid = _vertNeiEdges[vid][j];
				(*deriv)(eid) += derivList[efid](j);
			}
		}

		if (hessT)
		{
			for (int j = 0; j < _vertNeiEdges[vid].size(); j++)
			{
				int eid = _vertNeiEdges[vid][j];
				for (int k = 0; k < _vertNeiEdges[vid].size(); k++)
				{
					int eid1 = _vertNeiEdges[vid][k];
					hessT->push_back({ eid, eid1, hessList[efid](j, k) });
				}
			}
		}
	}

	return energy;
}

void WrinkleEditingModel::computeCombinedRefOmegaList(const std::vector<Eigen::VectorXd>& refOmegaList)
{
	int nedges = _mesh.nEdges();
	int nFrames = refOmegaList.size();

	_combinedRefOmegaList.resize(nFrames);

	std::vector<Eigen::Triplet<double>> T;
	// projection matrix
	std::vector<int> freeEid = _interfaceEids;
	Eigen::VectorXi fixedFlags = Eigen::VectorXi::Ones(nedges);

	for (int i = 0; i < freeEid.size(); i++)
	{
		T.push_back(Eigen::Triplet<double>(i, freeEid[i], 1.0));
		fixedFlags(freeEid[i]) = 0;
	}

	Eigen::SparseMatrix<double> projM(freeEid.size(), nedges);
	projM.setFromTriplets(T.begin(), T.end());

	Eigen::SparseMatrix<double> unProjM = projM.transpose();

	auto projVar = [&](const int frameId)
	{
		Eigen::MatrixXd fullX = Eigen::VectorXd::Zero(nedges);
		for (int i = 0; i < nedges; i++)
		{
			if (fixedFlags(i))
			{
				fullX(i) = refOmegaList[frameId](i);
			}
		}
		Eigen::VectorXd x0 = projM * fullX;
		return x0;
	};

	auto unProjVar = [&](const Eigen::VectorXd& x, const int frameId)
	{
		Eigen::VectorXd fullX = unProjM * x;

		for (int i = 0; i < nedges; i++)
		{
			if (fixedFlags(i))
			{
				fullX(i) = refOmegaList[frameId](i);
			}

		}
		return fullX;
	};

	Eigen::VectorXd prevw = refOmegaList[0];
	_combinedRefOmegaList[0] = refOmegaList[0];

	for (int k = 1; k < nFrames; k++)
	{
		std::cout << "Frame " << std::to_string(k) << ": free edges: " << freeEid.size() << std::endl;
		auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
			Eigen::VectorXd deriv, deriv1;
			std::vector<Eigen::Triplet<double>> T, T1;
			Eigen::SparseMatrix<double> H;
			Eigen::VectorXd w = unProjVar(x, k);

			double E = curlFreeEnergy(w, grad ? &deriv : NULL, hess ? &T : NULL);
			E += divFreeEnergy(w, grad ? &deriv1 : NULL, hess ? &T1 : NULL);

			if (grad)
				deriv += deriv1;
			if (hess)
			{
				std::copy(T1.begin(), T1.end(), std::back_inserter(T));
				H.resize(w.rows(), w.rows());
				H.setFromTriplets(T.begin(), T.end());
			}


			// we need some reg to remove the singularity, where we choose some kinetic energy (||w - prevw||^2), which coeff = 1e-3
			double c = std::min(1.0 / (nFrames * nFrames), 1e-3);
			E += c / 2.0 * (w - prevw).squaredNorm();

			if (grad)
			{
				(*grad) = projM * (deriv + c * (w - prevw));
			}

			if (hess)
			{
				Eigen::SparseMatrix<double> idMat(w.rows(), w.rows());
				idMat.setIdentity();
				(*hess) = projM * (H + c * idMat) * unProjM;
			}

			return E;
		};
		auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
			return 1.0;
		};

		Eigen::VectorXd x0 = projVar(k);
		Eigen::VectorXd fullx = unProjVar(x0, k);

		if (_nInterfaces)
		{
			OptSolver::newtonSolver(funVal, maxStep, x0, 1000, 1e-6, 1e-10, 1e-15, false);
			Eigen::VectorXd deriv;
			double E = funVal(x0, &deriv, NULL, false);
			std::cout << "terminated with energy : " << E << ", gradient norm : " << deriv.norm() << std::endl << std::endl;
		}

		prevw = unProjVar(x0, k);
		_combinedRefOmegaList[k] = prevw;
	}
}

Eigen::VectorXd WrinkleEditingModel::computeCombinedRefOmega(const Eigen::VectorXd& refOmega)
{
	int nedges = _mesh.nEdges();

	std::vector<Eigen::Triplet<double>> T;
	// projection matrix
	std::vector<int> freeEid = _interfaceEids;
	Eigen::VectorXi fixedFlags = Eigen::VectorXi::Ones(nedges);

	for (int i = 0; i < freeEid.size(); i++)
	{
		T.push_back(Eigen::Triplet<double>(i, freeEid[i], 1.0));
		fixedFlags(freeEid[i]) = 0;
	}

	Eigen::SparseMatrix<double> projM(freeEid.size(), nedges);
	projM.setFromTriplets(T.begin(), T.end());

	Eigen::SparseMatrix<double> unProjM = projM.transpose();

	auto projVar = [&]()
	{
		Eigen::MatrixXd fullX = Eigen::VectorXd::Zero(nedges);
		for (int i = 0; i < nedges; i++)
		{
			if (fixedFlags(i))
			{
				fullX(i) = refOmega(i);
			}
		}
		Eigen::VectorXd x0 = projM * fullX;
		return x0;
	};

	auto unProjVar = [&](const Eigen::VectorXd& x)
	{
		Eigen::VectorXd fullX = unProjM * x;

		for (int i = 0; i < nedges; i++)
		{
			if (fixedFlags(i))
			{
				fullX(i) = refOmega(i);
			}

		}
		return fullX;
	};

	
	auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
		Eigen::VectorXd deriv, deriv1;
		std::vector<Eigen::Triplet<double>> T, T1;
		Eigen::SparseMatrix<double> H;
		Eigen::VectorXd w = unProjVar(x);

		double E = curlFreeEnergy(w, grad ? &deriv : NULL, hess ? &T : NULL);
		E += divFreeEnergy(w, grad ? &deriv1 : NULL, hess ? &T1 : NULL);

		if (grad)
			deriv += deriv1;
		if (hess)
		{
			std::copy(T1.begin(), T1.end(), std::back_inserter(T));
			H.resize(w.rows(), w.rows());
			H.setFromTriplets(T.begin(), T.end());
		}


		if (grad)
		{
			(*grad) = projM * deriv;
		}

		if (hess)
		{
			(*hess) = projM * H * unProjM;
		}

		return E;
	};
	auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
		return 1.0;
	};

	Eigen::VectorXd x0 = projVar();
	Eigen::VectorXd fullx = unProjVar(x0);

	return fullx;
}



////////////////////////////////////////////// test functions ///////////////////////////////////////////////////////////////////////////
void WrinkleEditingModel::testCurlFreeEnergy(const Eigen::VectorXd& w)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> T;
	Eigen::SparseMatrix<double> hess;
	double E = curlFreeEnergy(w, &deriv, &T);
	hess.resize(w.rows(), w.rows());
	hess.setFromTriplets(T.begin(), T.end());

	std::cout << "tested curl free energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd deriv1;
		Eigen::VectorXd w1 = w + eps * dir;

		double E1 = curlFreeEnergy(w1, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void WrinkleEditingModel::testCurlFreeEnergyPerface(const Eigen::VectorXd& w, int faceId)
{
	Eigen::Matrix<double, 3, 1> deriv;
	Eigen::Matrix<double, 3, 3> hess;
	double E = curlFreeEnergyPerface(w, faceId, &deriv, &hess);
	Eigen::Matrix<double, 3, 1> dir = deriv;
	dir.setRandom();

	std::cout << "tested curl free energy for face: " << faceId << ", energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd w1 = w;
		for (int j = 0; j < 3; j++)
		{
			int eid = _mesh.faceEdge(faceId, j);
			w1(eid) += eps * dir(j);
		}
		Eigen::Matrix<double, 3, 1> deriv1;
		double E1 = curlFreeEnergyPerface(w1, faceId, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}

}

void WrinkleEditingModel::testDivFreeEnergy(const Eigen::VectorXd& w)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> T;
	Eigen::SparseMatrix<double> hess;
	double E = divFreeEnergy(w, &deriv, &T);
	hess.resize(w.rows(), w.rows());
	hess.setFromTriplets(T.begin(), T.end());

	std::cout << "tested div free energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd deriv1;
		Eigen::VectorXd w1 = w + eps * dir;

		double E1 = divFreeEnergy(w1, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void WrinkleEditingModel::testDivFreeEnergyPervertex(const Eigen::VectorXd& w, int vertId)
{
	Eigen::VectorXd deriv;
	Eigen::MatrixXd hess;
	double E = divFreeEnergyPervertex(w, vertId, &deriv, &hess);
	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	std::cout << "tested div free energy for vertex: " << vertId << ", energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd w1 = w;
		for (int j = 0; j < _vertNeiEdges[vertId].size(); j++)
		{
			int eid = _vertNeiEdges[vertId][j];
			w1(eid) += eps * dir(j);
		}
		Eigen::VectorXd deriv1;
		double E1 = divFreeEnergyPervertex(w1, vertId, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}

}


void WrinkleEditingModel::testAmpEnergyWithGivenOmega(const Eigen::VectorXd& amp, const Eigen::VectorXd& w)
{
	Eigen::VectorXd deriv;
	std::vector<Eigen::Triplet<double>> T;
	Eigen::SparseMatrix<double> hess;
	double E = amplitudeEnergyWithGivenOmega(amp, w, &deriv, &T);
	hess.resize(amp.rows(), amp.rows());
	hess.setFromTriplets(T.begin(), T.end());

	std::cout << "tested amp energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	Eigen::VectorXd dir = deriv;
	dir.setRandom();


	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd deriv1;
		Eigen::VectorXd amp1 = amp;
		for (int j = 0; j < amp.rows(); j++)
		{
			amp1(j) += eps * dir(j);
		}
		double E1 = amplitudeEnergyWithGivenOmega(amp1, w, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}

void WrinkleEditingModel::testAmpEnergyWithGivenOmegaPerface(const Eigen::VectorXd& amp, const Eigen::VectorXd& w, int faceId)
{
	Eigen::Vector3d deriv;
	Eigen::Matrix3d hess;
	double E = amplitudeEnergyWithGivenOmegaPerface(amp, w, faceId, &deriv, &hess);
	Eigen::Vector3d dir = deriv;
	dir.setRandom();

	std::cout << "tested amp energy for face: " << faceId << ", energy: " << E << ", gradient norm: " << deriv.norm() << std::endl;

	for (int i = 3; i < 10; i++)
	{
		double eps = std::pow(0.1, i);
		Eigen::VectorXd amp1 = amp;
		for (int j = 0; j < 3; j++)
		{
			int vid = _mesh.faceVertex(faceId, j);
			amp1(vid) += eps * dir(j);
		}
		Eigen::Vector3d deriv1;
		double E1 = amplitudeEnergyWithGivenOmegaPerface(amp1, w, faceId, &deriv1, NULL);

		std::cout << "\neps: " << eps << std::endl;
		std::cout << "gradient check: " << std::abs((E1 - E) / eps - dir.dot(deriv)) << std::endl;
		std::cout << "hess check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}

}

void WrinkleEditingModel::testEnergy(Eigen::VectorXd x)
{
	Eigen::VectorXd deriv;
	Eigen::SparseMatrix<double> hess;

	double e = computeEnergy(x, &deriv, &hess, false);
	Eigen::VectorXd x0 = x;
	x0.setZero();
	Eigen::VectorXd deriv0;
	Eigen::SparseMatrix<double> hess0;
	double e0 = computeEnergy(x, &deriv0, &hess0, false);

	std::cout << "energy: " << e << ", deriv: " << deriv.norm() << ", hess: " << hess.norm() << std::endl;
	
	Eigen::VectorXd dir = deriv;
	dir.setRandom();

	deriv = hess0 * x + deriv0;
	e = 0.5 * x.dot(hess0 * x) + deriv0.dot(x) + e0;


	for (int i = 3; i < 9; i++)
	{
		double eps = std::pow(0.1, i);

		Eigen::VectorXd deriv1;
		double e1 = computeEnergy(x + eps * dir, &deriv1, NULL, false);
		Eigen::VectorXd x1 = x + eps * dir;
		e1 = 0.5 * x1.dot(hess0 * x1) + deriv0.dot(x1) + e0;
		deriv1 = hess0 * x1 + deriv0;

		std::cout << "eps: " << eps << std::endl;
		std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
		std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
	}
}