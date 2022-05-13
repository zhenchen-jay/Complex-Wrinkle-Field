#include "../../include/IntrinsicFormula/WrinkleEditingLinearCWF.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>

using namespace IntrinsicFormula;


void WrinkleEditingLinearCWF::convertList2Variable(Eigen::VectorXd& x)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nverts + nedges);

	int DOFs = numFrames * DOFsPerframe;

	x.setZero(DOFs);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			x(i * DOFsPerframe + 2 * j) = _zvalsList[i + 1][j].real();
			x(i * DOFsPerframe + 2 * j + 1) = _zvalsList[i + 1][j].imag();
		}

		for (int j = 0; j < nedges; j++)
		{
			x(i * DOFsPerframe + 2 * nverts + j) = _edgeOmegaList[i + 1](j);
		}
	}
}

void WrinkleEditingLinearCWF::convertVariable2List(const Eigen::VectorXd& x)
{
	int nverts = _pos.rows();
	int nedges = _mesh.nEdges();

	int numFrames = _zvalsList.size() - 2;

	int DOFsPerframe = (2 * nverts + nedges);

	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < nverts; j++)
		{
			_zvalsList[i + 1][j] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
		}

		for (int j = 0; j < nedges; j++)
		{
			_edgeOmegaList[i + 1](j) = x(i * DOFsPerframe + 2 * nverts + j);
		}
	}
}


void WrinkleEditingLinearCWF::solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol, double xTol, double fTol, bool isdisplayInfo, std::string workingFolder)
{
	std::cout << "Linear model: " << std::endl;
	// just do linear interpolation.
	convertVariable2List(x);
	int nsize = _zvalsList.size();

	std::vector<std::complex<double>> initZvals = _zvalsList[0], tarZvals = _zvalsList[nsize - 1];

	double dt = 1. / (nsize - 1);

	for (int i = 1; i < nsize - 1; i++)
	{
		for (int j = 0; j < _zvalsList[i].size(); j++)
		{
			_zvalsList[i][j] = (1 - i * dt) * initZvals[j] + i * dt * tarZvals[j];
		}
		_edgeOmegaList[i] = (1 - i * dt) * _edgeOmegaList[0] + i * dt * _edgeOmegaList[nsize - 1];
	}
	convertList2Variable(x);
}