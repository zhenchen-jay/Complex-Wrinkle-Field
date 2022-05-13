#include "../../include/IntrinsicFormula/WrinkleEditingKnoppelCWF.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>

using namespace IntrinsicFormula;

void WrinkleEditingKnoppelCWF::convertList2Variable(Eigen::VectorXd& x)
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

void WrinkleEditingKnoppelCWF::convertVariable2List(const Eigen::VectorXd& x)
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


void WrinkleEditingKnoppelCWF::solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol, double xTol, double fTol, bool isdisplayInfo, std::string workingFolder)
{
	std::cout << "Knoppel model: " << std::endl;
	// we solve the knoppel things based on edge omega using the cotan edge weight
	convertVariable2List(x);
	for (int i = 1; i < _edgeOmegaList.size() - 1; i++)
	{
		roundZvalsFromEdgeOmega(_mesh, _edgeOmegaList[i], _edgeCotCoeffs, _vertArea, _vertArea.rows(), _zvalsList[i]);
		for (int j = 0; j < _zvalsList[i].size(); j++)
		{
			if (std::abs(_zvalsList[i][j]))
				_zvalsList[i][j] *= _combinedRefAmpList[i][j] / std::abs(_zvalsList[i][j]);
		}
	}
	convertList2Variable(x);

}