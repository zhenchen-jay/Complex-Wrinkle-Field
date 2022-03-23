#include "../../include/IntrinsicFormula/WrinkleEditingProcess.h"
#include "../../include/WrinkleFieldsEditor.h"
#include "../../include/Optimization/NewtonDescent.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>

using namespace IntrinsicFormula;

WrinkleEditingProcess::WrinkleEditingProcess(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, int quadOrd, const std::vector<Eigen::VectorXd>& refAmpList, std::vector<Eigen::MatrixXd>& refOmegaList)
{
	_pos = pos;
	_mesh = mesh;
	igl::cotmatrix_entries(pos, mesh.faces(), _cotMatrixEntries);
	igl::doublearea(pos, mesh.faces(), _faceArea);
	_faceArea /= 2.0;
    _quadOrd = quadOrd;
    _refAmpList = refAmpList;
    _refOmegaList = refOmegaList;
}

void WrinkleEditingProcess::initialization(const std::vector<std::vector<VertexOpInfo>>& vertOptInfoList)
{
    // apply the operations
    for (int i = 0; i < vertOptInfoList.size(); i++)
    {
        Eigen::MatrixXd vertOmega = intrinsicHalfEdgeVec2VertexVec(_refOmegaList[i], _pos, _mesh);
        WrinkleFieldsEditor::editWrinkles(_pos, _mesh, _refAmpList[i], vertOmega, vertOptInfoList[i], _refAmpList[i], vertOmega);
        _refOmegaList[i] = vertexVec2IntrinsicHalfEdgeVec(vertOmega, _pos, _mesh);
    }
  
    std::vector<std::complex<double>> initZvals;
    std::vector<std::complex<double>> tarZvals;

    int nFrames = _refOmegaList.size() - 2;
    roundVertexZvalsFromHalfEdgeOmegaVertexMag(_mesh, _refOmegaList[0], _refAmpList[0], _faceArea, _cotMatrixEntries, _pos.rows(), initZvals);
    roundVertexZvalsFromHalfEdgeOmegaVertexMag(_mesh, _refOmegaList[nFrames + 1], _refAmpList[nFrames + 1], _faceArea, _cotMatrixEntries, _pos.rows(), tarZvals);

    _model = IntrinsicKnoppelDrivenFormula(_mesh, _faceArea, _cotMatrixEntries, _refOmegaList, _refAmpList, initZvals, tarZvals, _refOmegaList[0], _refOmegaList[nFrames + 1], nFrames, 1.0, _quadOrd);
}


