#pragma once
#include "CommonTools.h"

namespace WrinkleFieldsEditor
{
     void editWrinkles(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::MatrixXd& omega, const std::vector<VertexOpInfo>& vertInfo, Eigen::VectorXd& ampNew, Eigen::MatrixXd& omegaNew);
     // amp and omega are vertex based, you can use vertexVec2IntrinsicHalfEdgeVec to convert it into the intrinsic values
     void editWrinklesPerVertex(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertNormals, const Eigen::VectorXd& amp, const Eigen::MatrixXd &omega, const std::vector<VertexOpInfo>& vertInfo, int vid, double& ampNew, Eigen::Vector3d& omegaNew);

     void edgeBasedWrinkleEdition(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::VectorXd& amp, const Eigen::MatrixXd& omega, const std::vector<VertexOpInfo>& vertInfo, Eigen::VectorXd& ampNew, Eigen::MatrixXd& omegaNew);

}