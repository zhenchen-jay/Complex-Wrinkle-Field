#pragma once
#include <map>
#include "CommonTools.h"

std::map<std::pair<int, int>, int> he2Edge(const Eigen::MatrixXi& faces);
std::map<std::pair<int, int>, int> he2Edge(const std::vector< std::vector<int>>& edgeToVert);
Eigen::VectorXd swapEdgeVec(const std::vector< std::vector<int>>& edgeToVert, const Eigen::VectorXd& edgeVec, int flag);
Eigen::VectorXd swapEdgeVec(const Eigen::MatrixXi& faces, const Eigen::VectorXd& edgeVec, int flag);
std::vector<std::vector<int>> swapEdgeIndices(const Eigen::MatrixXi& faces, const std::vector<std::vector<int>>& edgeIndices, int flag);
Eigen::MatrixXd edgeVec2FaceVec(const Mesh& mesh, Eigen::VectorXd& edgeVec);