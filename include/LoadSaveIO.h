#pragma once
#include "CommonTools.h"
#include <iostream>
#include <fstream>


bool loadEdgeOmega(const std::string& filename, const int& nlines, Eigen::VectorXd& edgeOmega);
bool loadVertexZvals(const std::string& filePath, const int& nlines, std::vector<std::complex<double>>& zvals);
bool loadVertexAmp(const std::string& filePath, const int& nlines, Eigen::VectorXd& amp);