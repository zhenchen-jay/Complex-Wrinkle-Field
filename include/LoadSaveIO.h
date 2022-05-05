#pragma once
#include "CommonTools.h"
#include <iostream>
#include <fstream>


bool loadEdgeOmega(const std::string& filename, const int& nlines, Eigen::VectorXd& edgeOmega);
bool loadVertexZvals(const std::string& filePath, const int& nlines, std::vector<std::complex<double>>& zvals);
bool loadVertexAmp(const std::string& filePath, const int& nlines, Eigen::VectorXd& amp);

bool saveEdgeOmega(const std::string& filename, const Eigen::VectorXd& edgeOmega);
bool saveVertexZvals(const std::string& filePath, const std::vector<std::complex<double>>& zvals);
bool saveVertexAmp(const std::string& filePath, const Eigen::VectorXd& amp);