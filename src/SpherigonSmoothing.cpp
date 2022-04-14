#include "../include/SpherigonSmoothing.h"
#include <iostream>

void spherigonSmoothing(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertN, const std::vector<std::pair<int, Eigen::Vector3d>> bary, Eigen::MatrixXd& upPos, Eigen::MatrixXd& upN)
{
	int nupverts = bary.size();
	upN.resize(nupverts, 3);
	upPos.resize(nupverts, 3);

	auto computeNewPos = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		//for(int i = 0; i < nupverts; i++)
		{
			int fid = bary[i].first;
			Eigen::Vector3d N = Eigen::Vector3d::Zero();
			Eigen::Vector3d P = N;
			double sum = 0;
			for (int j = 0; j < 3; j++)
			{
				int oldvid = mesh.faceVertex(fid, j);
				N += bary[i].second(j) * vertN.row(oldvid);
				P += bary[i].second(j) * pos.row(oldvid);
				sum += bary[i].second(j) * bary[i].second(j);
			}
			N = N / N.norm();
			upN.row(i) = N;
			Eigen::Vector3d Q = Eigen::Vector3d::Zero();

			for (int j = 0; j < 3; j++)
			{
				double s = 1;
				int oldvid = mesh.faceVertex(fid, j);
				Eigen::Vector3d Pi = pos.row(oldvid);
				Eigen::Vector3d Ni = vertN.row(oldvid);
				Eigen::Vector3d Ki = P + (Pi - P).dot(N) * N;
				Eigen::Vector3d Qi = Ki + (Pi - Ki).dot(Ni) / (2 + s * (N.dot(Ni) - 1)) * N;
				Q += bary[i].second(j) * bary[i].second(j) / sum * Qi;
			}
			upPos.row(i) = Q;
			
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nupverts, GRAIN_SIZE);
	tbb::parallel_for(rangex, computeNewPos);
}