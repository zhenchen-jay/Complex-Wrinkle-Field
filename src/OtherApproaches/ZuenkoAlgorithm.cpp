#include "../../include/OtherApproaches/ZuenkoAlgorithm.h"
#include "../../include/IntrinsicFormula/InterpolateZvals.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/CommonTools.h"
#include <tbb/tbb.h>
#include <igl/per_vertex_normals.h>

namespace ZuenkoAlg
{
    void spherigonSmoothingSequentially(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertN, Eigen::MatrixXd& upPos, MeshConnectivity& upMesh, Eigen::MatrixXd& upN, int numSubdiv, bool isG1)
    {
        auto curPos = pos;
        auto curMesh = mesh;
        auto curN = vertN;

        for(int n = 0; n < numSubdiv; n++)
        {
            // midpoint
            std::vector<std::pair<int, Eigen::Vector3d>> bary;
            Eigen::MatrixXi upFace;
            meshUpSampling(curPos, curMesh.faces(), upPos, upFace, 1, NULL, NULL, &bary);
            upMesh = MeshConnectivity(upFace);

            if(n > 0)
                igl::per_vertex_normals(curPos, curMesh.faces(), curN);

            int nupverts = bary.size();
            upN.resize(nupverts, 3);
            auto computeNewPos = [&](const tbb::blocked_range<uint32_t>& range)
            {
                for (uint32_t i = range.begin(); i < range.end(); ++i)
//                    for(int i = 0; i < nupverts; i++)
                {
                    //std::cout << i << std::endl;
                    int fid = bary[i].first;
                    Eigen::Vector3d N = Eigen::Vector3d::Zero();
                    Eigen::Vector3d P = N;
                    Eigen::Vector3d weights = Eigen::Vector3d::Zero();
                    for (int j = 0; j < 3; j++)
                    {
                        int oldvid = curMesh.faceVertex(fid, j);
                        N += bary[i].second(j) * curN.row(oldvid);
                        P += bary[i].second(j) * curPos.row(oldvid);
                    }
                    N = N / N.norm();
                    upN.row(i) = N;
                    Eigen::Vector3d Q = Eigen::Vector3d::Zero();
                    std::vector<Eigen::Vector3d> pitilde(3);

                    if (isG1)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            int oldvid = curMesh.faceVertex(fid, j);
                            Eigen::Vector3d Pi = curPos.row(oldvid);
                            pitilde[j] = Pi - (Pi - P).dot(N) * N;
                        }
                    }

                    // compute weight
                    for (int j = 0; j < 3; j++)
                    {
                        weights[j] = bary[i].second(j) * bary[i].second(j);

                        if (isG1)
                        {
                            if (bary[i].second(j) == 1)
                            {
                                weights[j] = 1;
                                continue;
                            }
                            if (bary[i].second(j) == 0)
                            {
                                weights[j] = 0;
                                continue;
                            }

                            double tmp1 = bary[i].second((j + 2) % 3) * bary[i].second((j + 2) % 3) * (pitilde[(j + 2) % 3] - P).squaredNorm();
                            tmp1 /= ((pitilde[(j + 2) % 3] - P).squaredNorm() + (pitilde[j] - P).squaredNorm());

                            double tmp2 = bary[i].second((j + 1) % 3) * bary[i].second((j + 1) % 3) * (pitilde[(j + 1) % 3] - P).squaredNorm();
                            tmp2 /= ((pitilde[(j + 1) % 3] - P).squaredNorm() + (pitilde[j] - P).squaredNorm());

                            weights[j] *= (tmp1 + tmp2);
                        }
                    }

                    // normalize the weights
                    weights /= weights.sum();

                    for (int j = 0; j < 3; j++)
                    {
                        double s = 1;
                        int oldvid = curMesh.faceVertex(fid, j);
                        Eigen::Vector3d Pi = curPos.row(oldvid);
                        Eigen::Vector3d Ni = curN.row(oldvid);
                        Eigen::Vector3d Ki = P + (Pi - P).dot(N) * N;
                        Eigen::Vector3d Qi = Ki + (Pi - Ki).dot(Ni) / (2 + s * (N.dot(Ni) - 1)) * N;
                        Q += weights[j] * Qi;
                    }
                    upPos.row(i) = Q;

                }
            };

            tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nupverts);
            tbb::parallel_for(rangex, computeNewPos);

            if(n < numSubdiv - 1)
            {
                std::swap(curPos, upPos);
                std::swap(curMesh, upMesh);
                std::swap(curN, upN);
            }

        }
    }
    void spherigonSmoothing(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const Eigen::MatrixXd& vertN, const std::vector<std::pair<int, Eigen::Vector3d>>& bary, Eigen::MatrixXd& upPos, Eigen::MatrixXd& upN, bool isG1)
    {
        int nupverts = bary.size();
        upN.resize(nupverts, 3);
        upPos.resize(nupverts, 3);

        auto computeNewPos = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i)
                //for(int i = 0; i < nupverts; i++)
            {
                //std::cout << i << std::endl;
                int fid = bary[i].first;
                Eigen::Vector3d N = Eigen::Vector3d::Zero();
                Eigen::Vector3d P = N;
                Eigen::Vector3d weights = Eigen::Vector3d::Zero();
                for (int j = 0; j < 3; j++)
                {
                    int oldvid = mesh.faceVertex(fid, j);
                    N += bary[i].second(j) * vertN.row(oldvid);
                    P += bary[i].second(j) * pos.row(oldvid);
                }
                N = N / N.norm();
                upN.row(i) = N;
                Eigen::Vector3d Q = Eigen::Vector3d::Zero();
                std::vector<Eigen::Vector3d> pitilde(3);

                if (isG1)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int oldvid = mesh.faceVertex(fid, j);
                        Eigen::Vector3d Pi = pos.row(oldvid);
                        pitilde[j] = Pi - (Pi - P).dot(N) * N;
                    }
                }

                // compute weight
                for (int j = 0; j < 3; j++)
                {
                    weights[j] = bary[i].second(j) * bary[i].second(j);

                    if (isG1)
                    {
                        if (bary[i].second(j) == 1)
                        {
                            weights[j] = 1;
                            continue;
                        }
                        if (bary[i].second(j) == 0)
                        {
                            weights[j] = 0;
                            continue;
                        }

                        double tmp1 = bary[i].second((j + 2) % 3) * bary[i].second((j + 2) % 3) * (pitilde[(j + 2) % 3] - P).squaredNorm();
                        tmp1 /= ((pitilde[(j + 2) % 3] - P).squaredNorm() + (pitilde[j] - P).squaredNorm());

                        double tmp2 = bary[i].second((j + 1) % 3) * bary[i].second((j + 1) % 3) * (pitilde[(j + 1) % 3] - P).squaredNorm();
                        tmp2 /= ((pitilde[(j + 1) % 3] - P).squaredNorm() + (pitilde[j] - P).squaredNorm());

                        weights[j] *= (tmp1 + tmp2);
                    }
                }

                // normalize the weights
                weights /= weights.sum();

                for (int j = 0; j < 3; j++)
                {
                    double s = 1;
                    int oldvid = mesh.faceVertex(fid, j);
                    Eigen::Vector3d Pi = pos.row(oldvid);
                    Eigen::Vector3d Ni = vertN.row(oldvid);
                    Eigen::Vector3d Ki = P + (Pi - P).dot(N) * N;
                    Eigen::Vector3d Qi = Ki + (Pi - Ki).dot(Ni) / (2 + s * (N.dot(Ni) - 1)) * N;
                    Q += weights[j] * Qi;
                }
                upPos.row(i) = Q;

            }
        };

        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nupverts);
        tbb::parallel_for(rangex, computeNewPos);
    }

    void getZuenkoSurfacePerframe(
            const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
            const std::vector<std::complex<double>>& unitZvals, const Eigen::VectorXd& vertAmp, const Eigen::VectorXd& edgeOmega,
            const Eigen::MatrixXd& upsampledV, const Eigen::MatrixXi& upsampledF, const Eigen::MatrixXd& upsampledN,
            const std::vector<std::pair<int, Eigen::Vector3d>>& bary,
            Eigen::MatrixXd& wrinkledV, Eigen::MatrixXi& wrinkledF,
            Eigen::VectorXd& upsampledAmp, Eigen::VectorXd& upsampledPhi, double ampScaling)
    {
        int nupverts = upsampledV.rows();
        wrinkledV = upsampledV;
        wrinkledF = upsampledF;
        upsampledAmp.resize(nupverts);
        upsampledPhi.resize(nupverts);


        auto computePhaseAmp = [&](const tbb::blocked_range<uint32_t>& range) {
            for (uint32_t i = range.begin(); i < range.end(); ++i)
            {
                int fid = bary[i].first;
                std::vector<std::complex<double>> vertzvals(3);
                Eigen::Vector3d edgews;
                upsampledAmp[i] = 0;

                for (int j = 0; j < 3; j++)
                {
                    int vid = baseMesh.faceVertex(fid, j);
                    int eid = baseMesh.faceEdge(fid, j);

                    vertzvals[j] = unitZvals[vid];
                    edgews(j) = edgeOmega(eid); // defined as mesh.edgeVertex(eid, 1) - mesh.edgeVertex(eid, 0)

                    if (baseMesh.edgeVertex(eid, 1) == baseMesh.faceVertex(fid, (j + 1) % 3))
                        edgews(j) *= -1;
                    upsampledAmp[i] += bary[i].second[j] * vertAmp[vid];
                }

                std::complex<double> zval = IntrinsicFormula::getZvalsFromEdgeOmega(bary[i].second, vertzvals, edgews);
                upsampledPhi[i] = std::arg(zval);
                wrinkledV.row(i) += ampScaling * upsampledAmp[i] * std::cos(upsampledPhi[i]) * upsampledN.row(i);
            }
        };

        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nupverts);
        tbb::parallel_for(rangex, computePhaseAmp);
    }

    void getZuenkoSurfaceSequence(
            const Eigen::MatrixXd& baseV, const MeshConnectivity& baseMesh,
            const std::vector<std::complex<double>>& initZvals,
            const std::vector<Eigen::VectorXd>& ampList, const std::vector<Eigen::VectorXd>& omegaList,
            Eigen::MatrixXd& upsampledV, Eigen::MatrixXi& upsampledF,
            std::vector<Eigen::MatrixXd>& wrinkledVList, std::vector<Eigen::MatrixXi>& wrinkledFList,
            std::vector<Eigen::VectorXd>& upsampledAmpList, std::vector<Eigen::VectorXd>& upsampledPhiList,
            int numSubdivs, bool isG1, double ampScaling,
            int innerIter, double blurCoeff)
    {
        int nframes = ampList.size();
        wrinkledVList.resize(nframes);
        wrinkledFList.resize(nframes);
        upsampledAmpList.resize(nframes);
        upsampledPhiList.resize(nframes);

        std::vector<std::pair<int, Eigen::Vector3d>> bary;
        meshUpSampling(baseV, baseMesh.faces(), upsampledV, upsampledF, numSubdivs, NULL, NULL, &bary);

        Eigen::MatrixXd baseN, upsampledN;
        igl::per_vertex_normals(baseV, baseMesh.faces(),  baseN);
        spherigonSmoothing(baseV, baseMesh, baseN, bary, upsampledV, upsampledN, isG1);
        igl::per_vertex_normals(upsampledV, upsampledF,  upsampledN);

        std::vector<std::complex<double>> curZvals = initZvals;
        for (int i = 0; i < curZvals.size(); i++)
        {
            double phi = std::arg(curZvals[i]);
            curZvals[i] = std::complex<double>(std::cos(phi), std::sin(phi));
        }
        // initial frame
        getZuenkoSurfacePerframe(baseV, baseMesh, curZvals, ampList[0], omegaList[0], upsampledV, upsampledF, upsampledN, bary, wrinkledVList[0], wrinkledFList[0], upsampledAmpList[0], upsampledPhiList[0], ampScaling);

        std::vector<std::vector<int>> vertNeiEdges, vertNeiFaces;
        buildVertexNeighboringInfo(baseMesh, int(baseV.rows()), vertNeiEdges, vertNeiFaces);

        // Zuenko's algorithm described in their section 5.1
        for (int i = 1; i < nframes; i++)
        {
            // step 1: apply their equation (10) k times, where k = 5, as they suggested in their paper
            std::vector<std::complex<double>> Phi0 = curZvals;
            std::vector<std::complex<double>> Phi1 = Phi0;
            for (int k = 0; k < innerIter; k++)
            {
                // update the Phi
                for (int v = 0; v < Phi0.size(); v++)
                {
                    std::complex<double> tmpZ = 0;
                    for (auto& e : vertNeiEdges[v]) // all neighboring edges
                    {
                        double deltaTheta = omegaList[i][e];
                        int vj = baseMesh.edgeVertex(e, 0);
                        if (baseMesh.edgeVertex(e, 0) == v)
                        {
                            deltaTheta *= -1;			// make sure deltaTheta = theta_i - theta_j
                            vj = baseMesh.edgeVertex(e, 1);
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
            for (int v = 0; v < Phi0.size(); v++)
            {
                std::complex<double> tmpZ = blurCoeff * curZvals[v] + (1 - blurCoeff) * Phi0[v];
                double theta = std::arg(tmpZ);
                curZvals[v] = std::complex<double>(std::cos(theta), std::sin(theta));
            }

            getZuenkoSurfacePerframe(baseV, baseMesh, curZvals, ampList[i], omegaList[i], upsampledV, upsampledF, upsampledN, bary, wrinkledVList[i], wrinkledFList[i], upsampledAmpList[i], upsampledPhiList[i], ampScaling);
        }
    }
}