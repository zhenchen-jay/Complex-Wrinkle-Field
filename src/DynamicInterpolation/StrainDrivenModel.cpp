#include <iostream>
#include "../../include/DynamicInterpolation/StrainDrivenModel.h"
#include "../../include/IntrinsicFormula/KnoppelStripePattern.h"
#include "../../include/CommonTools.h"

void StrainDrivenModel::convertList2Variable(Eigen::VectorXd& x)
{
    int nbaseVerts = _basePos.rows();
    int numFrames = _vertValsList.size() - 2;

    int DOFs = numFrames * nbaseVerts * 4;

    x.setZero(DOFs);

    for (int i = 0; i < numFrames; i++)
    {
        for (int j = 0; j < nbaseVerts; j++)
        {
            x(i * 4 * nbaseVerts + 2 * j) = _vertValsList[i + 1][j].real();
            x(i * 4 * nbaseVerts + 2 * j + 1) = _vertValsList[i + 1][j].imag();

            x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j) = _wList[i + 1](j, 0);
            x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j + 1) = _wList[i + 1](j, 1);
        }
    }
}

void StrainDrivenModel::convertVariable2List(const Eigen::VectorXd& x)
{
    int nbaseVerts = _basePos.rows();
    int numFrames = _vertValsList.size() - 2;

    for (int i = 0; i < numFrames; i++)
    {
        for (int j = 0; j < nbaseVerts; j++)
        {
            _vertValsList[i + 1][j] = std::complex<double>(x(i * 4 * nbaseVerts + 2 * j), x(i * 4 * nbaseVerts + 2 * j + 1));
            _wList[i + 1](j, 0) = x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j);
            _wList[i + 1](j, 1) = x(i * 4 * nbaseVerts + 2 * nbaseVerts + 2 * j + 1);
        }
    }
}


Eigen::VectorXd StrainDrivenModel::getEntries(const std::vector<std::complex<double>>& zvals, int entryId)
{
    if (entryId != 0 && entryId != 1)
    {
        std::cerr << "Error in get entry!" << std::endl;
        exit(1);
    }
    int size = zvals.size();
    Eigen::VectorXd vals(size);
    for (int i = 0; i < size; i++)
    {
        if (entryId == 0)
            vals(i) = zvals[i].real();
        else
            vals(i) = zvals[i].imag();
    }
    return vals;
}

double StrainDrivenModel::computeSpatialEnergyPerFramePerVertex(int frameId, int vertId, Eigen::Vector4d *deriv,
                                                         Eigen::Matrix4d *hess, bool isProj) {
    double amp = std::abs(_vertValsList[frameId][vertId]);
    Eigen::Vector2d omega = _wList[frameId].row(vertId);

    Eigen::Vector2d compress = _compressDir[frameId].row(vertId);
    Eigen::Vector2d omegaTar = _wTar[frameId].row(vertId);
    double ampTar = _aTar[frameId](vertId);

    double energy = 0;

    if(_modelFlag == 0)
    {
        // fake stretching
        Eigen::Vector2d diff = amp * omega - compress;
        energy = 0.5 * diff.dot(diff);

        // fake bending to prevent zero amp
        energy += _fakeThickness * 0.5 * amp * omega.dot(omega);

        if(deriv || hess)
        {
            Eigen::Vector2d gradAmp;
            Eigen::Matrix2d gradOmega = Eigen::Matrix2d::Identity();

            gradAmp << _vertValsList[frameId][vertId].real() / amp, _vertValsList[frameId][vertId].imag() / amp;
            Eigen::Matrix<double, 2, 4> gradDiff;
            gradDiff.block<2, 2>(0, 0) = omega * gradAmp.transpose();
            gradDiff.block<2, 2>(0, 2) = amp * gradOmega;

            if(deriv)
            {
                // fake stretching
                (*deriv) = gradDiff.transpose() * diff;

                // fake bending
                deriv->segment<2>(0) += _fakeThickness * 0.5 * omega.dot(omega) * gradAmp;
                deriv->segment<2>(2) += _fakeThickness * amp * omega.transpose() * gradOmega;
            }


            if(hess)
            {
                // fake stretching
                (*hess) = gradDiff.transpose() * gradDiff;
                std::vector<Eigen::Matrix4d> hessDiff(2, Eigen::Matrix4d::Zero());

                Eigen::Matrix2d hessAmp;
                double ampCube = std::pow(amp, 3);
                hessAmp << 1 / amp - _vertValsList[frameId][vertId].real() * _vertValsList[frameId][vertId].real() / ampCube, -_vertValsList[frameId][vertId].real() * _vertValsList[frameId][vertId].imag() / ampCube, -_vertValsList[frameId][vertId].real() * _vertValsList[frameId][vertId].imag() / ampCube,  1 / amp - _vertValsList[frameId][vertId].imag() * _vertValsList[frameId][vertId].imag() / ampCube;

                hessDiff[0].block<2, 2>(0, 0) = omega(0) * hessAmp;
                hessDiff[0].block<2, 2>(0, 2) = gradAmp * gradOmega.col(0).transpose();
                hessDiff[0].block<2, 2>(2, 0) = gradOmega.col(0) * gradAmp.transpose();

                hessDiff[1].block<2, 2>(0, 0) = omega(1) * hessAmp;
                hessDiff[1].block<2, 2>(0, 2) = gradAmp * gradOmega.col(1).transpose();
                hessDiff[1].block<2, 2>(2, 0) = gradOmega.col(1) * gradAmp.transpose();

                (*hess) += diff(0) * hessDiff[0] + diff(1) * hessDiff[1];

                // fake bending
                hess->block<2, 2>(0, 0) += _fakeThickness * 0.5 * omega.dot(omega) * hessAmp;
                hess->block<2, 2>(0, 2) += _fakeThickness * gradAmp * (omega.transpose() * gradOmega);
                hess->block<2, 2>(2, 0) += _fakeThickness * (omega.transpose() * gradOmega).transpose() * gradAmp.transpose();
                hess->block<2, 2>(2, 2) += _fakeThickness * amp * (gradOmega.transpose() * gradOmega);

                if(isProj)
                    (*hess) = SPDProjection((*hess));

            }
        }
    }
    else
    {
        double ampSq = amp * amp;
        double ampTarSq = ampTar * ampTar;
        energy = 0.5 * (ampSq - ampTarSq) * (ampSq - ampTarSq) + 0.5 * _fakeThickness * (omega - omegaTar).dot(omega - omegaTar);
        // we use amp^2 to avoid zero amp issue, use _fakeThickness to enlarge the influence of amplitude part
        if(deriv)
            deriv->setZero();
        if(hess)
            hess->setZero();

        if(deriv || hess)
        {
            Eigen::Vector2d gradAmpSq;
            Eigen::Matrix2d gradOmega = Eigen::Matrix2d::Identity();

            gradAmpSq << 2.0 * _vertValsList[frameId][vertId].real(), 2.0 * _vertValsList[frameId][vertId].imag();

            if(deriv)
            {
                deriv->segment<2>(0) = (ampSq - ampTarSq) * gradAmpSq;
                deriv->segment<2>(2) = _fakeThickness * (omega - omegaTar).transpose() * gradOmega;
            }

            if(hess)
            {
                Eigen::Matrix2d hessAmpSq = Eigen::Matrix2d::Identity() * 2;
                hess->block<2, 2>(0, 0) = ((ampSq - ampTarSq) * hessAmpSq + gradAmpSq * gradAmpSq.transpose());
                if(isProj)
                    hess->block<2, 2>(0, 0) = SPDProjection(hess->block<2, 2>(0, 0));
                hess->block<2, 2>(2, 2) = _fakeThickness * gradOmega.transpose() * gradOmega;
            }
        }
    }
    return energy;
}

double StrainDrivenModel::computeSpatialEnergyPerFrame(int frameId, Eigen::VectorXd *deriv,
                                                std::vector<Eigen::Triplet<double>> *hessT, bool isProj){
    int nverts = _basePos.rows();
    std::vector<double> energyList(nverts);
    std::vector<Eigen::Vector4d> derivList(nverts);
    std::vector<Eigen::Matrix4d> hessList(nverts);

//    auto computeEnergyPerVert = [&](const tbb::blocked_range<uint32_t>& range) {
//        for (uint32_t i = range.begin(); i < range.end(); ++i)
//        {
//            energyList[i] = computeSpatialEnergyPerFramePerVertex(frameId, i, deriv ?  &(derivList[i]) : NULL, hessT? &(hessList[i]) : NULL, isProj);
//        }
//    };
//
//    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nverts, GRAIN_SIZE);
//    tbb::parallel_for(rangex, computeEnergyPerVert);

    Eigen::VectorXd x(nverts), y(nverts);
    for(int i = 0; i < nverts; i++)
    {
        energyList[i] = computeSpatialEnergyPerFramePerVertex(frameId, i, deriv ?  &(derivList[i]) : NULL, hessT? &(hessList[i]) : NULL, isProj);
//        x(i) = _vertValsList[frameId][i].real();
//        y(i) = _vertValsList[frameId][i].imag();
    }

    if(deriv)
        deriv->setZero(4 * nverts);

    if(hessT)
        hessT->clear();

    double energy = 0;
//    energy -= 0.5 * x.dot(_cotMat * x) + 0.5 * y.dot(_cotMat * y);
//    Eigen::VectorXd lapDerivx( nverts), lapDerivy(nverts);
//    lapDerivx = -_cotMat * x;
//    lapDerivy = -_cotMat * y;

    Eigen::MatrixXd halfEdgewTar;
    Eigen::MatrixXd extendedW(nverts, 3);
    extendedW.setZero();

    extendedW.block(0, 0, nverts, 2) = _wTar[frameId];
    halfEdgewTar = vertexVec2IntrinsicHalfEdgeVec(extendedW,_basePos, _baseMesh);
    Eigen::VectorXd kDeriv;
    std::vector<Eigen::Triplet<double>> kT;

    double knoppel = IntrinsicFormula::KnoppelEnergy(_baseMesh, halfEdgewTar, _faceArea, _cotEntries, _vertValsList[frameId], deriv ? &kDeriv : NULL, hessT ? &kT : NULL);
    energy += knoppel;

    if(deriv)
        deriv->segment(0, 2 * nverts) += kDeriv;

    for(int i = 0; i < nverts; i++)
    {
        energy += energyList[i];
        if(deriv)
        {
            deriv->segment<2>(2 * i) += derivList[i].segment<2>(0);
            deriv->segment<2>(2 * i + 2 * nverts) += derivList[i].segment<2>(2);
//            (*deriv)(2 * i) += lapDerivx(i);
//            (*deriv)(2 * i + 1) += lapDerivy(i);
        }
        if(hessT)
        {
            for(int j = 0; j < 2; j++)
                for(int k = 0; k < 2; k++)
                {
                    hessT->push_back({2 * i + j, 2 * i + k , hessList[i](j, k)});
                    hessT->push_back({2 * i + j + 2 * nverts, 2 * i + k , hessList[i](2 + j, k)});
                    hessT->push_back({2 * i + j, 2 * i + k+ 2 * nverts , hessList[i](j, 2 + k)});
                    hessT->push_back({2 * i + j + 2 * nverts, 2 * i + k + 2 * nverts, hessList[i](2 + j, 2 + k)});
                }

        }
    }

    if(hessT)
    {
//        for (int k=0; k<_cotMat.outerSize(); ++k)
//            for (Eigen::SparseMatrix<double>::InnerIterator it(_cotMat,k); it; ++it)
//            {
//                hessT->push_back({2 * it.row(), 2 * it.col(), -it.value()});
//                hessT->push_back({2 * it.row() + 1, 2 * it.col() + 1, -it.value()});
//            }

        for(auto& t: kT)
            hessT->push_back(t);

    }

    return energy;
}

double StrainDrivenModel::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
    int nbaseVerts = _basePos.rows();
    int numFrames = _vertValsList.size() - 2;
    int DOFs = numFrames * nbaseVerts * 4;

    convertVariable2List(x);

    Eigen::VectorXd curDeriv;
    std::vector<Eigen::Triplet<double>> T, curT;

    Eigen::VectorXd spatialDeriv;
    std::vector<Eigen::Triplet<double>> spatialHessT;

    double energy = 0;
    if (deriv)
    {
        deriv->setZero(DOFs);
    }

    for (int i = 0; i < _vertValsList.size() - 1; i++)
    {
        energy += _model.zDotSquareIntegration(_wList[i], _wList[i + 1], _vertValsList[i], _vertValsList[i + 1], _dt, deriv ? &curDeriv : NULL, hess ? &curT : NULL, isProj);

        if(i > 0 && _spatialRate > 0)
        {
            energy += _spatialRate * computeSpatialEnergyPerFrame(i, deriv ? &spatialDeriv : NULL, hess ? &spatialHessT : NULL, isProj);
        }

        if (deriv)
        {
            if(i == 0)
                deriv->segment(0, 4 * nbaseVerts) += curDeriv.segment(4 * nbaseVerts, 4 * nbaseVerts);
            else if(i == _vertValsList.size() - 2)
                deriv->segment(4 * (i - 1) * nbaseVerts, 4 * nbaseVerts) += curDeriv.segment(0, 4 * nbaseVerts);
            else
            {
                deriv->segment(4 * (i - 1) * nbaseVerts, 8 * nbaseVerts) += curDeriv;
            }

            if(i > 0 && _spatialRate > 0)
            {
                deriv->segment(4 * (i - 1) * nbaseVerts, 4 * nbaseVerts) += _spatialRate * spatialDeriv;
            }
        }

        if (hess)
        {
            for (auto& it : curT)
            {
                if (i == 0)
                {
                    if (it.row() >= 4 * nbaseVerts && it.col() >= 4 * nbaseVerts)
                        T.push_back({ it.row() - 4 * nbaseVerts, it.col() - 4 * nbaseVerts, it.value() });
                }
                else if (i == _vertValsList.size() - 2)
                {
                    if (it.row() < 4 * nbaseVerts && it.col() < 4 * nbaseVerts)
                        T.push_back({ it.row() + 4 * (i - 1) * nbaseVerts, it.col() + 4 * (i - 1) * nbaseVerts, it.value() });
                }
                else
                {
                    T.push_back({ it.row() + 4 * (i - 1) * nbaseVerts, it.col() + 4 * (i - 1) * nbaseVerts, it.value() });
                }
            }
            curT.clear();

            if(i > 0 && _spatialRate > 0)
            {
                for (auto& it : spatialHessT)
                    T.push_back({ it.row() + 4 * (i - 1) * nbaseVerts, it.col() + 4 * (i - 1) * nbaseVerts, it.value() * _spatialRate});
            }
            spatialHessT.clear();
        }
    }
    if (hess)
    {
        //std::cout << "num of triplets: " << T.size() << std::endl;
        hess->resize(DOFs, DOFs);
        hess->setFromTriplets(T.begin(), T.end());
    }


    return energy;
}

void StrainDrivenModel::testSpatialEnergyPerFramePerVertex(int frameId, int vertId) {
    double energy;
    Eigen::Vector4d deriv;
    Eigen::Matrix4d hess;

    int nverts = _basePos.rows();
    Eigen::Vector4d backupframe;

    backupframe(0) = _vertValsList[frameId][vertId].real();
    backupframe(1) = _vertValsList[frameId][vertId].imag();
    backupframe(2) = _wList[frameId](vertId, 0);
    backupframe(3) = _wList[frameId](vertId, 1);

    energy = computeSpatialEnergyPerFramePerVertex(frameId, vertId, &deriv, &hess, false);

    Eigen::Vector4d randDir = deriv;
    randDir.setRandom();

    std::cout << "deriv: " << deriv << std::endl;
    std::cout << "hess:\n" << hess << std::endl;

    for(int i = 3; i < 9; i++) {
        double eps = std::pow(0.1, i);
        _vertValsList[frameId][vertId] = std::complex<double>(backupframe(0) + eps * randDir(0),
                                                         backupframe(1) + eps * randDir(1));
        _wList[frameId](vertId, 0) = backupframe(2) + eps * randDir(2);
        _wList[frameId](vertId, 1) = backupframe(3) + eps * randDir(3);
        Eigen::Vector4d deriv1;
        double energy1 = computeSpatialEnergyPerFramePerVertex(frameId, vertId, &deriv1, NULL, false);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (energy1 - energy) / eps - randDir.dot(deriv) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * randDir).norm() << std::endl;
    }

}

void StrainDrivenModel::testSpatialEnergyPerFrame(int frameId) {
    int nverts = _basePos.rows();
    Eigen::VectorXd backupframe(4 * nverts);
    for(int i = 0; i < nverts; i++)
    {
        backupframe(2 * i) = _vertValsList[frameId][i].real();
        backupframe(2 * i + 1) = _vertValsList[frameId][i].imag();
        backupframe(2 * nverts + 2 * i) = _wList[frameId](i,0);
        backupframe(2 * nverts + 2 * i + 1) = _wList[frameId](i,1);
    }

    Eigen::VectorXd deriv;
    Eigen::SparseMatrix<double> hess;
    std::vector<Eigen::Triplet<double>> T;
    double energy = computeSpatialEnergyPerFrame(frameId, &deriv, &T, false);
    hess.resize(deriv.size(), deriv.size());
    hess.setFromTriplets(T.begin(), T.end());

    Eigen::VectorXd randDir = deriv;
    randDir.setRandom();

    for(int j = 3; j < 9; j++)
    {
        double eps = std::pow(0.1, j);
        for(int i = 0; i < nverts; i++)
        {
            _vertValsList[frameId][i] = std::complex<double>(backupframe(2 * i)  + eps * randDir(2 * i), backupframe(2 * i + 1)  + eps * randDir(2 * i + 1));
            _wList[frameId](i,0) = backupframe(2 * nverts + 2 * i) + eps * randDir(2 * nverts + 2 * i);
            _wList[frameId](i,1) = backupframe(2 * nverts + 2 * i + 1) + eps * randDir(2 * nverts + 2 * i + 1);
        }
        Eigen::VectorXd deriv1;
        double energy1 = computeSpatialEnergyPerFrame(frameId, &deriv1, NULL, false);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (energy1 - energy) / eps - randDir.dot(deriv) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * randDir).norm() << std::endl;
    }
}

void StrainDrivenModel::testEnergy(Eigen::VectorXd x)
{
    Eigen::VectorXd deriv;
    Eigen::SparseMatrix<double> hess;

    double e = computeEnergy(x, &deriv, &hess, false);
    std::cout << "energy: " << e << std::endl;

    Eigen::VectorXd dir = deriv;
    dir.setRandom();

    for (int i = 3; i < 9; i++)
    {
        double eps = std::pow(0.1, i);

        Eigen::VectorXd deriv1;
        double e1 = computeEnergy(x + eps * dir, &deriv1, NULL, false);

        std::cout << "eps: " << eps << std::endl;
        std::cout << "value-gradient check: " << (e1 - e) / eps - dir.dot(deriv) << std::endl;
        std::cout << "gradient-hessian check: " << ((deriv1 - deriv) / eps - hess * dir).norm() << std::endl;
    }
}
