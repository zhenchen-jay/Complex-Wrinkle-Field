#include "../../include/IntrinsicFormula/WrinkleEditingCWF.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include <igl/cotmatrix_entries.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/boundary_loop.h>
#include <Eigen/SPQRSupport>
#include <unordered_set>
using namespace IntrinsicFormula;

void WrinkleEditingCWF::convertList2Variable(Eigen::VectorXd& x)
{
    int nverts = _pos.rows();
    int nedges = _mesh.nEdges();

    int numFrames = _unitZvalsList.size() - 2;

    int DOFsPerframe = 2 * nverts;

    int DOFs = numFrames * DOFsPerframe;

    x.setZero(DOFs);

    for (int i = 0; i < numFrames; i++)
    {
        for (int j = 0; j < nverts; j++)
        {
            x(i * DOFsPerframe + 2 * j) = _unitZvalsList[i + 1][j].real();
            x(i * DOFsPerframe + 2 * j + 1) = _unitZvalsList[i + 1][j].imag();
        }
    }
}

void WrinkleEditingCWF::convertVariable2List(const Eigen::VectorXd& x)
{
    int nverts = _pos.rows();

    int numFrames = _unitZvalsList.size() - 2;

    int DOFsPerframe = 2 * nverts;

    for (int i = 0; i < numFrames; i++)
    {
        for (int j = 0; j < nverts; j++)
        {
            _unitZvalsList[i + 1][j] = std::complex<double>(x(i * DOFsPerframe + 2 * j), x(i * DOFsPerframe + 2 * j + 1));
        }
    }
}

void WrinkleEditingCWF::computeAmpSqOmegaQuaticAverage()
{
    int numFrames = _unitZvalsList.size();
    _ampSqOmegaQauticAverageList.resize(numFrames, 0);
    _ampSqOmegaQuaticAverage = 0;
    std::unordered_set<int> unorderedSet = {};
    double activeArea = 0;
    if(!_selectedVids.size())
    {
        activeArea = _vertArea.sum();
    }
    else
        unorderedSet = std::unordered_set<int>(_unselectedVids.begin(), _unselectedVids.end());
    for (int i = 0; i < numFrames; i++)
    {
        int nverts = _pos.rows();
        for (int j = 0; j < nverts; j++)
        {
            if(unorderedSet.find(j) == unorderedSet.end())
            {
                _ampSqOmegaQuaticAverage +=
                        _ampTimesOmegaSq[i][j] * _ampTimesOmegaSq[i][j] * _vertArea(j) / numFrames;
                _ampSqOmegaQauticAverageList[i] +=
                        _ampTimesOmegaSq[i][j] * _ampTimesOmegaSq[i][j] * _vertArea(j);
                if(i == 0)
                    activeArea += _vertArea(j);
            }
        }
        _ampSqOmegaQauticAverageList[i] /= activeArea;
    }
    _ampSqOmegaQuaticAverage /= activeArea;
}

double WrinkleEditingCWF::temporalAmpDifference(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
    int nverts = _pos.rows();
    double energy = 0;

    if (deriv)
        deriv->setZero(2 * nverts);
    if (hessT)
        hessT->clear();

	double dt = 1. / (_unitZvalsList.size() - 1);

    for (int vid = 0; vid < nverts; vid++)
    {
        double ampSq = _unitZvalsList[frameId][vid].real() * _unitZvalsList[frameId][vid].real() +
                       _unitZvalsList[frameId][vid].imag() * _unitZvalsList[frameId][vid].imag();
        double refAmpSq = 1;
        // double ca = _spatialAmpRatio * _vertArea(vid) * dt *_ampTimesOmegaSq[frameId][vid] * _ampTimesOmegaSq[frameId][vid] / _ampSqOmegaQuaticAverage;
        double cf = (_ampTimesOmegaSq[0][vid] * _ampTimesOmegaSq[0][vid] + _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid] * _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid]) / 2;
        cf = 1;
        double ca = _spatialAmpRatio * _vertArea(vid) * dt * cf;

        energy += ca * (ampSq - refAmpSq) * (ampSq - refAmpSq);

        if (deriv)
        {
            (*deriv)(2 * vid) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _unitZvalsList[frameId][vid].real());
            (*deriv)(2 * vid + 1) += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * _unitZvalsList[frameId][vid].imag());
        }

        if (hessT)
        {
            Eigen::Matrix2d tmpHess;
            tmpHess <<
                    2.0 * _unitZvalsList[frameId][vid].real() * 2.0 * _unitZvalsList[frameId][vid].real(),
                    2.0 * _unitZvalsList[frameId][vid].real() * 2.0 * _unitZvalsList[frameId][vid].imag(),
                    2.0 * _unitZvalsList[frameId][vid].real() * 2.0 * _unitZvalsList[frameId][vid].imag(),
                    2.0 * _unitZvalsList[frameId][vid].imag() * 2.0 * _unitZvalsList[frameId][vid].imag();

            tmpHess *= 2.0 * ca;
            tmpHess += 2.0 * ca * (ampSq - refAmpSq) * (2.0 * Eigen::Matrix2d::Identity());


            if (isProj)
                tmpHess = SPDProjection(tmpHess);

            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++)
                    hessT->push_back({ 2 * vid + k, 2 * vid + l, tmpHess(k, l) });
        }
    }
    return energy;
}


double WrinkleEditingCWF::spatialKnoppelEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
    double energy = 0;
    int nedges = _mesh.nEdges();
    int nverts = _pos.rows();
    std::vector<Eigen::Triplet<double>> AT;
    AT.clear();
    double dt = 1. / (_unitZvalsList.size() - 1);
    for (int eid = 0; eid < nedges; eid++)
    {
        int vid0 = _mesh.edgeVertex(eid, 0);
        int vid1 = _mesh.edgeVertex(eid, 1);

        double r0 = 1;
        double r1 = 1;

        std::complex<double> expw0 = std::complex<double>(std::cos(_edgeOmegaList[frameId](eid)), std::sin(_edgeOmegaList[frameId](eid)));

        std::complex<double> z0 = _unitZvalsList[frameId][vid0];
        std::complex<double> z1 = _unitZvalsList[frameId][vid1];

        double cf = (_ampTimesOmegaSq[0][vid0] * _ampTimesOmegaSq[0][vid0] + _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid0] * _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid0]) / 2;
        cf += (_ampTimesOmegaSq[0][vid1] * _ampTimesOmegaSq[0][vid1] + _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid1] * _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid1]) / 2;

        cf = 2;
        double ce = _spatialKnoppelRatio * _edgeArea(eid) * dt * cf / 2;

        energy += 0.5 * norm((r1 * z0 * expw0 - r0 * z1)) * ce;

        if (deriv || hessT)
        {
            AT.push_back({ 2 * vid0, 2 * vid0, r1 * r1 * ce });
            AT.push_back({ 2 * vid0 + 1, 2 * vid0 + 1, r1 * r1 * ce });

            AT.push_back({ 2 * vid1, 2 * vid1, r0 * r0 * ce });
            AT.push_back({ 2 * vid1 + 1, 2 * vid1 + 1, r0 * r0 * ce });


            AT.push_back({ 2 * vid0, 2 * vid1, -ce * (expw0.real()) * r0 * r1 });
            AT.push_back({ 2 * vid0 + 1, 2 * vid1, -ce * (-expw0.imag()) * r0 * r1 });
            AT.push_back({ 2 * vid0, 2 * vid1 + 1, -ce * (expw0.imag()) * r0 * r1 });
            AT.push_back({ 2 * vid0 + 1, 2 * vid1 + 1, -ce * (expw0.real()) * r0 * r1 });

            AT.push_back({ 2 * vid1, 2 * vid0, -ce * (expw0.real()) * r0 * r1 });
            AT.push_back({ 2 * vid1, 2 * vid0 + 1, -ce * (-expw0.imag()) * r0 * r1 });
            AT.push_back({ 2 * vid1 + 1, 2 * vid0, -ce * (expw0.imag()) * r0 * r1 });
            AT.push_back({ 2 * vid1 + 1, 2 * vid0 + 1, -ce * (expw0.real()) * r0 * r1 });
        }
    }

    if (deriv || hessT)
    {
        Eigen::SparseMatrix<double> A;

        A.resize(2 * nverts, 2 * nverts);
        A.setFromTriplets(AT.begin(), AT.end());

        // check whether A is PD


        if (deriv)
        {
            Eigen::VectorXd fvals(2 * nverts);
            for (int i = 0; i < nverts; i++)
            {
                fvals(2 * i) = _unitZvalsList[frameId][i].real();
                fvals(2 * i + 1) = _unitZvalsList[frameId][i].imag();
            }
            (*deriv) = A * fvals;
        }

        if (hessT)
            (*hessT) = AT;
    }

    return energy;
}


double WrinkleEditingCWF::kineticEnergy(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
    int nverts = _pos.rows();
    double dt = 1. / (_unitZvalsList.size() - 1);
    double energy = 0;

    int DOFsPerframe = 2 * nverts;

    if (deriv)
        deriv->setZero(4 * nverts);

    for (int vid = 0; vid < nverts; vid++)
    {
        Eigen::Vector2d diff;
    //    double coeff = _vertWeight(vid) / (dt * dt) * _vertArea[vid] * _ampTimesOmegaSq[frameId][vid] * _ampTimesOmegaSq[frameId][vid] / _ampSqOmegaQuaticAverage * dt;
        double coeff = (_ampTimesOmegaSq[0][vid] * _ampTimesOmegaSq[0][vid] + _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid] * _ampTimesOmegaSq[_unitZvalsList.size() - 1][vid]) / 2;
        coeff /= _ampSqOmegaQuaticAverage;
        coeff *= _vertWeight(vid) / (dt * dt) * _vertArea[vid] * dt;
        diff << (_unitZvalsList[frameId + 1][vid] - _unitZvalsList[frameId][vid]).real(), (_unitZvalsList[frameId + 1][vid] - _unitZvalsList[frameId][vid]).imag();
        energy += 0.5 * coeff * diff.squaredNorm();

        if (deriv)
        {
            deriv->segment<2>(2 * vid) += -coeff * diff;
            deriv->segment<2>(2 * vid + DOFsPerframe) += coeff * diff;
        }

        if (hessT)
        {
            hessT->push_back({ 2 * vid, 2 * vid, coeff });
            hessT->push_back({ 2 * vid, DOFsPerframe + 2 * vid, -coeff });

            hessT->push_back({ 2 * vid + 1, 2 * vid + 1, coeff });
            hessT->push_back({ 2 * vid + 1, DOFsPerframe + 2 * vid + 1, -coeff });

            hessT->push_back({ DOFsPerframe + 2 * vid, DOFsPerframe + 2 * vid, coeff });
            hessT->push_back({ DOFsPerframe + 2 * vid, 2 * vid, -coeff });

            hessT->push_back({ DOFsPerframe + 2 * vid + 1, DOFsPerframe + 2 * vid + 1, coeff });
            hessT->push_back({ DOFsPerframe + 2 * vid + 1, 2 * vid + 1, -coeff });

        }
    }
    return energy;
}

double WrinkleEditingCWF::kineticEnergyWithoutFSq(int frameId, Eigen::VectorXd* deriv, std::vector<Eigen::Triplet<double>>* hessT, bool isProj)
{
    int nverts = _pos.rows();
    double dt = 1. / (_unitZvalsList.size() - 1);
    double energy = 0;

    int DOFsPerframe = 2 * nverts;

    if (deriv)
        deriv->setZero(4 * nverts);

    for (int vid = 0; vid < nverts; vid++)
    {
        Eigen::Vector2d diff;
        double coeff = _vertWeight(vid) / (dt * dt) * _vertArea[vid] * dt;
        diff << (_unitZvalsList[frameId + 1][vid] - _unitZvalsList[frameId][vid]).real(), (_unitZvalsList[frameId + 1][vid] - _unitZvalsList[frameId][vid]).imag();
        energy += 0.5 * coeff * diff.squaredNorm();

        if (deriv)
        {
            deriv->segment<2>(2 * vid) += -coeff * diff;
            deriv->segment<2>(2 * vid + DOFsPerframe) += coeff * diff;
        }

        if (hessT)
        {
            hessT->push_back({ 2 * vid, 2 * vid, coeff });
            hessT->push_back({ 2 * vid, DOFsPerframe + 2 * vid, -coeff });

            hessT->push_back({ 2 * vid + 1, 2 * vid + 1, coeff });
            hessT->push_back({ 2 * vid + 1, DOFsPerframe + 2 * vid + 1, -coeff });

            hessT->push_back({ DOFsPerframe + 2 * vid, DOFsPerframe + 2 * vid, coeff });
            hessT->push_back({ DOFsPerframe + 2 * vid, 2 * vid, -coeff });

            hessT->push_back({ DOFsPerframe + 2 * vid + 1, DOFsPerframe + 2 * vid + 1, coeff });
            hessT->push_back({ DOFsPerframe + 2 * vid + 1, 2 * vid + 1, -coeff });

        }
    }
    return energy;
}

double WrinkleEditingCWF::computeEnergy(const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
{
    int nverts = _pos.rows();

    int numFrames = _unitZvalsList.size() - 2;

    int DOFsPerframe = 2 * nverts;

    int DOFs = numFrames * DOFsPerframe;

    convertVariable2List(x);

    Eigen::VectorXd curDeriv;
    std::vector<Eigen::Triplet<double>> T, curT;

    double energy = 0;
    if (deriv)
    {
        deriv->setZero(DOFs);
    }

    std::vector<Eigen::VectorXd> curKDerivList(numFrames + 1);
    std::vector<std::vector<Eigen::Triplet<double>>> curKTList(numFrames + 1);
    std::vector<double> keList(numFrames + 1);

    auto kineticEnergyPerframe = [&](const tbb::blocked_range<uint32_t>& range)
    {
        for (uint32_t i = range.begin(); i < range.end(); ++i)
        {
            keList[i] = kineticEnergy(i, deriv ? &curKDerivList[i] : NULL, hess ? &curKTList[i] : NULL, isProj);
        }
    };

    tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)numFrames + 1, GRAIN_SIZE);
    tbb::parallel_for(rangex, kineticEnergyPerframe);

    double ke = 0;
    for (int i = 0; i < _unitZvalsList.size() - 1; i++)
    {
        ke += keList[i];

        if (deriv)
        {
            if (i == 0)
                deriv->segment(0, DOFsPerframe) += curKDerivList[i].segment(DOFsPerframe, DOFsPerframe);
            else if (i == _unitZvalsList.size() - 2)
                deriv->segment((i - 1) * DOFsPerframe, DOFsPerframe) += curKDerivList[i].segment(0, DOFsPerframe);
            else
            {
                deriv->segment((i - 1) * DOFsPerframe, 2 * DOFsPerframe) += curKDerivList[i];
            }
        }

        if (hess)
        {
            for (auto& it : curKTList[i])
            {

                if (i == 0)
                {
                    if (it.row() >= DOFsPerframe && it.col() >= DOFsPerframe)
                        T.push_back({ it.row() - DOFsPerframe, it.col() - DOFsPerframe, it.value() });
                }
                else if (i == _unitZvalsList.size() - 2)
                {
                    if (it.row() < DOFsPerframe && it.col() < DOFsPerframe)
                        T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
                }
                else
                {
                    T.push_back({ it.row() + (i - 1) * DOFsPerframe, it.col() + (i - 1) * DOFsPerframe, it.value() });
                }


            }
        }
    }
    energy += ke;

    
    std::vector<Eigen::VectorXd> ampDerivList(numFrames), knoppelDerivList(numFrames);
    std::vector<std::vector<Eigen::Triplet<double>>> ampTList(numFrames), knoppelTList(numFrames);
    std::vector<double> ampEnergyList(numFrames), knoppelEnergyList(numFrames);

    auto otherEnergiesPerframe = [&](const tbb::blocked_range<uint32_t>& range)
    {
        for (uint32_t i = range.begin(); i < range.end(); ++i)
        {
            ampEnergyList[i] = temporalAmpDifference(i + 1, deriv ? &ampDerivList[i] : NULL, hess ? &ampTList[i] : NULL, isProj);
            knoppelEnergyList[i] = spatialKnoppelEnergy(i + 1, deriv ? &knoppelDerivList[i] : NULL, hess ? &knoppelTList[i] : NULL, isProj);
        }
    };

    tbb::blocked_range<uint32_t> rangex1(0u, (uint32_t)numFrames, GRAIN_SIZE);
    tbb::parallel_for(rangex1, otherEnergiesPerframe);

    double ampE = 0, knoppelE = 0;

    for (int i = 0; i < numFrames; i++)
    {
        ampE += ampEnergyList[i];
        knoppelE += knoppelEnergyList[i];

        if (deriv)
        {
            deriv->segment(i * DOFsPerframe, DOFsPerframe) += knoppelDerivList[i];
            deriv->segment(i * DOFsPerframe, 2 * nverts) += ampDerivList[i];
        }

        if (hess)
        {
            for (auto& it : ampTList[i])
            {
                T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
            }
            for (auto& it : knoppelTList[i])
            {
                T.push_back({ i * DOFsPerframe + it.row(), i * DOFsPerframe + it.col(), it.value() });
            }
        }
    }

    energy += ampE + knoppelE;
    

    if (hess)
    {
        //std::cout << "num of triplets: " << T.size() << std::endl;
        hess->resize(DOFs, DOFs);
        hess->setFromTriplets(T.begin(), T.end());
        std::cout << "kinetic energy: " << ke << ", amp energy: " << ampE << ", knoppel energy: " << knoppelE << std::endl;
    }
    return energy;
}

void WrinkleEditingCWF::solveIntermeditateFrames(Eigen::VectorXd& x, int numIter, double gradTol, double xTol, double fTol, bool isdisplayInfo, std::string workingFolder)
{
    std::cout << "CWF model with new formula" << std::endl;
    computeAmpSqOmegaQuaticAverage();
    std::cout << "a^2 * |w|^4 = " << _ampSqOmegaQuaticAverage << std::endl;
    auto funVal = [&](const Eigen::VectorXd& x, Eigen::VectorXd* grad, Eigen::SparseMatrix<double>* hess, bool isProj) {
        Eigen::VectorXd deriv;
        Eigen::SparseMatrix<double> H;
        double E = computeEnergy(x, grad ? &deriv : NULL, hess ? &H : NULL, isProj);

        if (grad)
        {
            (*grad) = deriv;
        }

        if (hess)
        {
            (*hess) = H;
        }

        return E;
    };
    auto maxStep = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& dir) {
        return 1.0;
    };

    auto getVecNorm = [&](const Eigen::VectorXd& x, double& znorm, double& wnorm) {
        getComponentNorm(x, znorm, wnorm);
    };
    auto saveTmpRes = [&](const Eigen::VectorXd& x, std::string* folder)
    {
        save(x, folder);
    };

    OptSolver::testFuncGradHessian(funVal, x);
    auto x0 = x;
    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> hess;
    double f0 = funVal(x0, &grad, &hess, false);
    std::cout << "initial f: " << f0 << ", grad norm: " << grad.norm() << ", hess norm: " << hess.norm() << std::endl;

	if(std::isnan(grad.norm()) || std::isnan(hess.norm()))
	{
		std::cerr << "get nan error in hessian or gradient computation!" << std::endl;
		exit(EXIT_FAILURE);
	}
    OptSolver::newtonSolver(funVal, maxStep, x, numIter, gradTol, std::max(1e-16, xTol), std::max(1e-16, fTol), true, getVecNorm, &workingFolder, saveTmpRes);
    std::cout << "before optimization: " << x0.norm() << ", after optimization: " << x.norm() << std::endl;

    // get zvals
    convertVariable2List(x);
    for(int i = 1; i < _zvalsList.size() - 1; i++)
    {
        for(int j = 0; j < _zvalsList[i].size(); j++)
        {
            _zvalsList[i][j] = _unitZvalsList[i][j] * _combinedRefAmpList[i][j];
        }
    }

    std::cout << "solve finished." << std::endl;

    convertVariable2List(x);

    for(int i = 0; i < _zvalsList.size() - 2; i++)
    {
        std::cout << "frame: " << i << std::endl;
        double kinetic = kineticEnergy(i, NULL, NULL);
        double ampEnergy = temporalAmpDifference(i + 1, NULL, NULL);
        double knoppelEnergy = spatialKnoppelEnergy(i + 1, NULL, NULL);
        double kineticWithoutF = kineticEnergyWithoutFSq(i, NULL, NULL);
        std::cout << "kinetic: " << kinetic << ", without f^2: " << kineticWithoutF << ", amp: " << ampEnergy << ", knoppel: " << knoppelEnergy << std::endl;
    }
    std::cout << "frame: " << _zvalsList.size() - 2 << std::endl;
    double kinetic = kineticEnergy(_zvalsList.size() - 2, NULL, NULL);
    double kineticWithoutF = kineticEnergyWithoutFSq(_zvalsList.size() - 2, NULL, NULL);
    std::cout << "kinetic: " << kinetic << ", without f^2: " << kineticWithoutF << ", amp: " << 0 << ", knoppel: " << 0 << std::endl;
    return;
}