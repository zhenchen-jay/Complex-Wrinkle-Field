#include "../../include/DynamicInterpolation/ZdotIntegration.h"
#include <iostream>
double ZdotIntegration::triangleArea(const Eigen::Vector3d &P0, const Eigen::Vector3d &P1, const Eigen::Vector3d &P2)
{
    double a = (P0 - P1).norm();
    double b = (P0 - P2).norm();
    double c = (P1 - P2).norm();
    double p = (a + b + c) / 2;
    if(p - a <= 0 || p - b <= 0 || p - c <= 0)
    {
        std::cerr << "doesn't form an triagle, check your input!" << std::endl;
        exit(1);
    }
    return std::sqrt(p * (p - a) * (p - b) * (p - c));
}

/*
 * compute int_T bi(wi) bj(wj) dp, where bi(w) := f(alpha_i) exp(\tau * w.(p - pi)),
 * f(alpha_i) = 3 alpha_i^2 - 2 * alpha_i^3 - 2 * alpha_i * alpha_j * alpha_k,
 * (alpha_i, alpha_j, alpha_k) is the barycentric coordinates.
 * \tau^-2 = -1, i != j
 * All the nasty formula comes from mathematica
 */
double ZdotIntegration::computeBiBj(const Eigen::Vector2d &wi, const Eigen::Vector2d &wj, const Eigen::Vector3d &P0,
                                    const Eigen::Vector3d &P1, const Eigen::Vector3d &P2, int i , int j,
                                    Eigen::Vector4cd *deriv, Eigen::Matrix4cd *hess)
{
    double A = triangleArea(P0, P1, P2);
    std::vector<Eigen::Vector3d> pos(3);
    pos[0] = P0;
    pos[1] = P1;
    pos[2] = P2;

    Eigen::Vector3d Pi = pos[i];
    Eigen::Vector3d Pj = pos[j];

    int k = 0;
    if(i == 0 && j == 1 || i == 1 && j == 0)
        k = 2;
    else if(i == 0 && j == 2 || i == 2 && j == 0)
        k = 1;
    else
        k = 0;
    Eigen::Vector3d Pk = pos[k];
    double theta = wi(0) * (Pk - Pi)(0) + wi(1) * (Pk - Pi)(1) + wj(0) * (Pk - Pj)(0) + wj(1) * (Pk - Pj)(1);

    std::complex<double> c0 = std::complex<double>(std::cos(theta), std::sin(theta));

    double cK = (wi + wj).dot((Pi - Pk).segment<2>(0));
    double cL = (wi + wj).dot((Pj - Pk).segment<2>(0));

    // the formula comes from Mathematica
    double c1 = 2.0 / (std::pow(cK, 6) * std::pow(cL, 6) * std::pow(cL - cK, 7));
    std::complex<double> expK = std::complex<double>(std::cos(cK), std::sin(cK));
    std::complex<double> expL = std::complex<double>(std::cos(cL), std::sin(cL));
    std::complex<double> I = std::complex<double>(0, 1);

    std::complex<double> value =  240.0 * (-1.0 + expK) * std::pow(cL, 11) + 6 * std::pow(cK, 2) * std::pow(cL, 9) * (-652.0 + 120.0 * I * cL + 3 * cL * cL + (652.0 + (128.0 * I - 3 * cL) * expK));
    value += std::pow(cK, 4) * std::pow(cK, 7) * (-5112.0 + 2400.0 * I * cL - (-6.0 * I + cL) * (-852.0 * I + cL * (406.0 + cL * (-6.0 * I + cL))) * expK);
    value += 2 * std::pow(cK, 5) * std::pow(cK, 6) * (-21.0 * (-84.0 + 5.0 * cL * (8.0 * I + cL)) + (-1764.0 + cL * (-1716.0 * I + cL * (327.0 + cL * (-17.0 * I + 4.0 * cL)))) * expK);
    value += std::pow(cK, 11) * (6.0 * (40.0 + (20.0 * I - 3 * cL) * cL) + (-240.0 + cL * (120.0 * I + cL * (18.0 + cL * (2.0 * I + cL)))) * expL);
    value += 4 * std::pow(cK, 10) * cL * (-372.0 - 180.0 * I * cL + 25.0 * cL * cL + (372.0 - cL * (192.0 * I + cL * (31.0 + cL * (3.0 * I + 2.0 * cL)))) * expL);
    value += 2 * std::pow(cK, 6) * std::pow(cL, 5) * (-1764.0 + 49.0 * cL * cL + 804.0 * I * cL * expK - 302.0 * cL * cL * expK + 25.0 * I * std::pow(cL, 3) * expK - 11.0  * std::pow(cL, 4) * expK + (1764.0 - cL * (804.0 * I + cL * (107.0 + cL * (5.0 * I + 2 * cL)))) * expL);
    value += 2 * std::pow(cK, 9) * std::pow(cL, 2) * (1956.0 + 900.0 * I * cL - 107.0 * cL * cL + 2 * std::pow(cL, 4) * expK + (-1956.0 + cL * (1056.0 * I + cL * (185.0 + cL * (17.0 * I + 11.0 * cL)))) * expL);
    value += std::pow(cK, 8) * std::pow(cL, 3) * (-5688.0 + 30 * cL * (-80.0 * I + 7 * cL) - std::pow(cL, 3) * (-10.0 * I + 17 * cL) * expK - 2.0 * (-2844.0 + cL * (1644.0 * I + cL * (327.0 + cL * (25.0 * I + 14 * cL)))) * expL);
    value += std::pow(cK, 7) * std::pow(cL, 4) * (5112.0 + (-5112.0 + cL * (3432.0 * I + cL * (604.0 + cL * (36.0 * I + 17.0 * cL)))) * expL + 2.0 * cL * (840.0 * I - 49 * cL + cL * (107.0 + 2 * cL * (-9.0 * I + 7 * cL)) * expK));
    value += 24 * cK * std::pow(cL, 10) * (62.0 - 5.0 * I * cL + (-62.0 * I + 5.0 * cL) * (-I * std::cos(cK) + std::sin(cK)) + 2.0 * std::pow(cK, 3) * std::pow(cL, 8) * (2844.0 - 900.0 * I * cL - 50.0 * cL * cL + (-2844.0 * I + cL * (1056.0 + cL * (62.0 * I + cL))) * (-I * std::cos(cK) + std::sin(cK))));

    value = A * c0 * c1 * value;

    if(deriv || hess)
    {
        c1 = 2.0 /( std::pow(cK, 7) * std::pow(cK - cL, 8) * std::pow(cL, 6));
        std::complex<double> value0 = -1440.0 * I * std::pow(cL, 12) * (-1.0 + expK) - 120.0 * cK * std::pow(cL, 11) * (-88.0 * I * (-1.0 + expK) + cL * (5.0 + 7.0 * expK)) + 24 * cK * cK * std::pow(cL, 10) * (-1396.0 * I * (-1.0 + expK) + I * cL * cL * (-3.0 + 8.0 * expK) + 20 * cL * (9.0 + 13.0 * expK));
        value0 += -std::pow(cK, 6) * std::pow(cL, 6) * (752.0 * std::pow(cL, 3) * expK - 110.0 * I * std::pow(cL, 4) * expK + 9 * std::pow(cL, 5) * expK + 28224.0 * I * (-1.0 + expK) - 384 * cL * (35.0 + 94.0 * expK) - 48.0 * I * cL * cL * (-35.0 + 249.0 * expK)) + 2.0 * std::pow(cK, 4) * std::pow(cL, 8) * (-49 * std::pow(cL, 3) * expK + 2.0 * I * std::pow(cL, 4) * expK - 33552.0 * I * (-1.0 + expK) + 6.0 * I * cL * cL * (-119.0 + 405.0 * expK) + 24 * cL * (475.0 + 777.0 * expK));
        value0 += 6 * std::pow(cK, 3) * std::pow(cL, 9) * (2 * std::pow(cL, 3) * expK + 10016.0 * I * (-1.0 + expK) - I * cL * cL * (-83.0 + 243.0 * expK) - 4 * cL * (555.0 + 841.0 * expK)) + std::pow(cK, 5) * std::pow(cL, 7) * (352.0 * std::pow(cL, 3) * expK - 31.0 * I * std::pow(cL, 4) * expK + std::pow(cL, 5) * expK + 49536.0 * I * (-1.0 + expK) - 24.0 * I * cL * cL * (-89.0 + 391.0 * expK) - 48 * cL * (485.0 + 913.0 * expK));
        value0 += 2.0 * I * std::pow(cK, 12) * (2.0 * I * std::pow(cL, 3) * expL + std::pow(cL, 4) * expL - 240.0 * (-1.0 + expL) + 18 * std::pow(cL, 2) * (-1.0 + expL) + 120.0 * I * cL * (1.0 + expL)) - std::pow(cK, 10) * std::pow(cL, 2) * (21 * std::pow(cL, 5) * expK + 88 * std::pow(cL, 3) * expL + 9696.0 * I * (-1.0 + expL) - 2.0 * I * std::pow(cL, 4) * (13.0 * expK + 28.0 * expL) - 24.0 * I * std::pow(cL, 2) * (-19.0 + 41.0 * expL) + 96.0 * cL * (45.0 + 56.0 * expL));
        value0 += std::pow(cK, 11) * cL * (4 * std::pow(cL, 5) * expK + 26 * std::pow(cL, 3) * expL - 19.0 * I * std::pow(cL, 4) * expL + 3264.0 * I * (-1.0 + expL) - 6.0 * I * std::pow(cL, 2) * (-35.0 + 47.0 * expL) + 24.0 * cL * (65.0 + 71.0 * expL));
        value0 += std::pow(cK, 7) * std::pow(cL, 5) * (30 * std::pow(cL, 5) * expK - 48 * cL * (35.0 + 308.0 * expK - 163.0 * expL) + 19584.0 * I * (-1.0 +expL) - I * std::pow(cL, 4) * (210.0 * expK + 11.0 * expL) + std::pow(cL, 3) * (944.0 * expK + 34.0 * expL) - 6.0 * I * std::pow(cL, 2) * (-98.0 + 1509.0 * expK + 149.0 * expL));
        value0 += -2 * std::pow(cK, 8) * std::pow(cL, 4) * (25 * std::pow(cL, 5) * expK + 9648.0 * I * (-1.0 + expL) - I * std::pow(cL, 4) * (110.0 * expK + 23.0 * expL) + std::pow(cL, 3) * (311.0 * expK + 58.0 * expL) + 48 * cL * (55.0 + 146.0 * expL) - 6.0 * I * std::pow(cL, 2) * (-14.0 + 241.0 * expK + 193.0 * expL));
        value0 += std::pow(cL, 9) * std::pow(cL, 3) * (45 * std::pow(cL, 5) * expK + 16704.0 * I * (-1.0 + expL) + 4 * std::pow(cL, 3) * (41.0 * expK + 37.0 * expL) - I * std::pow(cL, 4) * (119.0 * expK + 74.0 * expL) - 24.0 * I * std::pow(cL, 2) * (-17.0 + 90.0 * expL) + 24 * cL * (275.0 + 421.0 * expL));

        value0 = A * c0 * c1 * value0;

        c1 = 2.0 /( std::pow(cK, 6) * std::pow(cK - cL, 8) * std::pow(cL, 7));
        std::complex<double> value1 =  (-480.0 * I * std::pow(cL, 12) * (-1.0 + expK) - 48 * cK * std::pow(cL, 11) * (-68.0 * I * (-1.0 + expK) + 5 * cL * (1.0 + expK)) - 2 * std::pow(cK, 3) * std::pow(cL, 9) * (2 * std::pow(cL, 3) * expK - 8352.0 * I * (-1.0 + expK) + 3.0 * I * std::pow(cL, 2) * (-35.0 + 47.0 * expK) + 48 * cL * (45.0 + 56.0 * expK)) + 12.0 * std::pow(cK, 2) * std::pow(cL, 10) * (-808.0 * I * (-1.0 + expK) + 3.0 * I * std::pow(cL, 2) * (-1.0 + expK) + 2.0 * cL * (65.0 + 71.0 * expK)) - I * std::pow(cK, 5) * std::pow(cL, 7) * (-88.0 * I * std::pow(cL, 3) * expK + 19 * std::pow(cL, 4) * expK - 19584.0 * (-1.0 + expK) + 24.0 * std::pow(cL, 2) * (-17.0 + 90.0 * expK) - 96.0 * I * cL * (55.0 + 146.0 * expK)) + 2 * std::pow(cK, 4) * std::pow(cL, 8) * (13 * std::pow(cL, 3) * expK + I * std::pow(cL, 4) * expK - 9648.0 * I * (-1.0 + expK) + 12.0 * I * std::pow(cL, 2) * (-19.0 + 41.0 * expK) + 12.0 * cL * (275.0 + 421.0 * expK)) + std::pow(cK, 12) * (12 * std::pow(cL, 3) * expL + 4.0 * I * std::pow(cL, 4) * expL + std::pow(cL, 5) * expL - 1440.0 * I * (-1.0 + expL) - 120 * cL * (5.0 + 7.0 * expL) + 24.0 * I * std::pow(cL, 2) * (-3.0 + 8.0 * expL)) + 2 * std::pow(cK, 6) * std::pow(cL, 6) * (2 * std::pow(cL, 5) * expL + 24.0 * cL * (-35.0 + 163.0 * expK - 308.0 * expL) - 14112.0 * I * (-1.0 + expL) + I * std::pow(cL, 4) * (28.0 * expK + 13.0 * expL) + std::pow(cL, 3) * (74.0 * expK + 82.0 * expL) + 6.0 * I * std::pow(cL, 2) * (-14.0 + 193.0 * expK + 241.0 * expL)) - std::pow(cK, 11) * cL * (98 * std::pow(cL, 3) * expL + 31.0 * I * std::pow(cL, 4) * expL + 9 * std::pow(cL, 5) * expL - 10560.0 * I * (-1.0 + expL) - 480.0 * cL * (9.0 + 13.0 * expL) + 6.0 * I * std::pow(cL, 2) * (-83.0 + 243.0 * expL)) - std::pow(cK, 9) * std::pow(cL, 3) * (752 * std::pow(cL, 3) * expL + 50 * std::pow(cL, 5) * expL - 60096.0  * I * (-1.0 + expL) + I * std::pow(cL, 4) * (11.0 * expK + 210.0 * expL) + 24.0 * I * std::pow(cL, 2) * (-89.0 + 391.0 * expL) - 48 * cL * (475.0 + 777.0 * expL)) + 2.0 * std::pow(cK, 10) * std::pow(cL, 2) * (176 * std::pow(cL, 3) * expL + 55.0 * I * std::pow(cL, 4) * expL + 15.0 * std::pow(cL, 5) * expL - 16752.0 * I * (-1.0 + expL) + 6.0 * I * std::pow(cL, 2) * (-119.0 + 405.0 * expL) - 12.0 * cL * (555.0 + 841.0 * expL)) + std::pow(cK, 8) * std::pow(cL, 4) * (45 * std::pow(cL, 5) * expL - 67104.0 * I * (-1.0 + expL) + 2.0 * I * std::pow(cL, 4) * (23.0 * expK + 110.0 * expL) + 48.0 * I * std::pow(cL, 2) * (-35.0 + 249.0 * expL) - 48.0 * cL * (485.0 + 913.0 * expL) + std::pow(cL, 3) * (34.0 * expK + 944.0 * expL)) - std::pow(cK, 7) * std::pow(cK, 5) * (21 * std::pow(cK, 5) * expL - 49536.0 * I * (-1.0 + expL) - 384 * cL * (35.0 + 94.0 * expL) + I * std::pow(cK, 4) * (74.0 * expK + 119.0 * expL) + 2.0 * std::pow(cK, 3) * (58.0 * expK + 311.0 * expL) + 6.0 * I * std::pow(cK, 2) * (-98.0 + 149.0 * expK + 1509.0 *  expL)));

        value1 = A * c0 * c1 * value1;

        if(deriv)
        {
            deriv->segment<2>(0) = I * (value * (Pk - Pi) + value0 * (Pi - Pk) + value1 * (Pj - Pk)).segment<2>(0).transpose();
            deriv->segment<2>(2) = I * (value * (Pk - Pj) + value0 * (Pi - Pk) + value1 * (Pj - Pk)).segment<2>(0).transpose();
        }

        if(hess)
        {
            std::complex<double> value01 = (2.0*(-2880*std::pow(cL,13)* (-1.0 + expK) + 240 * cK * std::pow(cL,12)*(96.0*(-1.0 + expK) + std::complex<double>(0,1)*cL*(5.0 + 7.0*expK)) + 48*std::pow(cK,2)*std::pow(cL,11)*(-1692.0*(-1.0 + expK) + std::pow(cL,2)*(-3.0 + 8.0*expK) - std::complex<double>(0,15)*cL*(13.0 + 19.0*expK)) + 6*std::pow(cK,3)*std::pow(cL,10)*(std::complex<double>(0,-4)*std::pow(cL,3)*expK + std::pow(cL,2)*(177.0 - 537.0*expK) + 27744.0*(-1.0 + expK) + std::complex<double>(0,96)*cL*(55.0 + 86.0*expK)) + 3*std::pow(cK,6)*std::pow(cL,7)*(std::complex<double>(0,784)*std::pow(cL,3)*expK + 95*std::pow(cL,4)*expK + std::complex<double>(0,7)*std::pow(cL,5)*expK - 58752.0*(-1.0 + expK) - std::complex<double>(0,288)*cL*(55.0 + 191.0*expK) + 24*std::pow(cL,2)*(-51.0 + 605.0*expK)) + std::pow(cK,5)*std::pow(cL,8)*(std::complex<double>(0,-918)*std::pow(cL,3)*expK - 69*std::pow(cL,4)*expK - std::complex<double>(0,2)*std::pow(cL,5)*expK + std::pow(cL,2)*(4968.0 - 27480.0*expK) + 212544.0*(-1.0 + expK) + std::complex<double>(0,144)*cL*(495.0 + 1049.0*expK)) + 2*std::pow(cK,4)*std::pow(cL,9)*(std::complex<double>(0,111)*std::pow(cL,3)*expK + 4*std::pow(cL,4)*expK - 111168.0*(-1.0 + expK) + 9*std::pow(cL,2)*(-179.0 + 675.0*expK) - std::complex<double>(0,24)*cL*(1265.0 + 2203.0*expK)) + 2*std::pow(cK,13)*(std::complex<double>(0,12)*std::pow(cL,3)*expL - 4*std::pow(cL,4)*expL + std::complex<double>(0,1)*std::pow(cL,5)*expL + 1440.0*(-1.0 + expL) - std::complex<double>(0,120)*cL*(5.0 + 7.0*expL) - 24*std::pow(cL,2)*(-3.0 + 8.0*expL)) + 3*std::pow(cK,12)*cL*(std::complex<double>(0,-74)*std::pow(cL,3)*expL + 23*std::pow(cL,4)*expL - std::complex<double>(0,7)*std::pow(cL,5)*expL - 7680.0*(-1.0 + expL) + std::complex<double>(0,240)*cL*(13.0 + 19.0*expL) + 6*std::pow(cL,2)*(-59.0 + 179.0*expL)) + std::pow(cK,11)*std::pow(cL,2)*(std::complex<double>(0,918)*std::pow(cL,3)*expL - 285*std::pow(cL,4)*expL - std::complex<double>(0,1)*std::pow(cL,5)*(11.0*expK - 75.0*expL) + 81216.0*(-1.0 + expL) - std::complex<double>(0,576)*cL*(55.0 + 86.0*expL) - 18*std::pow(cL,2)*(-179.0 + 675.0*expL)) + 3*std::pow(cK,9)*std::pow(cL,4)*(std::complex<double>(0,-2)*std::pow(cL,3)*(115.0*expK - 568.0*expL) - std::complex<double>(0,40)*std::pow(cL,5)*(expK - expL) + 74112.0*(-1.0 + expL) - std::pow(cL,4)*(131.0*expK + 230.0*expL) - 24*std::pow(cL,2)*(-51.0 + 605.0*expL) - std::complex<double>(0,48)*cL*(495.0 + 1049.0*expL)) + std::pow(cK,10)*std::pow(cL,3)*(std::complex<double>(0,-2352)*std::pow(cL,3)*expL + std::complex<double>(0,1)*std::pow(cL,5)*(57.0*expK - 130.0*expL) - 166464.0*(-1.0 + expL) + std::pow(cL,4)*(89.0*expK + 610.0*expL) + 24*std::pow(cL,2)*(-207.0 + 1145.0*expL) + std::complex<double>(0,48)*cL*(1265.0 + 2203.0*expL)) + std::pow(cK,7)*std::pow(cL,6)*(std::complex<double>(0,-6)*std::pow(cL,3)*(568.0*expK - 115.0*expL) - std::complex<double>(0,1)*std::pow(cL,5)*(75.0*expK - 11.0*expL) + std::complex<double>(0,82176)*cL*(expK - expL) + 176256.0*(-1.0 + expL) - std::pow(cL,4)*(610.0*expK + 89.0*expL) - 6*std::pow(cL,2)*(-126.0 + 6579.0*expK + 2347.0*expL)) + std::pow(cK,8)*std::pow(cL,5)*(std::complex<double>(0,1)*std::pow(cL,5)*(130.0*expK - 57.0*expL) + std::complex<double>(0,2466)*std::pow(cL,3)*(expK - expL) - 212544.0*(-1.0 + expL) + std::complex<double>(0,864)*cL*(55.0 + 191.0*expL) + std::pow(cL,4)*(690.0*expK + 393.0*expL) + 6*std::pow(cL,2)*(-126.0 + 2347.0*expK + 6579.0*expL))))/(std::pow(cK,7)*std::pow(cK - cL,9)*std::pow(cL,7));

            std::complex<double> value00 = (2.0*(10080*std::pow(cL,13)*(-1.0 + expK) + 720*cK*std::pow(cL,12)*(-118.0*(-1.0 + expK) - std::complex<double>(0,1)*cL*(5.0 + 9.0*expK)) -
                    120*std::pow(cK,2)*std::pow(cL,11)*(-2628.0*(-1.0 + expK) + 3*std::pow(cL,2)*(-1.0 + 5.0*expK) - std::complex<double>(0,2)*cL*(125.0 + 229.0*expK)) -
                    2*std::pow(cK,7)*std::pow(cL,6)*(std::complex<double>(0,-14052)*std::pow(cL,3)*expK + 57*std::pow(cL,4)*expK - std::complex<double>(0,111)*std::pow(cL,5)*expK + 5*std::pow(cL,6)*expK + 127008.0*(-1.0 + expK) -
                    1512*std::pow(cL,2)*(-5.0 + 62.0*expK) + std::complex<double>(0,864)*cL*(70.0 + 233.0*expK)) +
                    48*std::pow(cK,3)*std::pow(cL,10)*(std::complex<double>(0,5)*std::pow(cL,3)*expK - 14082.0*(-1.0 + expK) - std::complex<double>(0,30)*cL*(76.0 + 143.0*expK) + std::pow(cL,2)*(-61.0 + 321.0*expK)) +
                    std::pow(cK,6)*std::pow(cL,7)*(std::complex<double>(0,-18516)*std::pow(cL,3)*expK + 30*std::pow(cL,4)*expK - std::complex<double>(0,54)*std::pow(cL,5)*expK + std::pow(cL,6)*expK + 523584.0*(-1.0 + expK) +
                    std::complex<double>(0,49248)*cL*(5.0 + 12.0*expK) - 288*std::pow(cL,2)*(-80.0 + 649.0*expK)) +
                    36*std::pow(cK,4)*std::pow(cL,9)*(std::complex<double>(0,-58)*std::pow(cL,3)*expK + 25624.0*(-1.0 + expK) - 5*std::pow(cL,2)*(-57.0 + 325.0*expK) + std::complex<double>(0,4)*cL*(1585.0 + 3109.0*expK)) +
                    2*std::pow(cK,5)*std::pow(cL,8)*(std::complex<double>(0,4050)*std::pow(cL,3)*expK - 2*std::pow(cL,4)*expK + std::complex<double>(0,3)*std::pow(cL,5)*expK - 418608.0*(-1.0 + expK) +
                    54*std::pow(cL,2)*(-185.0 + 1201.0*expK) - std::complex<double>(0,72)*cL*(2065.0 + 4341.0*expK)) +
                    2*std::pow(cK,13)*(-2*std::pow(cL,6)*expK + std::complex<double>(0,6)*std::pow(cL,3)*expL + 3*std::pow(cL,4)*expL - 720.0*(-1.0 + expL) + 54*std::pow(cL,2)*(-1.0 + expL) +
                    std::complex<double>(0,360)*cL*(1.0 + expL)) + std::pow(cK,12)*cL*(std::complex<double>(0,-42)*std::pow(cL,5)*expK + 25*std::pow(cL,6)*expK - std::complex<double>(0,84)*std::pow(cL,3)*expL -
                    66*std::pow(cL,4)*expL + std::pow(cL,2)*(660.0 - 948.0*expL) + 10656.0*(-1.0 + expL) - std::complex<double>(0,144)*cL*(35.0 + 39.0*expL)) -
                    2*std::pow(cK,11)*std::pow(cL,2)*(std::complex<double>(0,-117)*std::pow(cL,5)*expK + 33*std::pow(cL,6)*expK - std::complex<double>(0,168)*std::pow(cL,3)*expL + 17*std::pow(cL,4)*(expK - 6.0*expL) +
                    17712.0*(-1.0 + expL) - 24*std::pow(cL,2)*(-30.0 + 79.0*expL) - std::complex<double>(0,48)*cL*(160.0 + 209.0*expL)) +
                    std::pow(cK,10)*std::pow(cL,3)*(std::complex<double>(0,-546)*std::pow(cL,5)*expK + 95*std::pow(cL,6)*expK + 6*std::pow(cL,4)*(25.0*expK - 46.0*expL) + 71136.0*(-1.0 + expL) -
                    std::complex<double>(0,12)*std::pow(cL,3)*(323.0*expK + 5.02*expL) - 72*std::pow(cL,2)*(-15.0 + 139.0*expL) - std::complex<double>(0,144)*cL*(185.0 + 309.0*expL)) +
                    std::pow(cK,8)*std::pow(cL,5)*(std::complex<double>(0,-510)*std::pow(cL,5)*expK + 39*std::pow(cL,6)*expK + std::complex<double>(0,96)*cL*(195.0 + 1526.0*expK - 506.0*expL) +
                    std::pow(cL,4)*(236.0*expK - 42.0*expL) + 137376.0*(-1.0 + expL) - std::complex<double>(0,12)*std::pow(cL,3)*(2328.0*expK + 13.0*expL) -
                    12*std::pow(cL,2)*(-378.0 + 10035.0*expK + 403.0*expL)) - 2*std::pow(cK,9)*std::pow(cL,4)*
                    (std::complex<double>(0,-345)*std::pow(cL,5)*expK + 40*std::pow(cL,6)*expK + 3*std::pow(cL,4)*(44.0*expK - 29.0*expL) + 50832.0*(-1.0 + expL) -
                    std::complex<double>(0,6)*std::pow(cL,3)*(1331.0*expK + 43.0*expL) - std::complex<double>(0,72)*cL*(165.0 + 541.0*expL) - 6*std::pow(cL,2)*(-30.0 + 2919.0*expK + 991.0*expL))))/
                            (std::pow(cK,8)*std::pow(cL,6)*std::pow(-cK + cL,9));

            std::complex<double> value11 = (2.0*(-1440*std::pow(cL,13)*(-1.0 + expK) + 144*cK*std::pow(cL,12)*(74.0*(-1.0 + expK) + std::complex<double>(0,5)*cL*(1.0 + expK)) +
                    36*std::pow(cK,2)*std::pow(cL,11)*(-984.0*(-1.0 + expK) + 3*std::pow(cL,2)*(-1.0 + expK) - std::complex<double>(0,4)*cL*(35.0 + 39.0*expK)) +
                    12*std::pow(cK,3)*std::pow(cL,10)*(std::complex<double>(0,1)*std::pow(cL,3)*expK + std::pow(cL,2)*(55.0 - 79.0*expK) + 5928.0*(-1.0 + expK) + std::complex<double>(0,8)*cL*(160.0 + 209.0*expK)) +
                    6*std::pow(cK,4)*std::pow(cL,9)*(std::complex<double>(0,-14)*std::pow(cL,3)*expK + std::pow(cL,4)*expK - 16944.0*(-1.0 + expK) + 8*std::pow(cL,2)*(-30.0 + 79.0*expK) -
                    std::complex<double>(0,24)*cL*(185.0 + 309.0*expK)) - 6*std::pow(cK,5)*std::pow(cL,8)*(std::complex<double>(0,-56)*std::pow(cL,3)*expK + 11*std::pow(cL,4)*expK - 22896.0*(-1.0 + expK) +
                    12*std::pow(cL,2)*(-15.0 + 139.0*expK) - std::complex<double>(0,24)*cL*(165.0 + 541.0*expK)) +
                    std::pow(cK,13)*(std::complex<double>(0,240)*std::pow(cL,3)*expL + std::complex<double>(0,6)*std::pow(cL,5)*expL + std::pow(cL,6)*expL + 10080.0*(-1.0 + expL) - 360*std::pow(cL,2)*(-1.0 + 5.0*expL) -
                    std::complex<double>(0,720)*cL*(5.0 + 9.0*expL)) - 2*std::pow(cK,12)*cL*(std::complex<double>(0,1044)*std::pow(cL,3)*expL + 2*std::pow(cL,4)*expL + std::complex<double>(0,27)*std::pow(cL,5)*expL +
                    5*std::pow(cL,6)*expL + 42480.0*(-1.0 + expL) - std::complex<double>(0,120)*cL*(125.0 + 229.0*expL) - 24*std::pow(cL,2)*(-61.0 + 321.0*expL)) +
                    3*std::pow(cK,11)*std::pow(cL,2)*(std::complex<double>(0,2700)*std::pow(cL,3)*expL + 10*std::pow(cL,4)*expL + std::complex<double>(0,74)*std::pow(cL,5)*expL + 13*std::pow(cL,6)*expL + 105120.0*(-1.0 + expL) -
                    std::complex<double>(0,480)*cL*(76.0 + 143.0*expL) - 60*std::pow(cL,2)*(-57.0 + 325.0*expL)) -
                    6*std::pow(cK,8)*std::pow(cL,5)*(std::complex<double>(0,91)*std::pow(cL,5)*expL + 11*std::pow(cL,6)*expL + 139536.0*(-1.0 + expL) - std::complex<double>(0,8208)*cL*(5.0 + 12.0*expL) +
                    std::pow(cL,4)*(-29.0*expK + 44.0*expL) - 504*std::pow(cL,2)*(-5.0 + 62.0*expL) + std::complex<double>(0,2)*std::pow(cL,3)*(13.0*expK + 2328.0*expL)) -
                    2*std::pow(cK,6)*std::pow(cL,7)*(std::complex<double>(0,21)*std::pow(cL,5)*expL + 2*std::pow(cL,6)*expL + std::complex<double>(0,48)*cL*(-195.0 + 506.0*expK - 1526.0*expL) -
                    17*std::pow(cL,4)*(6.0*expK - expL) + 127008.0*(-1.0 + expL) + std::complex<double>(0,6)*std::pow(cL,3)*(52.0*expK + 323.0*expL) -
                    6*std::pow(cL,2)*(-30.0 + 991.0*expK + 2919.0*expL)) - 2*std::pow(cK,10)*std::pow(cL,3)*
                    (std::complex<double>(0,9258)*std::pow(cL,3)*expL + 57*std::pow(cL,4)*expL + std::complex<double>(0,255)*std::pow(cL,5)*expL + 40*std::pow(cL,6)*expL + std::pow(cL,2)*(9990.0 - 64854.0*expL) +
                    337968.0*(-1.0 + expL) - std::complex<double>(0,72)*cL*(1585.0 + 3109.0*expL)) +
                    std::pow(cK,9)*std::pow(cL,4)*(std::complex<double>(0,28104)*std::pow(cL,3)*expL + std::complex<double>(0,690)*std::pow(cL,5)*expL + 95*std::pow(cL,6)*expL + 922464.0*(-1.0 + expL) +
                    std::pow(cL,4)*(-42.0*expK + 236.0*expL) - 288*std::pow(cL,2)*(-80.0 + 649.0*expL) - std::complex<double>(0,144)*cL*(2065.0 + 4341.0*expL)) +
                    std::pow(cK,7)*std::pow(cL,6)*(std::complex<double>(0,234)*std::pow(cL,5)*expL + 25*std::pow(cL,6)*expL - 6*std::pow(cL,4)*(46.0*expK - 25.0*expL) + 523584.0*(-1.0 + expL) -
                    std::complex<double>(0,1728)*cL*(70.0 + 233.0*expL) + std::complex<double>(0,12)*std::pow(cL,3)*(43.0*expK + 1331.0*expL) - 12*std::pow(cL,2)*(-378.0 + 403.0*expK + 10035.0*expL))))/
                            (std::pow(cK,6)*std::pow(cK - cL,9)*std::pow(cL,8));

            Eigen::Matrix2d Mkiki = (Pk - Pi).segment<2>(0) * (Pk - Pi).segment<2>(0).transpose();
            Eigen::Matrix2d Mkjkj = (Pk - Pj).segment<2>(0) * (Pk - Pj).segment<2>(0).transpose();
            Eigen::Matrix2d Mkikj = (Pk - Pi).segment<2>(0) * (Pk - Pj).segment<2>(0).transpose();
            Eigen::Matrix2d Mkjki = (Pk - Pj).segment<2>(0) * (Pk - Pi).segment<2>(0).transpose();

            hess->block<2, 2>(0, 0) = -value * Mkiki + 2 * Mkiki * value0 + (Mkikj + Mkjki) * value1 - Mkiki * value00 - (Mkikj + Mkjki) * value01 - Mkjkj * value11;
            hess->block<2, 2>(0, 2) = - value * Mkikj + (Mkiki + Mkikj) * value0 + (Mkikj + Mkjkj) * value1 - Mkiki * value00 - (Mkikj + Mkjki) * value01 - Mkjkj * value11;
            hess->block<2, 2>(2, 0) = - value * Mkjki + (Mkiki + Mkjki) * value0 + (Mkjki + Mkjkj) * value1 - Mkiki * value00 - (Mkikj + Mkjki) * value01 - Mkjkj * value11;
            hess->block<2, 2>(2, 2) = -value * Mkjkj + 2 * Mkjkj * value0 + (Mkikj + Mkjki) * value1 - Mkiki * value00 - (Mkikj + Mkjki) * value01 - Mkjkj * value11;
            
        }
    }

}