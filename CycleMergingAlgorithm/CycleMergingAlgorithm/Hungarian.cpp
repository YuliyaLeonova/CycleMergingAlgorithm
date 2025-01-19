#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <mkl.h>
#include "Hungarian.h"
#include <omp.h>

#define DBL_EPSILON 2.2204460492503131e-16

HungarianAlgorithm::HungarianAlgorithm() {}
HungarianAlgorithm::~HungarianAlgorithm() {}


double HungarianAlgorithm::Solve(int size, std::vector<double>& DistMatrix, std::vector<int>& Assignment) {

    int nOfRows = size;
    int nOfColumns = size;
    int nOfElements = nOfRows * nOfColumns;

    int chunk_size = 64;

    // ���������� std::vector ��� ���������� �������
    std::vector<double> distMatrix(nOfElements);
    std::vector<double> originalDistMatrix(nOfElements);
    std::vector<int> assignment(nOfRows);

    // ����������� DistMatrix � distMatrix
    cblas_dcopy(nOfElements, DistMatrix.data(), 1, distMatrix.data(), 1);

    // ����������� DistMatrix � originalDistMatrix
    cblas_dcopy(nOfElements, DistMatrix.data(), 1, originalDistMatrix.data(), 1);

    double cost = 0.0;
    assignmentoptimal(assignment.data(), &cost, distMatrix.data(), nOfRows, nOfColumns);

    // Build the assignment vector
    Assignment.assign(assignment.begin(), assignment.end());

    // Calculate cost using the original matrix
    computeassignmentcost(assignment.data(), &cost, originalDistMatrix.data(), nOfRows);

    return cost;
}

void HungarianAlgorithm::assignmentoptimal(int* assignment, double* cost, double* distMatrix, int nOfRows, int nOfColumns) {
    int nOfElements = nOfRows * nOfColumns;

    // ���������� std::vector<char> ��� ���������� �������
    std::vector<char> starMatrix(nOfElements, 0);
    std::vector<char> newStarMatrix(nOfElements, 0);
    std::vector<char> primeMatrix(nOfElements, 0);
    std::vector<char> coveredColumns(nOfColumns, 0);
    std::vector<char> coveredRows(nOfRows, 0);

    // Step 1: Subtract the smallest value in each row
    for (int row = 0; row < nOfRows; row++) {
        double minValue = distMatrix[row]; // ������������� ������ ��������� ������
        for (int col = 1; col < nOfColumns; col++) {
            double value = distMatrix[col * nOfRows + row];
            if (value < minValue) {
                minValue = value;
            }
        }
        for (int col = 0; col < nOfColumns; col++) {
            distMatrix[col * nOfRows + row] -= minValue;
        }
    }

    // Step 2a: �������� ���� � �������� ��������
    int coveredCount = 0;
    for (int row = 0; row < nOfRows; row++) {
        for (int col = 0; col < nOfColumns; col++) {
            if (fabs(distMatrix[col * nOfRows + row]) < DBL_EPSILON && !coveredColumns[col]) {
                starMatrix[col * nOfRows + row] = 1; // ������������� ������
                coveredColumns[col] = 1;            // ��������� �������
                coveredCount++;
                break;
            }
        }
        if (coveredCount == nOfColumns) break; // ��� ������� �������, ������ �����
    }

    // ������� �������� �� �����
    std::fill(coveredRows.begin(), coveredRows.end(), 0);

    int minDim = std::min(nOfRows, nOfColumns);

    // Step 3
    step3(assignment, distMatrix, starMatrix.data(), newStarMatrix.data(), primeMatrix.data(),
        coveredColumns.data(), coveredRows.data(), nOfRows, nOfColumns, minDim);

    // Compute the cost
    computeassignmentcost(assignment, cost, distMatrix, nOfRows);
}



void HungarianAlgorithm::buildassignmentvector(int* assignment, const char* starMatrix, int nOfRows, int nOfColumns) {
    int chunk_size = 64;
#pragma omp parallel for schedule(dynamic, chunk_size)
    for (int row = 0; row < nOfRows; row++) {
        assignment[row] = -1; // ���������� ��� ����������
        for (int col = 0; col < nOfColumns; col++) {
            if (starMatrix[col * nOfRows + row]) { // �������� �� "����������"
                assignment[row] = col; // ��������� �������
                break; // ��������� � ��������� ������
            }
        }
    }
}

void HungarianAlgorithm::computeassignmentcost(int* assignment, double* cost, const double* originalDistMatrix, int nOfRows) {
    std::vector<double> selectedValues(nOfRows, 0.0);

    // ��������� �������� �� originalDistMatrix �� ������ assignment
    for (int row = 0; row < nOfRows; ++row) {
        int col = assignment[row];
        if (col >= 0) {
            selectedValues[row] = originalDistMatrix[col * nOfRows + row]; // �������-������ � ������� column-major
        }
    }

    // ���������� MKL ��� ������������ ���� ��������
    *cost = cblas_dasum(nOfRows, selectedValues.data(), 1); // ��������� ��� �������� ������� selectedValues
}

void HungarianAlgorithm::step2a(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
    char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    int chunk_size = 64;

    // ������������ ����� ��������
#pragma omp parallel
    {
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < nOfColumns; ++i) {
            coveredColumns[i] = 0;
        }

#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < nOfRows; ++i) {
            coveredRows[i] = 0;
        }
    }

    // ������������ �������� ��������
#pragma omp parallel for schedule(dynamic, chunk_size)
    for (int col = 0; col < nOfColumns; ++col) {
        for (int row = 0; row < nOfRows; ++row) {
            if (starMatrix[col * nOfRows + row]) {
                coveredColumns[col] = 1;
                break; // ������� �� ����������� �����
            }
        }
    }

    // ������� � ���� 2b
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}



void HungarianAlgorithm::step2b(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
    char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    // ������� �������� ��������
    int nCoveredColumns = 0;
    int chunk_size = 64;
#pragma omp parallel for reduction(+:nCoveredColumns) schedule(static, 64)
    for (int col = 0; col < nOfColumns; ++col) {
        nCoveredColumns += coveredColumns[col]; // ������� ���������� �����
    }

    // �������� ����������
    if (nCoveredColumns == minDim) {
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    }
    else {
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
            coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

void HungarianAlgorithm::step3(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
    char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    bool zerosFound = true;

    while (zerosFound) {
        zerosFound = false;

        // ������ �� ���� ��������
        for (int col = 0; col < nOfColumns; col++) {
            if (!coveredColumns[col]) {
                for (int row = 0; row < nOfRows; row++) {
                    if (!coveredRows[row] && fabs(distMatrix[col * nOfRows + row]) < DBL_EPSILON) {
                        primeMatrix[col * nOfRows + row] = 1; // ������������ ����

                        // ����� �������� ���� � ������� ������
                        int starCol = -1;
                        for (int c = 0; c < nOfColumns; c++) {
                            if (starMatrix[c * nOfRows + row]) {
                                starCol = c;
                                break;
                            }
                        }

                        // ���� �������� ���� ���, ��������� � ���� 4
                        if (starCol == -1) {
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
                                coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return;
                        }
                        else {
                            // ��������� ������ � ������� �������� �� �������
                            coveredRows[row] = 1;
                            coveredColumns[starCol] = 0;
                            zerosFound = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    // ������� � ���� 5
    step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
        coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step4(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
    char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col) {
    int starRow = -1, starCol = col;

    // �������� ������ ������� ������� �����
    std::copy(starMatrix, starMatrix + nOfRows * nOfColumns, newStarMatrix);

    // ������������� ������� ���� ��� �������
    newStarMatrix[starCol * nOfRows + row] = 1;

    // ������� ������� ���� � ������� �������
    for (int r = 0; r < nOfRows; r++) {
        if (starMatrix[starCol * nOfRows + r]) {
            starRow = r;
            break;
        }
    }

    // ����������� ������� ���������� ������� �����
    while (starRow != -1) {
        // ������� ������ ������
        newStarMatrix[starCol * nOfRows + starRow] = 0;

        // ������� ������������� ���� � ������� ������
        int primeCol = -1;
        for (int c = 0; c < nOfColumns; c++) {
            if (primeMatrix[c * nOfRows + starRow]) {
                primeCol = c;
                break;
            }
        }

        // ������������� ����� ������
        newStarMatrix[primeCol * nOfRows + starRow] = 1;

        // ��������� ������� ������� � ���� ��������� ������
        starCol = primeCol;
        starRow = -1;
        for (int r = 0; r < nOfRows; r++) {
            if (starMatrix[starCol * nOfRows + r]) {
                starRow = r;
                break;
            }
        }
    }

    // �������� ���������� ������� ������� �����
    std::copy(newStarMatrix, newStarMatrix + nOfRows * nOfColumns, starMatrix);

    // ���������� ������������� ���� � ��������
    std::fill(primeMatrix, primeMatrix + nOfRows * nOfColumns, 0);
    std::fill(coveredRows, coveredRows + nOfRows, 0);
    std::fill(coveredColumns, coveredColumns + nOfColumns, 0);

    // ������� � ���� 2a
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step5(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
    char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim) {
    double h = DBL_MAX;

    //omp_set_schedule(omp_sched_dynamic, 20); // ������������� ���������� ������ �����
    int chunk_size = 64; // ������ �����

    // ����� ������������ �������� ����� ���������� ���������
#pragma omp parallel for reduction(min : h) schedule(dynamic, chunk_size)
    for (int row = 0; row < nOfRows; row++) {
        if (!coveredRows[row]) {
            for (int col = 0; col < nOfColumns; col++) {
                if (!coveredColumns[col]) {
                    h = std::min(h, distMatrix[col * nOfRows + row]);
                }
            }
        }
    }

    // ��������� h � �������� �������
#pragma omp parallel for schedule(dynamic, chunk_size)
    for (int row = 0; row < nOfRows; row++) {
        if (coveredRows[row]) {
            for (int col = 0; col < nOfColumns; col++) {
                distMatrix[col * nOfRows + row] += h;
            }
        }
    }

    // �������� h �� ���������� ��������
#pragma omp parallel for schedule(dynamic, chunk_size)
    for (int col = 0; col < nOfColumns; col++) {
        if (!coveredColumns[col]) {
            for (int row = 0; row < nOfRows; row++) {
                distMatrix[col * nOfRows + row] -= h;
            }
        }
    }

    // ������� � ���� 3
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
        coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

