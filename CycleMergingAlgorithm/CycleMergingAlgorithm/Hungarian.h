
#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

class HungarianAlgorithm {
public:
    HungarianAlgorithm();
    ~HungarianAlgorithm();

    // Main function to solve the assignment problem
    double Solve(int size, std::vector<double>& distMatrix, std::vector<int>& Assignment);

private:
    // Helper functions
    void assignmentoptimal(int* assignment, double* cost, double* distMatrix, int nOfRows, int nOfColumns);
    void buildassignmentvector(int* assignment, const char* starMatrix, int nOfRows, int nOfColumns);
    void computeassignmentcost(int* assignment, double* cost, const double* distMatrix, int nOfRows);

    // Steps of the Hungarian algorithm
    void step2a(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
        char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step2b(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
        char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step3(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
        char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step4(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
        char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
    void step5(int* assignment, double* distMatrix, char* starMatrix, char* newStarMatrix,
        char* primeMatrix, char* coveredColumns, char* coveredRows, int nOfRows, int nOfColumns, int minDim);
};

#endif // HUNGARIAN_H
