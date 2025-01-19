#include <iostream>
#include "Hungarian.h"
#include <fstream>
#include <vector>
#include <sstream>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <queue> 
#include <cmath>
#include <omp.h>
#include <mkl.h>
#include <iomanip>

#include <mpi.h>

using namespace std;
#define MKL_STATIC
#define MAX_COST 10000

const int INF = numeric_limits<int>::max();

void generateRandomCostMatrix(int size, std::vector<double>& costMatrix, int mpi_rank) {

    int chunk_size = 64; // Размер блока

    // Проверка корректности размера
    if (size <= 0) {
        std::cerr << "Invalid matrix size: " << size << "\n";
        return;
    }

    // Устанавливаем размер массива в формате MKL (Column-Major)
    costMatrix.resize(size * size);

    // Создаём массив для случайных чисел
    std::vector<double> randomValues(size * size);

    // Генерация случайных чисел от 1.0 до MAX_COST + 1.0
    VSLStreamStatePtr stream;
    unsigned int seed = static_cast<unsigned int>(time(nullptr)) + mpi_rank; // Уникальный seed для каждого процесса
    vslNewStream(&stream, VSL_BRNG_MT19937, seed);
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, size * size, randomValues.data(), 1.0, MAX_COST + 1.0);
    vslDeleteStream(&stream);

    // Заполняем матрицу в формате Column-Major
#pragma omp parallel for collapse(2) schedule(dynamic, chunk_size)
    for (int col = 0; col < size; ++col) {
        for (int row = 0; row < size; ++row) {
            if (row != col) {
                int intValue = static_cast<int>(randomValues[row + col * size]);
                costMatrix[col * size + row] = static_cast<double>(intValue);
            }
            else {
                costMatrix[col * size + row] = INF; // Диагональные элементы равны бесконечности
            }
        }
    }
}


// Функция для нахождения циклов в графе
double calculateAssignmentCost(const std::vector<double>& costMatrix, const std::vector<int>& match, int size) {
    std::vector<double> selectedElements(size);

    // Copy selected elements using MKL
    for (int i = 0; i < size; ++i) {
        selectedElements[i] = costMatrix[i + match[i] * size];
    }

    // Sum elements using MKL
    double totalCost = cblas_dasum(size, selectedElements.data(), 1);

    return totalCost;
}


vector<vector<int>> findCycles(const vector<int>& match) {
    int n = match.size();
    vector<bool> visited(n, false);
    vector<vector<int>> cycles;

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            vector<int> cycle;
            int current = i;
            while (!visited[current]) {
                visited[current] = true;
                cycle.push_back(current);
                current = match[current];
            }
            cycles.push_back(move(cycle));
        }
    }
    return cycles;
}

// Функция для поиска минимального комбинированного цикла
int findMinimumCombinedCycle(const std::vector<double>& costMatrix, const std::vector<int>& baseCycle, const std::vector<int>& otherCycle, std::vector<int>& combinedCycle, int size) {
    double minWeight = std::numeric_limits<double>::max();
    int chunk_size = 64;
#pragma omp parallel for collapse(2) schedule(dynamic, chunk_size) reduction(min:minWeight)
    for (size_t i = 0; i < baseCycle.size(); ++i) {
        for (size_t j = 0; j < otherCycle.size(); ++j) {
            int v1 = baseCycle[i];
            int v2 = baseCycle[(i + 1) % baseCycle.size()];
            int u1 = otherCycle[j];
            int u2 = otherCycle[(j + 1) % otherCycle.size()];

            double costDiff = costMatrix[v1 + u2 * size] + costMatrix[u1 + v2 * size]
                - costMatrix[v1 + v2 * size] - costMatrix[u1 + u2 * size];

            if (costDiff < minWeight) {
                minWeight = costDiff;

#pragma omp critical
                {
                    combinedCycle.clear();
                    combinedCycle.insert(combinedCycle.end(), baseCycle.begin(), baseCycle.begin() + i + 1);
                    combinedCycle.insert(combinedCycle.end(), otherCycle.begin() + j + 1, otherCycle.end());
                    combinedCycle.insert(combinedCycle.end(), otherCycle.begin(), otherCycle.begin() + j + 1);
                    combinedCycle.insert(combinedCycle.end(), baseCycle.begin() + i + 1, baseCycle.end());
                }
            }
        }
    }

    return minWeight;
}


// Function to calculate the cost of a combined cycle
double calculateCycleCost(const std::vector<double>& costMatrix, const std::vector<int>& cycle, int size) {
    int n = cycle.size();
    std::vector<double> selectedElements(n);

    // Извлекаем элементы, соответствующие ребрам цикла
    for (int i = 0; i < n; ++i) {
        int from = cycle[i];
        int to = cycle[(i + 1) % n];
        selectedElements[i] = costMatrix[from + to * size];
    }

    // Вычисляем сумму с помощью MKL
    double totalCost = cblas_dasum(n, selectedElements.data(), 1);

    return totalCost;
}

// Параллельная версия алгоритма соединения циклов
std::vector<int> CycleMergingAlgorithm(const std::vector<double>& costMatrix, const std::vector<int>& cycleCover, double& CMACost, int size) {
    std::vector<std::vector<int>> cycles = findCycles(cycleCover);
    auto maxIt = std::max_element(cycles.begin(), cycles.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        return a.size() < b.size();
        });
    std::vector<int> baseCycle = *maxIt;
    cycles.erase(maxIt);
    int chunk_size = 64;
    while (!cycles.empty()) {
        double minCombinedWeight = std::numeric_limits<double>::max();
        std::vector<int> bestCombinedCycle;
        int bestIndex = -1;

#pragma omp parallel for schedule(dynamic, chunk_size) reduction(min:minCombinedWeight)
        for (size_t i = 0; i < cycles.size(); ++i) {
            std::vector<int> combinedCycle;
            double combinedWeight = findMinimumCombinedCycle(costMatrix, baseCycle, cycles[i], combinedCycle, size);

#pragma omp critical
            {
                if (combinedWeight < minCombinedWeight) {
                    minCombinedWeight = combinedWeight;
                    bestCombinedCycle = combinedCycle;
                    bestIndex = i;
                }
            }
        }

        if (bestIndex != -1) {
            baseCycle = bestCombinedCycle;
            cycles.erase(cycles.begin() + bestIndex);
        }
    }

    CMACost = calculateCycleCost(costMatrix, baseCycle, size);
    return baseCycle;
}



struct Result {
    double totalAssignmentCost;
    double totalAssignmentTime;
    double totalParallelTSPSolutionCost;
    double totalParallelTime;
};

// Функция для создания пользовательского типа MPI для структуры Result
void create_mpi_result_type(MPI_Datatype& mpi_result_type) {
    int count = 4;
    int block_lengths[4] = { 1, 1, 1, 1 };
    MPI_Datatype types[4] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    MPI_Aint offsets[4];

    offsets[0] = offsetof(Result, totalAssignmentCost);
    offsets[1] = offsetof(Result, totalAssignmentTime);
    offsets[2] = offsetof(Result, totalParallelTSPSolutionCost);
    offsets[3] = offsetof(Result, totalParallelTime);

    MPI_Type_create_struct(count, block_lengths, offsets, types, &mpi_result_type);
    MPI_Type_commit(&mpi_result_type);
}

Result solve_tsp_task(const std::vector<double>& task_data, int task_size) {

    Result result = { 0.0, 0.0, 0.0, 0.0 };

    int vertices = task_size;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> costMatrix;
    generateRandomCostMatrix(vertices, costMatrix, rank);

    auto startAssPrb = chrono::high_resolution_clock::now();
    HungarianAlgorithm HungAlgo;
    vector<int> assignment;
    double assignmentCost = HungAlgo.Solve(vertices, costMatrix, assignment);
    vector<int> cycleCover = assignment;
    if (cycleCover.empty()) {
        cout << "Error" << endl;
        return result;
    }
    auto endAssPrb = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedAssPrb = endAssPrb - startAssPrb;

    // Параллельное решение
    auto startPar = chrono::high_resolution_clock::now();
    double CMACost = 0;
    vector<int> CMACycle = CycleMergingAlgorithm(costMatrix, cycleCover, CMACost, vertices);
    auto endPar = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedPar = endPar - startPar;

    result.totalAssignmentCost = assignmentCost;
    result.totalAssignmentTime = elapsedAssPrb.count();
    result.totalParallelTSPSolutionCost = CMACost;
    result.totalParallelTime = elapsedPar.count();

    cout << "vertices: " << vertices << endl;

    return result;
}

// Мастер-процесс распределяет задачи и собирает результаты
void master_process(int num_tasks, int num_workers, int task_size, ofstream& outputFile, MPI_Datatype mpi_result_type) {
    vector<double> task_data(task_size);
    int task_index = 0;
    int completed_tasks = 0;

    vector<MPI_Request> send_requests(num_workers);
    vector<MPI_Request> recv_requests(num_workers);
    vector<Result> recv_results(num_workers);
    MPI_Status status;

    // Инициализация задач и отправка первым `num_workers` воркерам
    for (int worker = 1; worker <= num_workers && task_index < num_tasks; worker++, task_index++) {
        MPI_Isend(task_data.data(), task_size, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD, &send_requests[worker - 1]);
        std::cout << "Task " << task_index + 1 << " sent to worker " << worker << std::endl;

        MPI_Irecv(&recv_results[worker - 1], 1, mpi_result_type, worker, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_requests[worker - 1]);
    }

    // Обработка завершённых задач
    while (completed_tasks < num_tasks) {
        int index;
        MPI_Waitany(num_workers, recv_requests.data(), &index, &status);

        completed_tasks++;
        Result result = recv_results[index];

        // Запись результата
        outputFile << task_size << "\t"
            << result.totalAssignmentCost << "\t"
            << result.totalAssignmentTime << "\t"
            << result.totalParallelTSPSolutionCost << "\t"
            << result.totalParallelTime << endl;

        // Отправка новой задачи или завершения
        if (task_index < num_tasks) {
            MPI_Isend(task_data.data(), task_size, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &send_requests[status.MPI_SOURCE - 1]);
            MPI_Irecv(&recv_results[status.MPI_SOURCE - 1], 1, mpi_result_type, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_requests[status.MPI_SOURCE - 1]);
            std::cout << "Task " << task_index + 1 << " sent to worker " << status.MPI_SOURCE << std::endl;
            task_index++;
        }
        else {
            MPI_Isend(NULL, 0, MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &send_requests[status.MPI_SOURCE - 1]);
        }
    }

    // Дождаться завершения всех отправок
    MPI_Waitall(num_workers, send_requests.data(), MPI_STATUSES_IGNORE);
}

// Worker-процессы
void worker_process(int task_size, MPI_Datatype mpi_result_type) {
    vector<double> task_data(task_size);
    Result result;
    MPI_Request send_request, recv_request;
    MPI_Status status;

    while (true) {
        // Асинхронное получение задачи
        MPI_Irecv(task_data.data(), task_size, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
        MPI_Wait(&recv_request, &status);

        if (status.MPI_TAG == 1) break; // Сигнал завершения

        // Решение задачи
        result = solve_tsp_task(task_data, task_size);

        // Асинхронная отправка результата
        MPI_Isend(&result, 1, mpi_result_type, 0, 0, MPI_COMM_WORLD, &send_request);
        MPI_Wait(&send_request, MPI_STATUSES_IGNORE);
    }
}

int main(int argc, char* argv[])
{
    omp_set_num_threads(omp_get_max_threads()); // Установить максимальное число потоков
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Создаем MPI-тип для структуры Result
    MPI_Datatype mpi_result_type;
    create_mpi_result_type(mpi_result_type);

    // Открываем файл для записи (только в мастер-процессе)
    ofstream outputFile;
    if (rank == 0) {
        outputFile.open("results.txt");
        outputFile << "Number of vertices\tAssignment cost\tAssignment time (s)\tSequential TSP cost\tSequential time (s)\tParallel TSP cost\tParallel time (s)" << endl;
    }

    for (int vertices = 100; vertices <= 3000; vertices += 100) {
        int numTrials = 10;

        for (int trial = 0; trial < numTrials; ++trial) {
            if (rank == 0) {
                // Мастер процесс
                master_process(numTrials, size - 1, vertices, outputFile, mpi_result_type);
            }
            else {
                // Worker процессы
                worker_process(vertices, mpi_result_type);
            }
        }


    }

    // Закрываем файл (только в мастер-процессе)
    if (rank == 0) {
        cout << "That's all";
        outputFile.close();
    }


    MPI_Barrier(MPI_COMM_WORLD); // Синхронизация
    // Освобождаем пользовательский MPI-тип
    MPI_Type_free(&mpi_result_type);

    MPI_Finalize();
    return 0;
}
