#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <climits>

using namespace std;

void min_seq(int* arr, int n, double& time_taken, int& result) {
    int min_val = INT_MAX;
    double start = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val) min_val = arr[i];
    }
    double end = omp_get_wtime();
    result = min_val;
    time_taken = end - start;
}

void min_par(int* arr, int n, double& time_taken, int& result) {
    int min_val = INT_MAX;
    double start = omp_get_wtime();
    #pragma omp parallel for reduction(min:min_val) schedule(static)
    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val) min_val = arr[i];
    }
    double end = omp_get_wtime();
    result = min_val;
    time_taken = end - start;
}

void max_seq(int* arr, int n, double& time_taken, int& result) {
    int max_val = INT_MIN;
    double start = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    double end = omp_get_wtime();
    result = max_val;
    time_taken = end - start;
}

void max_par(int* arr, int n, double& time_taken, int& result) {
    int max_val = INT_MIN;
    double start = omp_get_wtime();
    #pragma omp parallel for reduction(max:max_val) schedule(static)
    for (int i = 0; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    double end = omp_get_wtime();
    result = max_val;
    time_taken = end - start;
}

void avg_seq(int* arr, int n, double& time_taken, double& result, long long& sum_out) {
    long long sum = 0;
    double start = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    result = static_cast<double>(sum) / n;
    sum_out = sum;
    double end = omp_get_wtime();
    time_taken = end - start;
}

void avg_par(int* arr, int n, double& time_taken, double& result, long long& sum_out) {
    long long sum = 0;
    double start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    result = static_cast<double>(sum) / n;
    sum_out = sum;
    double end = omp_get_wtime();
    time_taken = end - start;
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    int* arr = new int[n];
    srand(time(0));

    omp_set_num_threads(omp_get_max_threads()); // Use all cores

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }

    cout << "\nArray size: " << n << endl;

    double t_seq, t_par;
    int result_seq_int, result_par_int;
    double result_seq_double, result_par_double;
    long long sum_seq, sum_par;

    // MIN
    min_seq(arr, n, t_seq, result_seq_int);
    min_par(arr, n, t_par, result_par_int);
    cout << "\nMIN:\nSequential = " << result_seq_int << " | Time: " << t_seq;
    cout << "\nParallel   = " << result_par_int << " | Time: " << t_par;
    cout << "\nSpeedup(min) = " << t_seq / t_par << "\n";

    // MAX
    max_seq(arr, n, t_seq, result_seq_int);
    max_par(arr, n, t_par, result_par_int);
    cout << "\nMAX:\nSequential = " << result_seq_int << " | Time: " << t_seq;
    cout << "\nParallel   = " << result_par_int << " | Time: " << t_par;
    cout << "\nSpeedup(max) = " << t_seq / t_par << "\n";

   // AVG + SUM
avg_seq(arr, n, t_seq, result_seq_double, sum_seq);
avg_par(arr, n, t_par, result_par_double, sum_par);

cout << "\nSUM:\nSequential = " << t_seq << " | Parallel = " << t_par;
cout << "\nSpeedup(sum) = " << t_seq / t_par;
cout<<endl;
cout << "\nAVG:\nSequential = " << result_seq_double << " | Time: " << t_seq;
cout << "\nParallel   = " << result_par_double << " | Time: " << t_par;
cout << "\nSpeedup(avg) = " << t_seq / t_par << "\n";

delete[] arr;
return 0;
//g++ -fopenmp reduc.cpp -o s  
//./s
}
