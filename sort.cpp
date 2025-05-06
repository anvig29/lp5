#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

void sequentialBubbleSort(std::vector<int> &arr)
{
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(std::vector<int> &arr)
{
#pragma omp parallel sections
    {
#pragma omp section
        sequentialBubbleSort(arr);
    }
}

void merge(std::vector<int> &arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;
    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; ++i)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void sequentialMergeSort(std::vector<int> &arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;
        sequentialMergeSort(arr, left, mid);
        sequentialMergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void parallelMergeSort(std::vector<int> &arr, int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

// Parallelize the recursive calls
    #pragma omp parallel sections
        {
    #pragma omp section
            sequentialMergeSort(arr, left, mid);

    #pragma omp section
            sequentialMergeSort(arr, mid + 1, right);

    #pragma omp section
            merge(arr, left, mid, right);
        }
    }
}

void printArray(const std::vector<int> &arr)
{
    for (const auto &num : arr)
        std::cout << num << " ";
    std::cout << std::endl;
    /*
int count = std::min(20, (int)arr.size());
    for (int i = 0; i < count; ++i)
        std::cout << arr[i] << " ";
    if (arr.size() > count)
        std::cout << "...";
    std::cout << std::endl;*/
}

int main()
{
    int size;
    cout << "Enter the number of elements:" << endl;
    cin >> size;
    std::vector<int> arr(size);

    for (int i = 0; i < size; ++i)
    {
        arr[i] = rand() % 1000;
    }
    
    //omp_set_num_threads(omp_get_max_threads);

    std::vector<int> arr1 = arr;
    double start_time = omp_get_wtime();
    sequentialBubbleSort(arr1);
    double end_time = omp_get_wtime();
    double sequential_time_bubble = (end_time - start_time);
    std::cout << "Sequential Bubble Sort Time: " << sequential_time_bubble << " seconds." << std::endl;

    std::vector<int> arr2 = arr;
    start_time = omp_get_wtime();
    parallelBubbleSort(arr2);
    end_time = omp_get_wtime();
    double parallel_time_bubble = (end_time - start_time);
    std::cout << "Parallel Bubble Sort Time: " << parallel_time_bubble << " seconds." << std::endl;
    double speedupBubble = sequential_time_bubble / parallel_time_bubble;
    std::cout << "Bubble Sort Speedup: " << speedupBubble << std::endl;

    std::vector<int> arr3 = arr;
    start_time = omp_get_wtime();
    sequentialMergeSort(arr3, 0, arr3.size() - 1);
    end_time = omp_get_wtime();
    double sequential_time_merge = (end_time - start_time);
    std::cout << "Sequential Merge Sort Time: " << sequential_time_merge << " seconds." << std::endl;

    std::vector<int> arr4 = arr;
    start_time = omp_get_wtime();
    parallelMergeSort(arr4, 0, arr4.size() - 1);
    end_time = omp_get_wtime();
    double parallel_time_merge = (end_time - start_time);
    std::cout << "Parallel Merge Sort Time: " << parallel_time_merge << " seconds." << std::endl;
    double speedupMerge = sequential_time_merge / parallel_time_merge;
    std::cout << "Merge Sort Speedup: " << speedupMerge << std::endl;

    /* cout << "\nOriginal array (first 20 elements):\n";
    printArray(arr);

    cout << "\nSequential Bubble Sort (first 20 elements):\n";
    printArray(arr1);

    cout << "\nParallel Bubble Sort (first 20 elements):\n";
    printArray(arr2);

    cout << "\nSequential Merge Sort (first 20 elements):\n";
    printArray(arr3);

    cout << "\nParallel Merge Sort (first 20 elements):\n";
    printArray(arr4);

    bool bubble_match = (arr1 == arr2);
    bool merge_seq_match = (arr1 == arr3);
    bool merge_par_match = (arr1 == arr4);

    cout << "\n=== Verification Results ===" << endl;
    cout << "Parallel Bubble == Sequential Bubble? " << (bubble_match ? "Yes" : "No") << endl;
    cout << "Sequential Merge == Sequential Bubble? " << (merge_seq_match ? "Yes" : "No") << endl;*/
    //g++ -fopenmp sort.cpp -o s  
    //./s
    return 0;
}
