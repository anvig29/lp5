#include <iostream>
#include <omp.h>
#include <vector>
#include <stack>
#include <queue>
#include <chrono>
#include <random>

using namespace std;

class Graph {
public:
    int vertices;
    vector<vector<int>> graph;
    vector<bool> visited;

    Graph(int v) {
        vertices = v;
        graph.resize(vertices);
        visited.resize(vertices, false);
    }

    void addEdge(int a, int b) {
        graph[a].push_back(b);
        graph[b].push_back(a);
    }

    void generateRandomGraph(int edges) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, vertices - 1);

        for (int i = 0; i < edges; i++) {
            int u = dis(gen);
            int v = dis(gen);
            if (u != v) addEdge(u, v);
        }
    }

    void initialize_visited() {
        fill(visited.begin(), visited.end(), false);
    }

    // Sequential DFS
    void dfs(int start) {
        stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int current = s.top();
            s.pop();
            for (int neighbor : graph[current]) {
                if (!visited[neighbor]) {
                    s.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
    }

    void parallel_dfs(int start) {
        stack<int> s;
        vector<bool> local_visited(vertices, false);
        s.push(start);
        local_visited[start] = true;

        #pragma omp parallel
        {
            while (true) {
                int current;
                bool has_work = false;

                // Critical section to safely access the stack
                #pragma omp critical
                {
                    if (!s.empty()) {
                        current = s.top();
                        s.pop();
                        has_work = true;
                    }
                }

                if (!has_work) break;  // Exit when there is no more work

                // Parallel for loop to process neighbors
                #pragma omp parallel for
                for (size_t j = 0; j < graph[current].size(); j++) {
                    int neighbor = graph[current][j];

                    if (!local_visited[neighbor]) {
                        #pragma omp critical
                        {
                            s.push(neighbor);
                            local_visited[neighbor] = true;
                        }
                    }
                }
            }
        }
    }

    // Sequential BFS
    void bfs(int start) {
        queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            for (int neighbor : graph[current]) {
                if (!visited[neighbor]) {
                    q.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
    }

    // Parallel BFS using OpenMP
    void parallel_bfs(int start) {
        queue<int> q;
        vector<bool> local_visited(vertices, false);
        q.push(start);
        local_visited[start] = true;

        while (!q.empty()) {
            int level_size = q.size();
            vector<int> current_level;

            // Process all nodes at the current level
            #pragma omp parallel for
            for (int j = 0; j < level_size; j++) {
                int current;
                #pragma omp critical
                {
                    if (!q.empty()) {
                        current = q.front();
                        q.pop();
                        current_level.push_back(current);
                    }
                }
            }

            // Process neighbors of all nodes in current_level
            #pragma omp parallel for
            for (int j = 0; j < current_level.size(); j++) {
                for (int neighbor : graph[current_level[j]]) {
                    if (!local_visited[neighbor]) {
                        #pragma omp critical
                        {
                            q.push(neighbor);
                            local_visited[neighbor] = true;
                        }
                    }
                }
            }
        }
    }
};

int main() {
    int numVertices;
    cout<<"Enter number of vertices: ";
    cin>>numVertices;  
   
    int numEdges;   
    cout<<"Enter number of edges: ";
    cin>>numEdges; 
    Graph g(numVertices);
    g.generateRandomGraph(numEdges);
    
    omp_set_num_threads(omp_get_max_threads);
    // Sequential DFS
    auto start = chrono::high_resolution_clock::now();
    g.initialize_visited();
    g.dfs(0);
    auto end = chrono::high_resolution_clock::now();
    double time_dfs = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "DFS Time: " << time_dfs << " ms" << endl;

    // Parallel DFS
    start = chrono::high_resolution_clock::now();
    g.initialize_visited();
    g.parallel_dfs(0);
    end = chrono::high_resolution_clock::now();
    double time_parallel_dfs = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Parallel DFS Time: " << time_parallel_dfs << " ms" << endl;
    cout << "Speedup: " << time_dfs / time_parallel_dfs << "\n" << endl;

    // Sequential BFS
    start = chrono::high_resolution_clock::now();
    g.initialize_visited();
    g.bfs(0);
    end = chrono::high_resolution_clock::now();
    double time_bfs = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "BFS Time: " << time_bfs << " ms" << endl;

    // Parallel BFS
    start = chrono::high_resolution_clock::now();
    g.initialize_visited();
    g.parallel_bfs(0);
    end = chrono::high_resolution_clock::now();
    double time_parallel_bfs = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Parallel BFS Time: " << time_parallel_bfs << " ms" << endl;
    cout << "Speedup: " << time_bfs / time_parallel_bfs << "x\n" << endl;
    //g++ -fopenmp reduc.cpp -o s  
    //./s
    return 0;
}
