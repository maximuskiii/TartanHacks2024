#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

void printPath(const vector<pair<int, int>> &path)
{
    for (const auto &p : path)
    {
        cout << "(" << p.first << ", " << p.second << ") ";
    }
    cout << endl;
}

void selectWaypoints(const vector<vector<int>> &grid, vector<pair<int, int>> &path, int viewSize)
{
    const int rows = grid.size();
    const int cols = grid[0].size();
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));

    
    for (int i = 0; i < rows; i += viewSize)
    {
        for (int j = 0; j < cols; j += viewSize)
        {
            
            bool waypointSet = false;
            for (int di = 0; di < viewSize && i + di < rows && !waypointSet; ++di)
            {
                for (int dj = 0; dj < viewSize && j + dj < cols && !waypointSet; ++dj)
                {
                    if (grid[i + di][j + dj] == 1 && !visited[i + di][j + dj])
                    {
                        path.push_back({i + di, j + dj});
                        waypointSet = true;
                        
                        for (int vi = 0; vi < viewSize && i + vi < rows; ++vi)
                        {
                            for (int vj = 0; vj < viewSize && j + vj < cols; ++vj)
                            {
                                visited[i + vi][j + vj] = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

int main()
{
    vector<vector<int>> grid;
    string line;
    ifstream inputFile("input_grid.txt");

    while (getline(inputFile, line))
    {
        vector<int> row;
        istringstream iss(line);
        int val;
        while (iss >> val)
        {
            row.push_back(val);
        }
        grid.push_back(row);
    }

    vector<pair<int, int>> path; 

    selectWaypoints(grid, path, 40);

    ofstream outputFile("output_path.txt");
    for (const auto &p : path)
    {
        outputFile << p.first << " " << p.second << "\n";
    }
    outputFile.close();

    printPath(path);

    return 0;
}
