/*
    * Smith-Waterman Algorithm Implementation in C++
    * This implementation performs local sequence alignment using the Smith-Waterman algorithm.
    * It computes the optimal local alignment score and the corresponding aligned sequences.
    * Parallelized using OpenMP for performance improvement on large sequences.
*/
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

struct AlignmentResult {
    int score;
    std::string aligned1;
    std::string aligned2;
};

int main(int argc, char* argv[]) { 
    printf("Smith-Waterman Local Alignment\n");

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <sequence1> <sequence2> [gap] [match] [mismatch]\n", argv[0]);
        exit(1);
    }

    std::string s1 = argv[1];
    std::string s2 = argv[2];
    int gap = argc > 3 ? std::atoi(argv[3]) : -2; // Default gap penalty
    int match = argc > 4 ? std::atoi(argv[4]) : 3; // Default match score
    int mismatch = argc > 5 ? std::atoi(argv[5]) : -3; // Default mismatch penalty

    int s1_len = s1.length(); 
    int s2_len = s2.length(); 

    std::vector<std::vector<int>> ScoringMatrix(s1_len + 1, std::vector<int>(s2_len + 1, 0));
    int maxScore = 0;
    int maxI = 0, maxJ = 0;

    // Fill the DP table
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= s1_len; ++i) {
        for (int j = 1; j <= s2_len; ++j) {
            int diag = ScoringMatrix[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? match : mismatch);
            int up = ScoringMatrix[i - 1][j] + gap;
            int left = ScoringMatrix[i][j - 1] + gap;
            
            ScoringMatrix[i][j] = std::max({0, diag, up, left});
            
            if (ScoringMatrix[i][j] > maxScore) {
                maxScore = ScoringMatrix[i][j];
                maxI = i;
                maxJ = j;
            }
        }
    }

    // Traceback from the highest scoring cell
    std::string align1 = "";
    std::string align2 = "";
    int i = maxI, j = maxJ;

    while (i > 0 && j > 0 && ScoringMatrix[i][j] > 0) {
        int diagScore = ScoringMatrix[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? match : mismatch);
        int upScore = ScoringMatrix[i - 1][j] + gap;
        int leftScore = ScoringMatrix[i][j - 1] + gap;
        if (ScoringMatrix[i][j] == diagScore) {
            align1 += s1[i - 1];
            align2 += s2[j - 1];
            i--; j--;
        } else if (ScoringMatrix[i][j] == upScore) {
            align1 += s1[i - 1];
            align2 += '-';
            i--;
        } else {
            align1 += '-';
            align2 += s2[j - 1];
            j--;
        }
    }

    std::reverse(align1.begin(), align1.end());
    std::reverse(align2.begin(), align2.end());

    printf("Alignment complete. Score: %d\n", maxScore);
    printf("Aligned Sequences:\n\t%s\n\t%s\n", align1.c_str(), align2.c_str());
}