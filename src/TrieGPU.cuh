// src/TrieGPU.cuh

#ifndef TRIE_GPU_CUH
#define TRIE_GPU_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>

#define ALPHABET_SIZE 26  // For lowercase English letters
#define MAX_NODES 1000000 // Maximum number of nodes in the trie

struct TrieNode
{
    int children[ALPHABET_SIZE]; // Indices of child nodes
    bool isEndOfWord;
};

class TrieGPU
{
public:
    TrieGPU();
    ~TrieGPU();

    void insertWords(const std::vector<std::string> &words);
    std::vector<char> searchWords(const std::vector<std::string> &queries);

private:
    TrieNode *d_trie;
    int *d_nodeCounter;
};

#endif // TRIE_GPU_CUH