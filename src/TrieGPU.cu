// src/TrieGPU.cu

#include "TrieGPU.cuh"
#include <cuda.h>
#include <iostream>

__global__ void initTrieKernel(TrieNode *trie)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_NODES)
    {
        for (int i = 0; i < ALPHABET_SIZE; ++i)
        {
            trie[idx].children[i] = -1;
        }
        trie[idx].isEndOfWord = false;
    }
}

__global__ void insertKernel(TrieNode *trie, int *nodeCounter, char *d_words, int *d_wordOffsets, int *d_wordLengths, int numWords)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numWords)
        return;

    int offset = d_wordOffsets[idx];
    int length = d_wordLengths[idx];
    int nodeIdx = 0; // Start at root

    for (int i = 0; i < length; ++i)
    {
        int charIdx = d_words[offset + i] - 'a';
        int *childPtr = &trie[nodeIdx].children[charIdx];

        int childIdx = atomicCAS(childPtr, -1, atomicAdd(nodeCounter, 1));
        if (childIdx == -1)
        {
            nodeIdx = *childPtr;
        }
        else
        {
            nodeIdx = childIdx;
        }
    }
    trie[nodeIdx].isEndOfWord = true;
}

__global__ void searchKernel(TrieNode *trie, char *d_queries, int *d_queryOffsets, int *d_queryLengths, char *d_results, int numQueries)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numQueries)
        return;

    int offset = d_queryOffsets[idx];
    int length = d_queryLengths[idx];
    int nodeIdx = 0;

    for (int i = 0; i < length; ++i)
    {
        int charIdx = d_queries[offset + i] - 'a';
        nodeIdx = trie[nodeIdx].children[charIdx];
        if (nodeIdx == -1)
        {
            d_results[idx] = 0; // Not found
            return;
        }
    }
    d_results[idx] = trie[nodeIdx].isEndOfWord ? 1 : 0;
}

// Constructor
TrieGPU::TrieGPU()
{
    cudaMalloc((void **)&d_trie, MAX_NODES * sizeof(TrieNode));
    cudaMalloc((void **)&d_nodeCounter, sizeof(int));

    // Initialize d_nodeCounter to 1 on the device
    int one = 1;
    cudaMemcpy(d_nodeCounter, &one, sizeof(int), cudaMemcpyHostToDevice);

    // Initialize trie nodes
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_NODES + threadsPerBlock - 1) / threadsPerBlock;
    initTrieKernel<<<blocksPerGrid, threadsPerBlock>>>(d_trie);
    cudaDeviceSynchronize();
}

// Destructor
TrieGPU::~TrieGPU()
{
    cudaFree(d_trie);
    cudaFree(d_nodeCounter);
}

void TrieGPU::insertWords(const std::vector<std::string> &words)
{
    int numWords = words.size();

    // Flatten words into a single array
    std::string allWords;
    std::vector<int> wordOffsets(numWords);
    std::vector<int> wordLengths(numWords);
    int offset = 0;
    for (size_t i = 0; i < words.size(); ++i)
    {
        allWords += words[i];
        wordOffsets[i] = offset;
        wordLengths[i] = words[i].size();
        offset += words[i].size();
    }

    // Copy data to device
    char *d_words;
    int *d_wordOffsets;
    int *d_wordLengths;
    cudaMalloc((void **)&d_words, allWords.size() * sizeof(char));
    cudaMalloc((void **)&d_wordOffsets, numWords * sizeof(int));
    cudaMalloc((void **)&d_wordLengths, numWords * sizeof(int));

    cudaMemcpy(d_words, allWords.c_str(), allWords.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wordOffsets, wordOffsets.data(), numWords * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wordLengths, wordLengths.data(), numWords * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the insert kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numWords + threadsPerBlock - 1) / threadsPerBlock;
    insertKernel<<<blocksPerGrid, threadsPerBlock>>>(d_trie, d_nodeCounter, d_words, d_wordOffsets, d_wordLengths, numWords);
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_words);
    cudaFree(d_wordOffsets);
    cudaFree(d_wordLengths);
}

std::vector<char> TrieGPU::searchWords(const std::vector<std::string> &queries)
{
    int numQueries = queries.size();

    // Flatten queries into a single array
    std::string allQueries;
    std::vector<int> queryOffsets(numQueries);
    std::vector<int> queryLengths(numQueries);
    int offset = 0;
    for (size_t i = 0; i < queries.size(); ++i)
    {
        allQueries += queries[i];
        queryOffsets[i] = offset;
        queryLengths[i] = queries[i].size();
        offset += queries[i].size();
    }

    // Copy query data to device
    char *d_queries;
    int *d_queryOffsets;
    int *d_queryLengths;
    char *d_results;
    cudaMalloc((void **)&d_queries, allQueries.size() * sizeof(char));
    cudaMalloc((void **)&d_queryOffsets, numQueries * sizeof(int));
    cudaMalloc((void **)&d_queryLengths, numQueries * sizeof(int));
    cudaMalloc((void **)&d_results, numQueries * sizeof(char));

    cudaMemcpy(d_queries, allQueries.c_str(), allQueries.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queryOffsets, queryOffsets.data(), numQueries * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queryLengths, queryLengths.data(), numQueries * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the search kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numQueries + threadsPerBlock - 1) / threadsPerBlock;
    searchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_trie, d_queries, d_queryOffsets, d_queryLengths, d_results, numQueries);
    cudaDeviceSynchronize();

    // Retrieve results
    std::vector<char> h_results(numQueries);
    cudaMemcpy(h_results.data(), d_results, numQueries * sizeof(char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_queries);
    cudaFree(d_queryOffsets);
    cudaFree(d_queryLengths);
    cudaFree(d_results);

    return h_results;
}