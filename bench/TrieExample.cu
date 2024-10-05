// bench/TrieExample.cu

#include <iostream>
#include <vector>
#include <string>
#include "../src/TrieGPU.cuh"

int main()
{
    // Initialize trie
    TrieGPU trie;

    // Words to insert
    std::vector<std::string> words = {"apple", "app", "ape", "banana", "band", "bandana", "bandit"};
    trie.insertWords(words);

    // Words to search
    std::vector<std::string> queries = {"apple", "app", "apex", "band", "bandage", "bandit", "cat"};

    // Search and get results
    std::vector<char> results = trie.searchWords(queries);

    // Display results
    for (size_t i = 0; i < queries.size(); ++i)
    {
        std::cout << queries[i] << ": " << (results[i] ? "Found" : "Not Found") << std::endl;
    }

    return 0;
}