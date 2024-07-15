#include <iostream>
#include <memory>
#include <chrono>

#include "RNNDescent.h"
#include "utils/io.hpp"

using namespace std;

void stat(const vector<vector<unsigned>> &graph)
{
    size_t max_edge = 0;
    size_t min_edge = graph.size();
    size_t avg_edge = 0;
    for (auto &nbhood : graph)
    {
        auto size = nbhood.size();
        max_edge = std::max(max_edge, size);
        min_edge = std::min(min_edge, size);
        avg_edge += size;
    }
    std::cout << "max_edge = " << max_edge << "\nmin_edge = " << min_edge << "\navg_edge = " << (1.0 * avg_edge / graph.size()) << "\n";
}

int main(int argc, char *argv[])
{

    std::string base_path = "/home/jeff/data/sift1m/sift1m_base.fvecs";
    string sav_pth{"out_index.ivecs"};

    if (argc > 1)
        base_path = argv[1];
    if (argc > 2)
        sav_pth = argv[2];

    rnndescent::rnn_para para;
    para.S = 36;
    para.T1 = 3;
    para.T2 = 8;

    rnndescent::Matrix<float> base_data;
    base_data.load(base_path, 128, 0, 4);


    // size_t data_size;
    // base_data.resize(data_size, 100);
    // for (unsigned id = 0; id < data_size; ++id) {
    //     base_data.add_test((float*)(data + (size_t)id * 416 + 8));
    // }

    rnndescent::MatrixOracle<float, rnndescent::metric::l2sqr> oracle(base_data);

    std::unique_ptr<rnndescent::RNNDescent> index(new rnndescent::RNNDescent(oracle, para));

    auto start = chrono::high_resolution_clock::now();
    index->build(oracle.size(), true);
    auto end = chrono::high_resolution_clock::now();

    cout << "Elapsed time in milliseconds: "
         << 1.0 * std::chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000
         << " s" << endl;

    std::cout << "sav_pth = " << sav_pth << "\n";
    std::vector<std::vector<unsigned>> index_graph;
    index->extract_index_graph(index_graph);

    stat(index_graph);

    IO::saveBinVec(sav_pth, index_graph);

    return 0;
}