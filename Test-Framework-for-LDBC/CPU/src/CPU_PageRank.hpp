#pragma once
#include <graph_structure/graph_structure.hpp>
#include <vector>
#include <iostream>
#include <ThreadPool.h>
#include <algorithm>

std::vector<double> PageRank(std::vector<std::vector<std::pair<int, double>>>& in_edge,
    std::vector<std::vector<std::pair<int, double>>>& out_edge, double damp, int iters) {

    int N = in_edge.size();

    std::vector<double> rank(N, 1 / N);
    std::vector<double> new_rank(N);

    double d = damp;
    double teleport = (1 - damp) / N;

    std::vector<int> sink;
    for (int i = 0; i < N; i++)
    {
        if (out_edge[i].size() == 0)
            sink.push_back(i);
    }

    for (int i = 0; i < iters; i++) {
        double sink_sum = 0;
        for (int i = 0; i < sink.size(); i++)
        {
            sink_sum += rank[sink[i]];
        }

        double x = sink_sum * d / N + teleport;

        ThreadPool pool_dynamic(100);
        std::vector<std::future<int>> results_dynamic;
        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &rank, &out_edge, &new_rank, &x]
                {
                    int start = q * N / 100, end = std::min(N - 1, (q + 1) * N / 100);
                    for (int i = start; i <= end; i++) {
                        rank[i] /= out_edge[i].size();
                        new_rank[i] = x;
                    }

                    return 1; }));
        }
        for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);

        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &in_edge, &rank, &new_rank, &d]
                {
                    int start = q * N / 100, end = std::min(N - 1, (q + 1) * N / 100);
                    for (int v = start; v <= end; v++) {
                        for (auto& y : in_edge[v]) {
                            new_rank[v] += d * rank[y.first];
                        }
                    }
                    return 1; }));
        }
        for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);


        rank.swap(new_rank);
    }
    return rank;
}

/*std::map<long long int, double> PR_Bind_node(LDBC<double> & graph){

    vector<double> prValueVec =  PageRank(graph.INs, graph.OUTs, graph.pr_damping, graph.pr_its);
    std::map<long long int, double> strId2value;

    std::vector<long long int> converted_numbers;

    for (const auto& str : graph.vertex_id_to_str) {
        long long int num = std::stoll(str);
        converted_numbers.push_back(num);
    }

    std::sort(converted_numbers.begin(), converted_numbers.end());

	for( int i = 0; i < prValueVec.size(); ++i){
		strId2value.emplace(converted_numbers[i], prValueVec[i]);
    }

	// std::string path = "../data/cpu_pr_75.txt";
	// storeResult(strId2value, path);//ldbc file

    return strId2value;

}

std::vector<std::string> PageRank_v2(LDBC<double> & graph) {

    std::vector<std::string> PageRankVec;
    vector<double> prValueVec =  PageRank(graph.INs, graph.OUTs, graph.pr_damping, graph.pr_its);

    for(auto & it : prValueVec){
        PageRankVec.push_back(std::to_string(it));
    }
    
    return PageRankVec;
}*/

std::vector<std::pair<std::string, double>> CPU_PR(graph_structure<double>& graph, int iterations, double damping) {
    std::vector<double> prValueVec = PageRank(graph.INs, graph.OUTs, damping, iterations);
    return graph.res_trans_id_val(prValueVec);
}