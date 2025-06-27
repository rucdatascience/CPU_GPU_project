#pragma once

#include <GPU_adj_list/GPU_vector.cuh>

#include <utility>

template <typename T>
__host__ int binary_insert(cuda_vector<int>* vec, cuda_vector<T>* vec_weight, int key, T load) {
    int left = 0, right = vec->size() - 1;
    if (right > -1) {
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (vec->operator[](mid) == key) {
                vec_weight->operator[](mid) = load;
                return mid;
            }
            else if (vec->operator[](mid) > key)
                right = mid - 1;
            else
                left = mid + 1;
        }
    }
    vec->insert(left, key);
    vec_weight->insert(left, load);
    return left;
}

template <typename T>
__host__ void binary_earse(cuda_vector<int>* vec, cuda_vector<T>* vec_weight, int key) {
    int left = 0, right = vec->size() - 1;
    if (right > -1) {
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (vec->operator[](mid) == key) {
                vec->remove(mid);
                vec_weight->remove(mid);
                return;
            }
            else if (vec->operator[](mid) > key)
                right = mid - 1;
            else
                left = mid + 1;
        }
    }
}
