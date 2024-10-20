#pragma once

#include <GPU_csr/GPU_vector.cuh>

#include <utility>

template <typename T>
__host__ int binary_insert(cuda_vector<std::pair<int, T>>* vec, int key, T load) {
    int left = 0, right = vec->size() - 1;
    if (right > -1) {
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (vec->operator[](mid).first == key) {
                vec->operator[](mid).second = load;
                return mid;
            }
            else if (vec->operator[](mid).first > key)
                right = mid - 1;
            else
                left = mid + 1;
        }
    }
    vec->insert(left, std::make_pair(key, load));
    return left;
}

template <typename T>
__host__ void binary_earse(cuda_vector<std::pair<int, T>>* vec, int key) {
    int left = 0, right = vec->size() - 1;
    if (right > -1) {
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (vec->operator[](mid).first == key) {
                vec->remove(mid);
                return;
            }
            else if (vec->operator[](mid).first > key)
                right = mid - 1;
            else
                left = mid + 1;
        }
    }
}
