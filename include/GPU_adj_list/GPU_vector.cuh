#pragma once

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cassert>
#include <utility>

#define DEFAULT_CAPACITY 50

typedef unsigned long long Rank;

template <typename T> class cuda_vector {

protected:
    Rank _size = 0, _capacity = 0;
    T* _elem;
    __host__ void copyFrom(const T* A, Rank lo, Rank hi);
    __host__ void expand();
    __host__ void expand(Rank n);
    __host__ void shrink();

public:
    __host__ cuda_vector(Rank c = DEFAULT_CAPACITY, Rank s = 0);
    __host__ cuda_vector(const T* A, Rank n);
    __host__ cuda_vector(const T* A, Rank lo, Rank hi);
    __host__ cuda_vector(const cuda_vector<T>& V);
    __host__ cuda_vector(const cuda_vector<T>& V, Rank lo, Rank hi);

    __host__ ~cuda_vector();

    // read-only
    __host__ __device__ Rank size() const { return _size; }
    __host__ __device__ bool empty() const { return !_size; }

    // read-write
    __host__ __device__ T& operator[](Rank r) const;
    __host__ cuda_vector<T>& operator=(const cuda_vector<T>& V);
    __host__ __device__ T* data() const { return _elem; }
    __host__ void resize(Rank n) { expand(n); _size = n; }
    __host__ void clear() { resize(0); }
    __host__ Rank remove(Rank lo, Rank hi);
    __host__ T remove(Rank r);
    __host__ Rank insert(Rank r, T const& e);
    __host__ Rank insert(T const& e) { return insert(_size, e); }
    __host__ Rank push_back(T const& e) { return insert(_size, e); }

};

template <typename T>
__host__ cuda_vector<T>::cuda_vector(Rank c, Rank s) {
    assert(s <= c);
    _capacity = c;
    _size = s;

    cudaError_t cudaStatus = cudaMallocManaged((void**)&_elem, c * sizeof(T));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMallocManaged failed!");
}

template <typename T>
__host__ cuda_vector<T>::cuda_vector(const T* A, Rank n) {
    copyFrom(A, 0, n);
}

template <typename T>
__host__ cuda_vector<T>::cuda_vector(const T* A, Rank lo, Rank hi) {
    copyFrom(A, lo, hi);
}

template <typename T>
__host__ cuda_vector<T>::cuda_vector(const cuda_vector<T>& V) {
    copyFrom(V._elem, 0, V._size);
}

template <typename T>
__host__ cuda_vector<T>::cuda_vector(const cuda_vector<T>& V, Rank lo, Rank hi) {
    copyFrom(V._elem, lo, hi);
}

template <typename T>
__host__ cuda_vector<T>::~cuda_vector() {
    cudaFree(_elem);
}

template <typename T>
__host__ void cuda_vector<T>::copyFrom(const T* A, Rank lo, Rank hi) {
    assert(lo < hi);

    _capacity = 2 * (hi - lo);

    cudaError_t cudaStatus = cudaMallocManaged((void**)&_elem, _capacity * sizeof(T));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMallocManaged failed!");

    cudaMemcpy(_elem, A + lo, (hi - lo) * sizeof(T), cudaMemcpyDeviceToDevice);

    _size = hi - lo;
}

template <typename T>
__host__ __device__ T& cuda_vector<T>::operator[](Rank r) const {
    assert(r < _size);

    return _elem[r];
}

template <typename T>
__host__ cuda_vector<T>& cuda_vector<T>::operator=(const cuda_vector<T>& V) {
    if (_elem)
        cudaFree(_elem);
    copyFrom(V._elem, 0, V._size);
    return *this;
}

template <typename T>
__host__ void cuda_vector<T>::expand() {
    if (_size < _capacity)
        return;

    if (_capacity < DEFAULT_CAPACITY)
        _capacity = DEFAULT_CAPACITY;

    T* oldElem = _elem;
    _capacity <<= 1;
    cudaError_t cudaStatus = cudaMallocManaged((void**)&_elem, _capacity * sizeof(T));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMallocManaged failed!");

    cudaMemcpy(_elem, oldElem, _size * sizeof(T), cudaMemcpyDeviceToDevice);

    cudaFree(oldElem);

    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
}

template <typename T>
__host__ void cuda_vector<T>::expand(Rank n) {
    if (n <= _capacity)
        return;

    T* oldElem = _elem;
    _capacity = n;
    cudaError_t cudaStatus = cudaMallocManaged((void**)&_elem, _capacity * sizeof(T));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMallocManaged failed!");

    cudaMemcpy(_elem, oldElem, _size * sizeof(T), cudaMemcpyDeviceToDevice);

    cudaFree(oldElem);

    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
}

template <typename T>
__host__ void cuda_vector<T>::shrink() {
    if (_capacity < DEFAULT_CAPACITY << 1)
        return;

    if (_size << 2 > _capacity)
        return;

    T* oldElem = _elem;
    _capacity >>= 1;
    cudaError_t cudaStatus = cudaMallocManaged((void**)&_elem, _capacity * sizeof(T));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMallocManaged failed!");

    cudaMemcpy(_elem, oldElem, _size * sizeof(T), cudaMemcpyDeviceToDevice);

    cudaFree(oldElem);

    cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
}

template <typename T>
__host__ Rank cuda_vector<T>::insert(Rank r, T const& e) {
    assert(r <= _size);

    expand();

    for (Rank i = _size; i > r; i--)
        _elem[i] = _elem[i - 1];

    _elem[r] = e;
    _size++;

    return r;
}

template <typename T>
__host__ Rank cuda_vector<T>::remove(Rank lo, Rank hi) {
    if (lo == hi)
        return 0;

    while (hi < _size)
        _elem[lo++] = _elem[hi++];

    _size = lo;
    shrink();

    return hi - lo;
}

template <typename T>
__host__ T cuda_vector<T>::remove(Rank r) {
    T e = _elem[r];
    remove(r, r + 1);
    return e;
}
