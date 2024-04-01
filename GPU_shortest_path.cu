#include <stdio.h>      //I/O
#include <stdlib.h>
#include<float.h>
#include <time.h>       //for code timing purposes
#include <math.h>
#include <chrono>
#include <driver_types.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h>
#include <graph_v_of_v_idealID/common_algorithms/graph_v_of_v_idealID_shortest_paths.h>
#include "device_launch_parameters.h"
#define weightType float
#define VERTICES  16384
#define INT_INF  9999999
#define THREADS_BLOCK 512
void makeGraph(graph_v_of_v_idealID& graph, int size,unsigned int seed);

int N, totalEdgeSize;
int* source = (int*)malloc(sizeof(int));
int* graphVertice = (int*)malloc(VERTICES*VERTICES * sizeof(int));
weightType* edgeWeight=(weightType*)malloc(VERTICES*VERTICES*sizeof(weightType));
int* edgeNum= (int*)malloc(VERTICES * sizeof(int));
int* startVertice=(int*)malloc(VERTICES * sizeof(int));

void setStartVertice(graph_v_of_v_idealID& graph, int*& startVertice);
void setGraphVertice(graph_v_of_v_idealID& graph, int*& graphVertice, const int* startVertice);
void setEdgeWeight(graph_v_of_v_idealID& graph, weightType*& edgeWeight, const int* startVertice);
void setEdgeNum(graph_v_of_v_idealID& graph, int*& edgeNum);
void printVariables();
void setCPUnodeDist(weightType*& nodeDist, weightType a);
void setCPUparentNode(int*& parentNode);
void setCPUvisitedNode(int*& visitedNode, int a);
__global__ void closestNodeCUDA(weightType* nodeDist, int* visitedNode, int* globalClosest,int verticeNum);
__global__ void relaxCUDA(int* graphVertice, weightType* nodeDist, int* parentNode, int* visitedNode, int* globalClosest, int* startVertice, int* edgeNum, weightType* edgeWeight);
void printANS(int N,int*parentNode,weightType*dist);
bool checkANS(int N,int* CUDAparent, weightType* CUDAdist, int*CPUparent,weightType*CPUdist);

/* GPU variables*/
weightType* gpu_nodeDist;
int* gpu_graphVertice;
weightType* gpu_edgeWeight;
int* gpu_parentNode;
int* gpu_visitedNode;
int* gpu_edgeNum;
int* gpu_startVertice;
int* closest_vertex = (int*)malloc(sizeof(int));
int* gpu_closest_vertex;

weightType* nodeDist = (weightType*)malloc(VERTICES*sizeof(weightType));
int* parentNode = (int*)malloc(VERTICES * sizeof(int));
int* visitedNode = (int*)malloc(VERTICES * sizeof(int));



cudaError_t check3= cudaMalloc((void**)&gpu_graphVertice, VERTICES*VERTICES * sizeof(int));
cudaError_t status = cudaMalloc((void**)&gpu_nodeDist, VERTICES * sizeof(weightType));
cudaError_t check8 = cudaMalloc((void**)&gpu_parentNode, VERTICES * sizeof(int));
cudaError_t check9 = cudaMalloc((void**)&gpu_visitedNode, VERTICES * sizeof(int));
cudaError_t check10 = cudaMalloc((void**)&gpu_edgeWeight, VERTICES*VERTICES * sizeof(weightType));
cudaError_t check12 = cudaMalloc((void**)&gpu_edgeNum, VERTICES * sizeof(int));
cudaError_t check13 = cudaMalloc((void**)&gpu_startVertice, VERTICES * sizeof(int));

int main() {
	printf("Please input the size of graph : ");
	scanf("%d", &N);//size of graph
	graph_v_of_v_idealID graph(N);
	makeGraph(graph, N,10);
	totalEdgeSize = graph_v_of_v_idealID_total_edge_num(graph) * 2;
	//graph_v_of_v_idealID_print(graph);
	setStartVertice(graph, startVertice);
	setGraphVertice(graph, graphVertice, startVertice);
	setEdgeWeight(graph, edgeWeight, startVertice);
	setEdgeNum(graph, edgeNum);
	//printVariables();
	setCPUnodeDist(nodeDist, FLT_MAX);
	setCPUparentNode(parentNode);
	setCPUvisitedNode(visitedNode, 0);
	printf("Input the source vertices :");
	int t;
	scanf("%d", &t);
	*source = t;
	printf("Running CUDA dijkstra ......\n");
	nodeDist[*source] = 0;
	closest_vertex[0] = -1;
	//CUDA dijkstra


	if (check3 != cudaSuccess) {
		fprintf(stderr, "31cudaMalloc failed: %s\n", cudaGetErrorString(check3));
	}
	if (status != cudaSuccess) {
		fprintf(stderr, "1cudaMalloc failed: %s\n", cudaGetErrorString(status));
	}
	cudaError_t check1 = cudaMalloc((void**)&gpu_closest_vertex, (sizeof(int)));
	if (check1 != cudaSuccess) {
		fprintf(stderr, "1cudaMalloc failed: %s\n", cudaGetErrorString(check1));
	}
	cudaError_t check2 = cudaMemcpy(gpu_closest_vertex, closest_vertex, sizeof(int), cudaMemcpyHostToDevice);
	if (check2 != cudaSuccess) {
		fprintf(stderr, "2cudaMalloc failed: %s\n", cudaGetErrorString(check2));
	}
	cudaError_t check4 = cudaMemcpy(gpu_graphVertice, graphVertice, VERTICES * VERTICES * sizeof(int), cudaMemcpyHostToDevice);
	if (check4 != cudaSuccess) {
		fprintf(stderr, "4cudaMalloc failed: %s\n", cudaGetErrorString(check4));
	}
	cudaError_t check5 = cudaMemcpy(gpu_nodeDist, nodeDist, VERTICES * sizeof(weightType), cudaMemcpyHostToDevice);
	if (check5 != cudaSuccess) {
		fprintf(stderr, "5cudaMalloc failed: %s\n", cudaGetErrorString(check5));
	}
	cudaError_t check6 = cudaMemcpy(gpu_parentNode, parentNode, VERTICES * sizeof(int), cudaMemcpyHostToDevice);
	if (check6 != cudaSuccess) {
		fprintf(stderr, "6cudaMalloc failed: %s\n", cudaGetErrorString(check6));
	}
	cudaError_t check7 = cudaMemcpy(gpu_visitedNode, visitedNode, VERTICES * sizeof(int), cudaMemcpyHostToDevice);
	if (check7 != cudaSuccess) {
		fprintf(stderr, "7cudaMalloc failed: %s\n", cudaGetErrorString(check7));
	}
	cudaError_t check11 = cudaMemcpy(gpu_edgeWeight, edgeWeight, VERTICES * VERTICES * sizeof(weightType), cudaMemcpyHostToDevice);
	cudaError_t check14 = cudaMemcpy(gpu_startVertice, startVertice, VERTICES * sizeof(weightType), cudaMemcpyHostToDevice);
	cudaError_t check15 = cudaMemcpy(gpu_edgeNum, edgeNum, VERTICES * sizeof(int), cudaMemcpyHostToDevice);
	dim3 gridMin(1, 1, 1);
	dim3 blockMin(1, 1, 1);
	dim3 gridRelax(VERTICES / THREADS_BLOCK, 1, 1);
	dim3 blockRelax(THREADS_BLOCK, 1, 1);

	cudaEvent_t GPUstart, GPUstop;
	cudaEventCreate(&GPUstart);
	cudaEventCreate(&GPUstop);
	cudaEventRecord(GPUstart, 0);
	for (int i = 0; i < VERTICES; i++) {
		closestNodeCUDA << <gridMin, blockMin >> > (gpu_nodeDist, gpu_visitedNode, gpu_closest_vertex, N);
		cudaError_t syncError1 = cudaDeviceSynchronize();
		if (syncError1 != cudaSuccess) {
			fprintf(stderr, "closestNodeCUDA Sync Error: %s\n", cudaGetErrorString(syncError1));
		}
		relaxCUDA << <gridRelax, blockRelax >> > (gpu_graphVertice, gpu_nodeDist, gpu_parentNode, gpu_visitedNode, gpu_closest_vertex, gpu_startVertice, gpu_edgeNum, gpu_edgeWeight);
		cudaError_t syncError2 = cudaDeviceSynchronize();
		if (syncError2 != cudaSuccess) {
			fprintf(stderr, "relaxCUDA Sync Error: %s\n", cudaGetErrorString(syncError2));
		}
	}
	cudaEventRecord(GPUstop, 0);
	cudaEventSynchronize(GPUstop);

	float CUDAtime = 0;
	cudaEventElapsedTime(&CUDAtime, GPUstart, GPUstop);

	cudaEventDestroy(GPUstart);
	cudaEventDestroy(GPUstop);
	cudaError_t st1 = cudaMemcpy(nodeDist, gpu_nodeDist, VERTICES * sizeof(weightType), cudaMemcpyDeviceToHost);
	if (st1 != cudaSuccess) {
		fprintf(stderr, "st1 cudaMalloc failed: %s\n", cudaGetErrorString(st1));
	}
	cudaError_t st2 = cudaMemcpy(parentNode, gpu_parentNode, VERTICES * sizeof(int), cudaMemcpyDeviceToHost);
	cudaError_t st3 = cudaMemcpy(visitedNode, gpu_visitedNode, VERTICES * sizeof(int), cudaMemcpyDeviceToHost);
	
	std::vector<float> distance;
	std::vector<int> parent;
	auto CPUstart = std::chrono::high_resolution_clock::now();
	graph_v_of_v_idealID_shortest_paths(graph, *source, distance, parent);
	auto CPUstop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(CPUstop - CPUstart);
	auto CPUtime = duration.count();
	cout << " ------ " << CPUtime << endl;
	cout << " ======" << CUDAtime << endl;
	weightType* CPUdist = (weightType*)malloc(N * sizeof(weightType));
	int* CPUparent = (int*)malloc(N * sizeof(int));
	for (int i = 0; i < N; i++) {
		CPUdist[i] = distance[i];
		CPUparent[i] = parent[i];
	}
	
	if (checkANS(N, parentNode, nodeDist, CPUparent, CPUdist)) {
		printf("CUDAtime : %f    CPUtime : %f\n", CUDAtime, CPUtime);
		if (CUDAtime < CPUtime) {
			printf("BETTER");
		}
		else {
			printf("NEED UPDATING");
		}
	}

}


void makeGraph(graph_v_of_v_idealID& graph, int size,unsigned int seed) {
	srand(seed); // 设置随机种子

	for (int i = 0; i < size; i++) {
		for (int j = i + 1; j < size; j++) {
			int weight = rand() % VERTICES*10 + 1; 
			graph_v_of_v_idealID_add_edge(graph, i, j, weight);
		}
	}
}

void setStartVertice(graph_v_of_v_idealID& graph, int*& startVertice) {
	startVertice[0] = 0;
	for (int i = 1; i < graph.size(); i++) {
		startVertice[i] = graph[i-1].size() + startVertice[i - 1];
	}
}

void setEdgeNum(graph_v_of_v_idealID& graph, int*& edgeNum) {
	for (int i = 0; i < graph.size(); i++) {
		edgeNum[i] = graph[i].size();
	}
}

void setGraphVertice(graph_v_of_v_idealID& graph, int*& graphVertice, const int* startVertice) {
	for (int i = 0; i < graph.size(); i++) {
		int cnt = 0;
		for (auto it : graph[i]) {
			graphVertice[startVertice[i] + cnt] = it.first;
			cnt++;
		}
	}
}

void setEdgeWeight(graph_v_of_v_idealID& graph, weightType*& edgeWeight, const int* startVertice) {
	for (int i = 0; i < graph.size(); i++) {
		int cnt = 0;
		for (auto it : graph[i]) {
			edgeWeight[startVertice[i] + cnt] = it.second;
			cnt++;
		}
	}
}

void printVariables() {
	printf("Printing variables : \n");
	printf("startVertice : ");
	for (int i = 0; i < N; i++) {
		printf("%d ", startVertice[i]);
	}
	printf("\n");
	printf("graphVertice : ");
	for (int i = 0; i < totalEdgeSize; i++) {
		printf("%d ", graphVertice[i]);
	}
	printf("\n");
	printf("edgeWeight");
	for (int i = 0; i < totalEdgeSize; i++) {
		printf("%f ", edgeWeight[i]);
	}
	printf("\n");
	printf("edgeNum : ");
	for (int i = 0; i < N; i++) {
		printf("%d ", edgeNum[i]);
	}
	printf("\nFinished\n");
}


void setCPUnodeDist(weightType*& nodeDist, weightType a) {
	for (int i = 0; i < N; i++) {
		nodeDist[i] = a;
	}
}
void setCPUparentNode(int*& parentNode) {
	for (int i = 0; i < N; i++) {
		parentNode[i] = i;
	}
}
void setCPUvisitedNode(int*& visitedNode, int a) {
	for (int i = 0; i < N; i++) {
		visitedNode[i] = a;
	}
}

__global__ void closestNodeCUDA(weightType* nodeDist, int* visitedNode, int* globalClosest,int verticeNum) {
	weightType dist = FLT_MAX;
	int node = -1;
	for (int i = 0; i < verticeNum; i++) {
		if ((nodeDist[i] < dist) && (visitedNode[i] != 1)) {
			dist = nodeDist[i];
			node = i;
		}
	}
	globalClosest[0] = node; 
	visitedNode[node] = 1;
}

__global__ void relaxCUDA(int* graphVertice, weightType* nodeDist, int* parentNode, int* visitedNode, int* globalClosest,int* startVertice,int*edgeNum,weightType*edgeWeight) {
	int next = blockIdx.x * blockDim.x + threadIdx.x;
	int source = globalClosest[0];

	if (next < edgeNum[source]) {

		weightType edge = edgeWeight[startVertice[source] + next];
		weightType newDist = nodeDist[source] + edge;
		int target = graphVertice[startVertice[source] + next];

		if ((visitedNode[target] != 1) && (newDist < nodeDist[target])) {
			nodeDist[target]=newDist;
			parentNode[target]=source;
		}
	}
}

void printANS(int N, int* parentNode, weightType* dist) {
	printf("PRINT Distance : ");
	for (int i = 0; i < N; i++) {
		printf("%f ", dist[i]);
	}
	printf("\n");
	printf("PRINT Parent   : ");
	for (int i = 0; i < N; i++) {
		printf("%d ", parentNode[i]);
	}
	printf("\n");
}

bool checkANS(int N, int* CUDAparent, weightType* CUDAdist,int* CPUparent,weightType*CPUdist ) {
	bool flag = true;
	for (int i = 0; i < N; i++) {
		if (CUDAdist[i] != CPUdist[i] || CUDAparent[i] != CPUparent[i]) {
			flag = false;
			printf("CUDAdist : %f  CUDAparent : %d   CPUdist : %f   CPUparent : %d\n", CUDAdist[i], CUDAparent[i], CPUdist[i], CPUparent[i]);
		}
	}
	if (flag) {
		printf("SAME\n");
	}
	else {
		printf("NOT SAME\n");
	}
	return flag;
}
