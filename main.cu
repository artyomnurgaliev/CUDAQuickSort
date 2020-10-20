#include <iostream>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define GET_OFFSET(idx) (idx >> LOG_NUM_BANKS)

// Must be divisible by 32 and a divisor of 1024
const int block_size = 256;

// Akhtyamov Pavel's realization of Scan
// https://github.com/akhtyamovpavel/ParallelComputationExamples/blob/master/CUDA/05-scan/05-scan_bank_conflicts.cu
__global__ void Scan(int *in_data, int *out_data) {
    // in_data ->  [1 2 3 4 5 6 7 8], block_size 4
    // block_idx -> [0 0 0 0 1 1 1 1 ]

    extern __shared__ int shared_data[];
    // block_idx = 0

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid + GET_OFFSET(tid)] = in_data[index];

    // shared_data[tid + (tid >> LOG_NUM_BANKS)] = in_data[index];

    // shared_data -> [1, 2, 3, 4]
    __syncthreads();

    // shift = 2^(d - 1)
    for (unsigned int shift = 1; shift < blockDim.x; shift <<= 1) {
        int ai = shift * (2 * tid + 1) - 1; // tid = 0, shift = 1, ai = 0; // tid = 16, shift = 1, ai = 32 = 0
        int bi = shift * (2 * tid + 2) - 1;

        if (bi < blockDim.x) {
            shared_data[bi + GET_OFFSET(bi)] += shared_data[ai + GET_OFFSET(ai)];
        }

        __syncthreads();
    }

    if (tid == 0) {
        shared_data[blockDim.x - 1 + GET_OFFSET(blockDim.x - 1)] = 0;
    }

    __syncthreads();

    int temp;
    for (unsigned int shift = blockDim.x / 2; shift > 0; shift >>= 1) {
        int bi = shift * (2 * tid + 2) - 1;
        int ai = shift * (2 * tid + 1) - 1;
        int ai_offset = ai + GET_OFFSET(ai);
        int bi_offset = bi + GET_OFFSET(bi);
        if (bi < blockDim.x) {
            temp = shared_data[ai_offset]; // blue in temp

            // temp = 4
            shared_data[ai_offset] = shared_data[bi_offset]; // orange

            // 1 2 1 0 1 2 1 0 // temp = 4
            shared_data[bi_offset] = temp + shared_data[bi_offset];
        }
        __syncthreads();

    }
    // if (blockIdx.x == 16383) {
    //     printf("%d %d %d %d\n", tid, tid + GET_OFFSET(tid), shared_data[tid + GET_OFFSET(tid)], index);
    //     // std::cout << shared_data[tid] << std::endl;
    // }
    // block_idx = 0 -> [a0, a1, a2, a3]
    // block_idx = 1 -> [a4, a5, a6, a7]
    out_data[index] = shared_data[tid + GET_OFFSET(tid)];

    __syncthreads();

    // out_data[block_idx == 0] = [1, 3, 6, 10]

    // out_data[block_idx == 1] = [5, 11, 18, 26]

}


/*
 * Calculates grid size for array
 * size - array size
 */
int GridSize(int size) {
    // Calc grid size, the whole array must be covered with blocks
    int grid_size = size / block_size;
    if (size % block_size) {
        grid_size += 1;
    }

    return grid_size;
}


/*
 * Put prefix sums of each block in one array (d_blocks)
 *
 * d_array - source array
 * d_prefix_sum - calculated prefix sum on blocks
 * d_blocks - result
 */
__global__ void FindBlocks(int *d_array, int *d_prefix_sum, int *d_blocks) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int last_elem_in_block_index = (index + 1) * block_size - 1;

    d_blocks[index] = d_array[last_elem_in_block_index] +
                      d_prefix_sum[last_elem_in_block_index];

    __syncthreads();
}


/*
 * Fills masks of elements greater than pivot and less than or equal to pivot
 * using GPU
 *
 * d_array - array of elements under consideration
 * d_greater_mask - mask of elements greater than pivot
 * d_less_or_equal_mask - mask of elements less (or equal) than pivot
 */
__global__ void FillMasks(int *d_array, int size, int pivot, int *d_greater_mask, int *d_less_or_equal_mask) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size) {
        if (d_array[index] > pivot) {
            d_greater_mask[index] = 1;
            d_less_or_equal_mask[index] = 0;
        } else {
            d_greater_mask[index] = 0;
            d_less_or_equal_mask[index] = 1;
        }
    }

    __syncthreads();
}


/*
 * Add prefix sum of previous blocks to prefix sum, calculated on one block
 *
 * d_prefix_sum - prefix sum calculated on each block
 * d_block_prefix_sum - prefix sum calculated on previous blocks
 */
__global__ void AddBlocksPrefixSum(int *d_prefix_sum, int *d_block_prefix_sum) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    d_prefix_sum[index] += d_block_prefix_sum[blockIdx.x];

    __syncthreads();
}


/*
 * Calculates scan (cumsum) for input array
 * using GPU
 *
 * d_array - input array
 * size - input array size
 * d_prefix_sum - result array
 */
void ParallelPrefixSum(int *d_array, int size, int *d_prefix_sum) {
    // Calc prefix sum on each block
    int grid_size = GridSize(size);
    Scan<<<grid_size, block_size, sizeof(int) * (block_size + GET_OFFSET(block_size))>>>(d_array, d_prefix_sum);

    // If size of array greater than size of one block,
    // we have to add to each block prefix sum of previous blocks
    if (grid_size > 1) {
        int block_grid_size = GridSize(grid_size);

        // Array of last elements of blocks
        int *d_block_array;
        cudaMalloc(&d_block_array, sizeof(int) * grid_size);
        // Array for prefix sum on blocks
        int *d_block_prefix_sum;
        cudaMalloc(&d_block_prefix_sum, sizeof(int) * grid_size);

        // Put prefix sum of each block in d_block_array
        FindBlocks<<<block_grid_size, block_size>>>(d_array, d_prefix_sum, d_block_array);
        // Find prefix sum on blocks
        ParallelPrefixSum(d_block_array, grid_size, d_block_prefix_sum);
        // Add prefix sum of all previous blocks to prefix sum
        AddBlocksPrefixSum<<<grid_size, block_size>>>(d_prefix_sum, d_block_prefix_sum);

        cudaFree(d_block_array);
        cudaFree(d_block_prefix_sum);
    }
}

/*
 * Divide elements into two arrays
 * d_array - source array
 * size - size of array
 *
 * d_less_or_equal_mask, d_greater_mask - masks of elements in d_array
 * d_less_or_equal_prefix_sum, d_greater_prefix_sum - prefix sums of masks
 *
 * d_less_or_equal_elems, d_greater_elems - result
 */
__global__ void ParallelDivide(int *d_array,
                               int *d_less_or_equal_prefix_sum,
                               int *d_greater_prefix_sum,
                               int *d_less_or_equal_mask,
                               int *d_greater_mask,
                               int *d_less_or_equal_elems,
                               int *d_greater_elems,
                               int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size) {
        if (d_less_or_equal_mask[index]) {
            d_less_or_equal_elems[d_less_or_equal_prefix_sum[index]] = d_array[index];
        } else {
            d_greater_elems[d_greater_prefix_sum[index]] = d_array[index];
        }
    }

    __syncthreads();
}

/*
 * Copy elements from d_source to d_array
 */
__global__ void ParallelCopy(int *d_array, int *d_source, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size) {
        d_array[index] = d_source[index];
    }

    __syncthreads();
}

/*
 * Returns last value of array in global memory
 * d_array - array in global memory
 * size - d_array size
 */
int GetLast(int *d_array, int size) {
    int result = 0;
    cudaMemcpy(&result, d_array + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}

/*
 * Place elements in correct order
 * Firstly, elements less (or equal) than pivot
 * Then, pivot,
 * Then, elements greater than pivot
 */
void ArrangeElements(int *d_array,
                     int size,
                     int pivot,
                     int *d_less_or_equal_elems,
                     int *d_greater_elems,
                     int less_or_equal_size,
                     int greater_size) {
    int grid_size = GridSize(size);
    ParallelCopy<<<grid_size, block_size>>>(d_array, d_less_or_equal_elems, less_or_equal_size);
    cudaMemcpy(d_array + less_or_equal_size, &pivot, sizeof(int), cudaMemcpyHostToDevice);
    ParallelCopy<<<grid_size, block_size>>>(d_array + less_or_equal_size + 1, d_greater_elems, greater_size);
}

/*
 * Fills arrays of elements less than pivot and greater than pivot
 */
void ProcessMasks(int *d_array,
                  int size,
                  int pivot,
                  int **d_less_or_equal_elems,
                  int **d_greater_elems,
                  int &less_or_equal_size,
                  int &greater_size) {
    // Storages of masks for elements less (or equal) than pivot
    // and greater than pivot
    int *d_less_or_equal_mask = nullptr;
    int *d_greater_mask = nullptr;
    // Storages of its prefix sums
    int *d_less_or_equal_prefix_sum = nullptr;
    int *d_greater_prefix_sum = nullptr;

    cudaMalloc(&d_less_or_equal_mask, sizeof(int) * size);
    cudaMalloc(&d_greater_mask, sizeof(int) * size);

    cudaMalloc(&d_less_or_equal_prefix_sum, sizeof(int) * size);
    cudaMalloc(&d_greater_prefix_sum, sizeof(int) * size);

    int grid_size = GridSize(size);
    FillMasks<<<grid_size, block_size>>>(d_array, size, pivot,
                                         d_greater_mask,
                                         d_less_or_equal_mask);


    ParallelPrefixSum(d_less_or_equal_mask, size, d_less_or_equal_prefix_sum);
    ParallelPrefixSum(d_greater_mask, size, d_greater_prefix_sum);

    // Get count of elements less or equal than pivot, held in last elem of prefix sum
    less_or_equal_size = GetLast(d_less_or_equal_prefix_sum, size);
    // Get count of elements greater than pivot, held in last elem of prefix sum
    greater_size = GetLast(d_greater_prefix_sum, size);

    cudaMalloc(d_less_or_equal_elems, sizeof(int) * less_or_equal_size);
    cudaMalloc(d_greater_elems, sizeof(pivot) * greater_size);

    // Divide elements into two arrays
    ParallelDivide<<<grid_size, block_size>>>(d_array,
                                              d_less_or_equal_prefix_sum,
                                              d_greater_prefix_sum,
                                              d_less_or_equal_mask,
                                              d_greater_mask,
                                              *d_less_or_equal_elems,
                                              *d_greater_elems,
                                              size - 1);

    cudaFree(d_less_or_equal_mask);
    cudaFree(d_greater_mask);
    cudaFree(d_less_or_equal_prefix_sum);
    cudaFree(d_greater_prefix_sum);
}

/*
 * Sorts d_array using GPU
 * d_array - array in global memory
 * size - d_array size
 */
void QuickSort(int *d_array, int size) {
    // Already sorted
    if (size <= 1) return;

    /// Partition
    // Take last element of array as a pivot
    int pivot = GetLast(d_array, size);

    // Storages of elements less (or equal) than pivot
    // and greater than pivot
    int *d_less_or_equal_elems = nullptr;
    int *d_greater_elems = nullptr;
    // Its sizes
    int less_or_equal_size = 0;
    int greater_size = 0;

    ProcessMasks(d_array,
                 size,
                 pivot,
                 &d_less_or_equal_elems,
                 &d_greater_elems,
                 less_or_equal_size,
                 greater_size);

    ArrangeElements(d_array,
                    size,
                    pivot,
                    d_less_or_equal_elems,
                    d_greater_elems,
                    less_or_equal_size,
                    greater_size);

    cudaFree(d_less_or_equal_elems);
    cudaFree(d_greater_elems);

    // Recursively process parts
    QuickSort(d_array, less_or_equal_size);
    QuickSort(d_array + less_or_equal_size + 1, greater_size);
}

void Shuffle(int *array, int size) {
    for (int i = size - 1; i > 0; --i) {
        std::swap(array[i], array[std::rand() % (i + 1)]);
    }
}

int main() {
    size_t size = 0;
    std::cout << "Insert the size of array" << std::endl;
    std::cin >> size;

    int *h_array_to_be_sorted = new int[size];
    for (int i = 0; i < size; ++i) {
        h_array_to_be_sorted[i] = i;
    }
    // Create shuffled example to be sorted later
    Shuffle(h_array_to_be_sorted, size);

    // Create array to be processed to GPU
    int *d_array_to_be_sorted;
    cudaMalloc(&d_array_to_be_sorted, size * sizeof(float));
    cudaMemcpy(d_array_to_be_sorted,
               h_array_to_be_sorted,
               sizeof(int) * size,
               cudaMemcpyHostToDevice);


    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0 /* Stream ID */);
    QuickSort(d_array_to_be_sorted, size);
    cudaEventRecord(stop, 0);

    cudaMemcpy(h_array_to_be_sorted,
               d_array_to_be_sorted,
               sizeof(int) * size,
               cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time is :" << elapsedTime << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Sorted array:" << std::endl;
    for (int i = 0; i < size; ++i) {
        std::cout << h_array_to_be_sorted[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_array_to_be_sorted);
    delete[] h_array_to_be_sorted;
    return 0;
}