<!-- ## Cuda101 : getting started to write **kernels** for GPU

This repository contains my solutions of a variety function to run on the GPU.

#### Files contents :

.
| 1_add # easy add function : to get familiar with **cudeMemCopy()** and **threading**
| 2_debug # Cuda code isn't that easy to debug, this file contains some **Debugging** methods
| 3_SAXPY # this code contains how to implement linear operation the $y= a \times x + y$ where y and x are Huge vectors.
| 4_Matrix_Multiplication # Here we dive alittle in details and we use the **shared_memory** and discover the **tiling** to benefit for more acceleration, and compare our implementation with th **cuBLAS** -->

## CUDA 101: Getting Started with Writing **Kernels** for GPU

This repository contains a variety of CUDA functions designed to run on the GPU, providing solutions to common computational problems. These examples will guide you through basic CUDA concepts such as memory management, threading, debugging, and performance optimization.

### File Contents:

---

### 1. `1_add/` – **Basic Add Function**

This example demonstrates a simple addition of two arrays using CUDA. It covers the basics of:

- **cudaMemcpy()**: Transfer data between host (CPU) and device (GPU).
- **Threading**: Using CUDA threads to parallelize the addition operation across multiple cores.

This is a great starting point for getting comfortable with CUDA memory management and setting up your first kernel.

---

### 2. `2_debug/` – **Debugging CUDA Code**

Debugging CUDA code can be difficult due to its parallel nature. This section provides:

- Techniques for debugging CUDA kernels.
- Usage of **cuda-gdb** and **printf()** within device code.
- Common CUDA errors and how to avoid them, such as memory access violations and improper synchronization.

This example focuses on making the debugging process more manageable when writing CUDA programs.

---

### 3. `3_SAXPY/` – **SAXPY Operation**

This code implements the SAXPY operation:

$$ y = a \times x + y $$

Where `x` and `y` are large vectors and `a` is a scalar. The code demonstrates how to:

- Work with large data sets on the GPU.
- Implement vectorized operations using CUDA kernels.
- Efficiently handle parallel computations for performance gains.

SAXPY is a fundamental operation in many scientific and linear algebra applications.

---

### 4. `4_Matrix_Multiplication/` – **Optimized Matrix Multiplication**

This example dives deeper into CUDA programming by implementing matrix multiplication, focusing on performance optimization through:

- **Shared Memory**: Utilizing faster memory on the GPU for efficient data reuse.
- **Tiling**: Dividing the workload into smaller tiles to improve computation performance.
- **Comparison with cuBLAS**: A comparison between custom implementations and the high-performance cuBLAS library to highlight the power of GPU acceleration.

Matrix multiplication is a key operation in many machine learning and numerical computing tasks, and optimizing it is critical for performance.

---

## How to Run the Code

1. Ensure you have CUDA Toolkit installed.
2. Compile the CUDA code with `nvcc`. For example:
