## Cuda101 : getting started to write **kernels** for GPU

This repository contains my solutions of a variety function to run on the GPU.

#### Files contents :

.
| 1_add # easy add function : to get familiar with **cudeMemCopy()** and **threading**
| 2_debug # Cuda code isn't that easy to debug, this file contains some **Debugging** methods
| 3_SAXPY # this code contains how to implement linear operation the $y= a \times x + y$ where y and x are Huge vectors.
| 4_Matrix_Multiplication # Here we dive alittle in details and we use the **shared_memory** and discover the **tiling** to benefit for more acceleration, and compare our implementation with th **cuBLAS**
