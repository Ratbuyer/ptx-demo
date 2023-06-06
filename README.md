# ptx-demo

CUDA codes that demonstrate how to use mma and mma.sp ptx instructions

requires nvidia gpu compute capability 7.5 + and compatible CUDA toolkit installed

compile the codes using

`nvcc -arch=sm_{your sm version} -O3 mma.cu -lcudart -lcuda -o demo`

and

`nvcc -arch=sm_{your sm version} -O3 mma.sp.cu -lcudart -lcuda -o demo`

for example with RTX3060 that has compute capability of 8.6, use

`nvcc -arch=sm_86 -O3 mma.sp.cu -lcudart -lcuda -o demo`

run the binary with

`./demo M N K ITERATIONS`

where M, N, K are the matrix dimensions and ITERATIONS is the number of matrix multiplications you want to repeat for profiling

for example

`./demo 1024 1024 1024 50`
