#include <stdint.h>

__global__ void kernel()
{
  uint64_t desc_a = 0;
  uint64_t desc_b = 0;
  int32_t metaE = 0;
  int32_t scaleD = 0;
  const int32_t scaleA = 0;
  const int32_t scaleB = 0;
  const int32_t tnspA = 0;
  const int32_t tnspB = 0;

  int32_t reg = 0;

  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %68, 0;\n"
      "wgmma.mma_async.sp.aligned.m64n256k32.f16.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63},"
      "%64,"
      "%65,"
      "%66,"
      "0x0,"
      "p, %69, %70, %71, %72;\n"
      "}\n"
      : "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg),
        "+r"(reg), "+r"(reg), "+r"(reg), "+r"(reg)
      : "l"(desc_a),
        "l"(desc_b),
        "r"(metaE),
        "r"(0x0),
        "r"(int32_t(scaleD)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
}

int main()
{
}