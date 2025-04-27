#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_H

#if (__gfx1010__ || __gfx1011__ || __gfx1012__ || __gfx1030__ || __gfx1031__) && __AMDGCN_WAVEFRONT_SIZE == 64
#error HIP is not supported on GFX10 with wavefront size 64
#endif

// Some standard header files, these are included by hc.hpp and so want to make them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors that only appears
// on NVCC path:
#include <stdint.h>
#include "/public/home/aicao/jiazifan/submit-bak/fhi-aims-dcu-mod/src/gpu/stdio.h"
#include <stdlib.h>
#include <assert.h>

#if __cplusplus > 199711L
#include <thread>
#endif

#include <hip/hip_version.h>
#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
#include <hip/hcc_detail/hip_runtime.h>
#elif defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
#include <hip/nvcc_detail/hip_runtime.h>
#else
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <hip/library_types.h>

#endif
