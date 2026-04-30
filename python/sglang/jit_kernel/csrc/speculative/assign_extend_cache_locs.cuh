#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename PoolIndexT, typename TokenT, typename OffsetT>
__global__ void assign_extend_cache_locs_kernel(
    const PoolIndexT* __restrict__ req_pool_indices,
    const TokenT* __restrict__ req_to_token,
    const OffsetT* __restrict__ start_offset,
    const OffsetT* __restrict__ end_offset,
    int64_t* __restrict__ out_cache_loc,
    size_t pool_len,
    size_t batch_size) {
  const size_t row = blockIdx.x;
  if (row >= batch_size) {
    return;
  }

  int64_t out_offset = 0;
  for (size_t i = 0; i < row; ++i) {
    out_offset += static_cast<int64_t>(end_offset[i]) - static_cast<int64_t>(start_offset[i]);
  }

  const int64_t kv_start = static_cast<int64_t>(start_offset[row]);
  const int64_t kv_end = static_cast<int64_t>(end_offset[row]);
  const int64_t req_idx = static_cast<int64_t>(req_pool_indices[row]);
  const TokenT* token_pool = req_to_token + req_idx * pool_len;

  for (int64_t pos = kv_start + threadIdx.x; pos < kv_end; pos += blockDim.x) {
    out_cache_loc[out_offset + pos - kv_start] = static_cast<int64_t>(token_pool[pos]);
  }
}

constexpr size_t kBlockSize = 128;

template <typename PoolIndexT, typename TokenT, typename OffsetT>
struct AssignExtendCacheLocs {
  static void run(
      tvm::ffi::TensorView req_pool_indices,
      tvm::ffi::TensorView req_to_token,
      tvm::ffi::TensorView start_offset,
      tvm::ffi::TensorView end_offset,
      tvm::ffi::TensorView out_cache_loc) {
    using namespace host;

    SymbolicSize batch_size = {"batch_size"};
    SymbolicSize req_pool_size = {"req_pool_size"};
    SymbolicSize pool_len = {"pool_len"};
    SymbolicSize out_len = {"out_len"};
    SymbolicDevice device;
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({batch_size}).with_dtype<PoolIndexT>().with_device(device).verify(req_pool_indices);
    TensorMatcher({batch_size}).with_dtype<OffsetT>().with_device(device).verify(start_offset).verify(end_offset);
    TensorMatcher({req_pool_size, pool_len}).with_dtype<TokenT>().with_device(device).verify(req_to_token);
    TensorMatcher({out_len}).with_dtype<int64_t>().with_device(device).verify(out_cache_loc);

    const size_t batch = batch_size.unwrap();
    if (batch == 0) {
      return;
    }

    const DLDevice dl_device = device.unwrap();
    LaunchKernel(batch, kBlockSize, dl_device)(
        assign_extend_cache_locs_kernel<PoolIndexT, TokenT, OffsetT>,
        static_cast<const PoolIndexT*>(req_pool_indices.data_ptr()),
        static_cast<const TokenT*>(req_to_token.data_ptr()),
        static_cast<const OffsetT*>(start_offset.data_ptr()),
        static_cast<const OffsetT*>(end_offset.data_ptr()),
        static_cast<int64_t*>(out_cache_loc.data_ptr()),
        pool_len.unwrap(),
        batch);
  }
};

}  // namespace
