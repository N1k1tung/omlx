#include "packed_sparse_attention.h"
#include "omlx/binary_dir.h"

#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "attn_params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::attention {

namespace {

using namespace mlx::core;
using namespace mlx::steel;

inline int64_t bcast_stride(const array& a, int axis) {
  return a.shape(axis) == 1 ? 0 : a.strides(axis);
}

bool last_dim_contiguous(const array& arr) {
  return arr.strides(-1) == 1;
}

class PackedKVUnpackPrimitive : public Primitive {
 public:
  explicit PackedKVUnpackPrimitive(Stream stream) : Primitive(stream) {}

  static bool unsupported(
      const array& local_kv, const array& pooled, Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (local_kv.dtype() != uint8 || pooled.dtype() != uint8) {
      return true;
    }
    if (local_kv.ndim() != 4 || pooled.ndim() != 3) {
      return true;
    }
    if (!last_dim_contiguous(local_kv) || !last_dim_contiguous(pooled)) {
      return true;
    }
    if (local_kv.shape(0) != pooled.shape(0) || local_kv.shape(1) != 1 ||
        local_kv.shape(3) != 528 || pooled.shape(2) != 288) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "PackedKVUnpackPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    const auto& local_kv = inputs[0];
    const auto& pooled = inputs[1];
    auto& local_dense = outputs[0];
    auto& pooled_dense = outputs[1];
    local_dense.set_data(allocator::malloc(local_dense.nbytes()));
    pooled_dense.set_data(allocator::malloc(pooled_dense.nbytes()));

    constexpr int dim = 512;
    const int B = local_kv.shape(0);
    const int localL = local_kv.shape(2);
    const int pooledL = pooled.shape(1);

    DeepseekV4SparseAttentionParams params{
        /* int B = */ B,
        /* int H = */ 0,
        /* int qL = */ 0,
        /* int localL = */ localL,
        /* int pooledL = */ pooledL,
        /* int topk = */ 0,
        /* int local_window = */ 0,
        /* int compress_ratio = */ 1,
        /* int q_offset = */ 0,
        /* float scale = */ 0.0f,
        /* int64_t Q_strides[3] = */ {0, 0, 0},
        /* int64_t Local_strides[3] = */ {
            local_kv.strides(0), bcast_stride(local_kv, 1), local_kv.strides(2)},
        /* int64_t Pooled_strides[2] = */ {
            pooled.strides(0), pooled.strides(1)},
        /* int64_t Topk_strides[3] = */ {0, 0, 0},
        /* int64_t O_strides[3] = */ {0, 0, 0}};

    auto lib = d.get_library("omlx_attention_kernels", binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto unpack_kernel = d.get_kernel("deepseek_v41_prefill_unpack_kv", lib);
    compute_encoder.set_compute_pipeline_state(unpack_kernel);
    compute_encoder.set_input_array(local_kv, 0);
    compute_encoder.set_input_array(pooled, 1);
    compute_encoder.set_output_array(local_dense, 2);
    compute_encoder.set_output_array(pooled_dense, 3);
    compute_encoder.set_bytes(params, 4);
    compute_encoder.dispatch_threads(
        MTL::Size(size_t(B) * (localL + pooledL) * dim, 1, 1),
        MTL::Size(256, 1, 1));
  }

  DEFINE_NAME(OMLXPackedKVUnpack)
  std::vector<Shape> output_shapes(
      const std::vector<array>& inputs) override {
    return {
        Shape{inputs[0].shape(0), 1,
              static_cast<int>(std::max<int64_t>(inputs[0].shape(2), int64_t(1))),
              512},
        Shape{inputs[1].shape(0),
              static_cast<int>(std::max<int64_t>(inputs[1].shape(1), int64_t(1))),
              512}};
  }
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }
};

class PackedSparseAttentionPrimitive : public Primitive {
 public:
  PackedSparseAttentionPrimitive(
      Stream stream,
      float scale,
      int q_offset,
      int compress_ratio,
      int local_window)
      : Primitive(stream),
        scale_(scale),
        q_offset_(q_offset),
        compress_ratio_(compress_ratio),
        local_window_(local_window) {}

  static bool unsupported(
      const array& q,
      const array& local_kv,
      const array& pooled,
      const array& topk_indices,
      const array& sinks,
      int q_offset,
      int compress_ratio,
      int local_window,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    const bool packed_kv =
        local_kv.dtype() == uint8 && pooled.dtype() == uint8;
    const bool dense_kv =
        local_kv.dtype() == bfloat16 && pooled.dtype() == bfloat16;
    if ((!packed_kv && !dense_kv) || sinks.dtype() != float32) {
      return true;
    }
    if (q.dtype() != bfloat16) {
      return true;
    }
    if (q.ndim() != 4 || local_kv.ndim() != 4 || pooled.ndim() != 3 ||
        topk_indices.ndim() != 4 || sinks.ndim() != 1) {
      return true;
    }
    if (!last_dim_contiguous(q) || !last_dim_contiguous(local_kv) ||
        !last_dim_contiguous(pooled) || !last_dim_contiguous(topk_indices) ||
        !last_dim_contiguous(sinks)) {
      return true;
    }
    if (q.shape(0) != local_kv.shape(0) || q.shape(0) != pooled.shape(0) ||
        q.shape(0) != topk_indices.shape(0) || q.shape(1) != 64 ||
        q.shape(3) != 512 || local_kv.shape(1) != 1 ||
        local_kv.shape(3) != (packed_kv ? 528 : 512) ||
        pooled.shape(2) != (packed_kv ? 288 : 512) ||
        topk_indices.shape(1) != 1 || topk_indices.shape(2) != q.shape(2) ||
        sinks.shape(0) != q.shape(1)) {
      return true;
    }
    if (q.shape(2) <= 1 || local_kv.shape(2) < q.shape(2) ||
        topk_indices.dtype() != uint32) {
      return true;
    }
    if (q_offset < 0 || compress_ratio <= 0 || local_window <= 0) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "PackedSparseAttentionPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    const auto& q = inputs[0];
    const auto& local_kv = inputs[1];
    const auto& pooled = inputs[2];
    const auto& topk = inputs[3];
    const auto& sinks = inputs[4];
    auto& o = outputs[0];

    constexpr int bk = 64;
    constexpr int dc = 32;
    constexpr int h = 64;
    constexpr int dim = 512;
    constexpr int wm = 8;

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int qL = q.shape(2);
    const int localL = local_kv.shape(2);
    const int pooledL = pooled.shape(1);
    const bool dense_inputs = local_kv.dtype() == bfloat16;
    const int topkN = topk.shape(3);

    int64_t str_oD = 1;
    int64_t str_oL = o.shape(3);
    int64_t str_oH = o.shape(2) * str_oL;
    int64_t str_oB = o.shape(1) * str_oH;
    size_t data_size = o.shape(0) * str_oB;
    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 1,
        /* bool col_contiguous = */ 0,
    };
    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    std::string base_name;
    concatenate(
        base_name,
        "deepseek_v41_packed_attention_",
        type_to_name(q),
        "_bk",
        bk,
        "_dc",
        dc,
        "_h",
        h,
        "_d",
        dim,
        "_wm",
        wm);

    // Prefill reads Q fragments directly into registers; short verification
    // blocks keep the staged kernel.
    if (qL > 8) {
      base_name += "_directq";
    }

    // Packed inputs amortize decoding KV over the query chunk when the dense
    // scratch fits the budget. Dense inputs are already-unpacked cache rows
    // reused across adjacent layers; no unpack pass is needed for them.
    const size_t unpack_bytes =
        size_t(B) * (localL + std::max(pooledL, 1)) * dim * sizeof(uint16_t);
    const size_t unpack_budget = std::min(
        size_t(512 * 1024 * 1024),
        std::max(size_t(80 * 1024 * 1024), size_t(qL) * 80 * 1024));
    const bool unpack_kv =
        !dense_inputs && qL >= 256 && unpack_bytes <= unpack_budget;
    if (unpack_kv || dense_inputs) {
      base_name += "_unpacked";
    }

    auto lib = d.get_library("omlx_attention_kernels", binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto kernel = d.get_kernel(base_name, lib);

    DeepseekV4SparseAttentionParams params{
        /* int B = */ B,
        /* int H = */ H,
        /* int qL = */ qL,
        /* int localL = */ localL,
        /* int pooledL = */ pooledL,
        /* int topk = */ topkN,
        /* int local_window = */ local_window_,
        /* int compress_ratio = */ compress_ratio_,
        /* int q_offset = */ q_offset_,

        /* float scale = */ scale_,

        /* int64_t Q_strides[3] = */ {
            q.strides(0), q.strides(1), q.strides(2)},
        /* int64_t Local_strides[3] = */ {
            local_kv.strides(0),
            bcast_stride(local_kv, 1),
            local_kv.strides(2)},
        /* int64_t Pooled_strides[2] = */ {
            pooled.strides(0), pooled.strides(1)},
        /* int64_t Topk_strides[3] = */ {
            topk.strides(0), bcast_stride(topk, 1), topk.strides(2)},
        /* int64_t O_strides[3] = */ {
            o.strides(0), o.strides(1), o.strides(2)}};

    auto local_input = local_kv;
    auto pooled_input = pooled;
    if (unpack_kv) {
      local_input = array(Shape{B, localL, dim}, bfloat16, nullptr, {});
      pooled_input =
          array(Shape{B, std::max(pooledL, 1), dim}, bfloat16, nullptr, {});
      local_input.set_data(allocator::malloc(local_input.nbytes()));
      pooled_input.set_data(allocator::malloc(pooled_input.nbytes()));
      compute_encoder.add_temporary(local_input);
      compute_encoder.add_temporary(pooled_input);
      auto unpack_kernel = d.get_kernel("deepseek_v41_prefill_unpack_kv", lib);
      compute_encoder.set_compute_pipeline_state(unpack_kernel);
      compute_encoder.set_input_array(local_kv, 0);
      compute_encoder.set_input_array(pooled, 1);
      compute_encoder.set_output_array(local_input, 2);
      compute_encoder.set_output_array(pooled_input, 3);
      compute_encoder.set_bytes(params, 4);
      compute_encoder.dispatch_threads(
          MTL::Size(size_t(B) * (localL + pooledL) * dim, 1, 1),
          MTL::Size(256, 1, 1));
    }
    if (unpack_kv || dense_inputs) {
      // Attention receives byte pointers and strides, with the original shape
      // and masking metadata; only the row representation has changed.
      params.Local_strides[0] = int64_t(localL) * dim * sizeof(uint16_t);
      params.Local_strides[2] = dim * sizeof(uint16_t);
      params.Pooled_strides[0] = int64_t(pooledL) * dim * sizeof(uint16_t);
      params.Pooled_strides[1] = dim * sizeof(uint16_t);
    }
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(local_input, 1);
    compute_encoder.set_input_array(pooled_input, 2);
    compute_encoder.set_input_array(topk, 3);
    compute_encoder.set_input_array(sinks, 4);
    compute_encoder.set_output_array(o, 5);
    compute_encoder.set_bytes(params, 6);

    MTL::Size grid_dims = MTL::Size(qL, B, 1);
    MTL::Size group_dims = MTL::Size(32, wm, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXPackedSparseAttention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const PackedSparseAttentionPrimitive&>(other);
    return scale_ == rhs.scale_ && q_offset_ == rhs.q_offset_ &&
        compress_ratio_ == rhs.compress_ratio_ &&
        local_window_ == rhs.local_window_;
  }
  auto state() const {
    return std::make_tuple(
        nullptr, scale_, q_offset_, compress_ratio_, local_window_);
  }

 private:
  float scale_;
  int q_offset_;
  int compress_ratio_;
  int local_window_;
};

} // namespace

array packed_sparse_attention(
    const array& q,
    const array& local_kv,
    const array& pooled,
    const array& topk_indices,
    const array& sinks,
    float scale,
    int q_offset,
    int compress_ratio,
    int local_window,
    StreamOrDevice s) {
  if (q.ndim() != 4 || local_kv.ndim() != 4 || pooled.ndim() != 3 ||
      topk_indices.ndim() != 4 || sinks.ndim() != 1) {
    std::ostringstream msg;
    msg << "[attention.packed_sparse_attention] incompatible "
        << "ranks: " << q.shape() << ", " << local_kv.shape() << ", "
        << pooled.shape() << ", " << topk_indices.shape() << ", "
        << sinks.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  const bool dense_kv =
      local_kv.dtype() == bfloat16 && pooled.dtype() == bfloat16;
  if (q.shape(0) != local_kv.shape(0) || q.shape(0) != pooled.shape(0) ||
      q.shape(0) != topk_indices.shape(0) || q.shape(1) != 64 ||
      q.shape(3) != 512 || local_kv.shape(1) != 1 ||
      local_kv.shape(3) != (dense_kv ? 512 : 528) ||
      pooled.shape(2) != (dense_kv ? 512 : 288) ||
      topk_indices.shape(1) != 1 || topk_indices.shape(2) != q.shape(2) ||
      sinks.shape(0) != q.shape(1)) {
    std::ostringstream msg;
    msg << "[attention.packed_sparse_attention] incompatible "
        << "shapes: " << q.shape() << ", " << local_kv.shape() << ", "
        << pooled.shape() << ", " << topk_indices.shape() << ", "
        << sinks.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk_indices.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[attention.packed_sparse_attention] topk_indices "
        << "must be uint32, got " << topk_indices.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = q.dtype();
  if (final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[attention.packed_sparse_attention] expected "
        << "bf16 queries, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto q_cast = astype(q, final_type, stream);
  auto local_cast = local_kv;
  auto pooled_cast = pooled;
  auto sinks_cast = astype(sinks, float32, stream);

  if (PackedSparseAttentionPrimitive::unsupported(
          q_cast,
          local_cast,
          pooled_cast,
          topk_indices,
          sinks_cast,
          q_offset,
          compress_ratio,
          local_window,
          stream)) {
    throw std::invalid_argument(
        "[attention.packed_sparse_attention] unsupported DeepSeek V4.1 packed attention shape.");
  }

  Shape out_shape{q_cast.shape(0), q_cast.shape(1), q_cast.shape(2), q_cast.shape(3)};
  std::vector<array> inputs = {
      q_cast, local_cast, pooled_cast, topk_indices, sinks_cast};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<PackedSparseAttentionPrimitive>(
          stream, scale, q_offset, compress_ratio, local_window),
      std::move(inputs));
}

std::tuple<array, array> packed_kv_unpack(
    const array& local_kv,
    const array& pooled,
    StreamOrDevice s) {
  if (local_kv.ndim() != 4 || pooled.ndim() != 3) {
    throw std::invalid_argument(
        "[attention.packed_kv_unpack] local_kv must be rank 4 and pooled rank 3.");
  }
  if (local_kv.dtype() != uint8 || pooled.dtype() != uint8) {
    throw std::invalid_argument(
        "[attention.packed_kv_unpack] inputs must be packed uint8 cache rows.");
  }
  if (local_kv.shape(0) != pooled.shape(0) || local_kv.shape(1) != 1 ||
      local_kv.shape(3) != 528 || pooled.shape(2) != 288) {
    throw std::invalid_argument(
        "[attention.packed_kv_unpack] incompatible packed cache row shapes.");
  }
  auto stream = to_stream(s);
  if (PackedKVUnpackPrimitive::unsupported(local_kv, pooled, stream)) {
    throw std::invalid_argument(
        "[attention.packed_kv_unpack] unsupported packed cache layout.");
  }

  constexpr int dim = 512;
  const int B = local_kv.shape(0);
  const int localL = local_kv.shape(2);
  const int pooledL = pooled.shape(1);
  std::vector<Shape> shapes = {
      Shape{B, 1, std::max(localL, 1), dim},
      Shape{B, std::max(pooledL, 1), dim}};
  std::vector<Dtype> dtypes = {bfloat16, bfloat16};
  std::vector<array> inputs = {local_kv, pooled};
  auto outputs = array::make_arrays(
      shapes, dtypes, std::make_shared<PackedKVUnpackPrimitive>(stream),
      inputs);
  return std::make_tuple(outputs[0], outputs[1]);
}
} // namespace omlx::attention
