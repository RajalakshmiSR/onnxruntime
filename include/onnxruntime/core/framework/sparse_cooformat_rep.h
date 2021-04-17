// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/sparse_tensor.h"

namespace onnxruntime {
/// <summary>
/// This is a representation of Coo format that is generic.
/// </summary>
class SparseCooFomatRep : public SparseRep {
 public:
  SparseCooFomatRep(const TensorShape& ind_shape, const AllocatorPtr& allocator) : indices_(DataTypeImpl::GetType<int64_t>(),
                                                                                            ind_shape,
                                                                                            allocator) {
  }

  SparseCooFomatRep(const TensorShape& ind_shape, const OrtMemoryInfo& info, void* indices_data) : indices_(DataTypeImpl::GetType<int64_t>(),
                                                                                                            ind_shape,
                                                                                                            indices_data,
                                                                                                            info,
                                                                                                            0) {}

  ~SparseCooFomatRep() override;

  const Tensor& Indices() const noexcept {
    return indices_;
  }

  Tensor& MutableIndices() noexcept {
    return indices_;
  }

  Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  Status Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator, int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

 private:
  Tensor indices_;  // may be 1-D or 2-D.
};

class SparseCooBuilder {
  AllocatorPtr allocator_;
  SparseTensor* sp_;
  std::unique_ptr<SparseRep>* rep_;

 public:
  SparseCooBuilder(AllocatorPtr allocator, SparseTensor& sp, std::unique_ptr<SparseRep>& rep) noexcept : 
    allocator_(std::move(allocator)),
    sp_(&sp),
    rep_(&rep) {}

  /// <summary>
  /// Creates an owned representation using SparseTensor allocator
  /// and dense shape dimensions. Indexes will have a shape of [nnz, shape.NumDimensions]
  /// </summary>
  /// <returns></returns>
  SparseCooFomatRep* GetOrCreate();

  /// <summary>
  /// Create a non-owning representation.
  /// Indexes will have a shape of [nnz, shape.NumDimensions]
  /// The builder is going to use the same OrtMemoryInfo as for values
  /// </summary>
  /// <param name="indicies_data">ptr to indicies. No attempt will be made to verify the data</param>
  /// <returns></returns>
  SparseCooFomatRep* GetOrCreate(void* indicies_data);
};

template <>
inline const SparseCooFomatRep* SparseTensor::GetRep<SparseCooFomatRep>() const {
  ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCoo), "Expecting COO format");
  return static_cast<const SparseCooFomatRep*>(rep_.get());
}

template <>
inline SparseCooBuilder SparseTensor::RepBuilder<SparseCooBuilder>() {
  if (!rep_) {
    format_flags_ = Set(format_flags_, SparseFormatFlags::kCoo);
  }
  return SparseCooBuilder(allocator_, *this, rep_);
}

}  // namespace onnxruntime
