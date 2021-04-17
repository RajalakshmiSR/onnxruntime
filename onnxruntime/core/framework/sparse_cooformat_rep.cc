// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/sparse_cooformat_rep.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
SparseCooFomatRep ::~SparseCooFomatRep() = default;

Status SparseCooFomatRep::Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
                               int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = make_unique<SparseCooFomatRep>(indices_.Shape(), allocator);
  ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(indices_, rep_copy->indices_, exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

Status SparseCooFomatRep::Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
                               int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const {
  auto rep_copy = make_unique<SparseCooFomatRep>(indices_.Shape(), allocator);
  ORT_RETURN_IF_ERROR(data_transfer.CopyTensor(rep_copy->Indices(), rep_copy->MutableIndices(), exec_q_id));
  dst_rep = std::move(rep_copy);
  return Status::OK();
}

SparseCooFomatRep* SparseCooBuilder::GetOrCreate() {
  if (rep_->get()) {
    return static_cast<SparseCooFomatRep*>(rep_->get());
  }

  ORT_ENFORCE(allocator_ != nullptr, "Must have an allocator set with Sparse Tensor instance");
  auto result = new SparseCooFomatRep({sp_->Shape().Size(),
                                     static_cast<int64_t>(sp_->Shape().NumDimensions())},
                                     allocator_);
  rep_->reset(result);
  return result;
}

SparseCooFomatRep* SparseCooBuilder::GetOrCreate(void* indices_data) {
  if (rep_->get()) {
    return static_cast<SparseCooFomatRep*>(rep_->get());
  }

  ORT_ENFORCE(allocator_ == nullptr, "Should not have an allocator set");
  auto result = new SparseCooFomatRep({sp_->Shape().Size(),
                                       static_cast<int64_t>(sp_->Shape().NumDimensions())},
                                      sp_->Location(), indices_data);
  rep_->reset(result);
  return result;
}

}  // namespace onnxruntime