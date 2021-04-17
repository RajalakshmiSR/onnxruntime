// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "SnpeLib.h"
#include "snpe_execution_provider.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

class Snpe : public OpKernel {
 public:
  explicit Snpe(const OpKernelInfo& info) : OpKernel(info) {
    const auto snpeEp = static_cast<const SNPEExecutionProvider*>(info.GetExecutionProvider());
    const auto payload = info.GetAttrOrDefault<std::string>("payload", "");
    output_dims_ = info.GetAttrsOrDefault<int64_t>("output1_shape");
    const bool enforeDsp = snpeEp->GetEnforceDsp();
    snpe_rt_ = SnpeLib::SnpeLibFactory(reinterpret_cast<const unsigned char*>(payload.c_str()), payload.length(), nullptr, enforeDsp);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* input_tensor = context->Input<Tensor>(0);
    const auto input_data = input_tensor->DataRaw();
    const size_t input_size = input_tensor->Shape().Size();
    const size_t input_element_byte_size = input_tensor->DataType()->Size();

    TensorShape output_shape = TensorShape(output_dims_);
    auto output_tensor = context->Output(0, output_shape);
    auto output_data = output_tensor->MutableDataRaw();
    const size_t output_element_byte_size = output_tensor->DataType()->Size();

    snpe_rt_->SnpeProcess(
        static_cast<const unsigned char*>(input_data),
        input_size * input_element_byte_size,
        static_cast<unsigned char*>(output_data),
        output_shape.Size() * output_element_byte_size);

    return Status::OK();
  }

 private:
  std::vector<int64_t> output_dims_;
  std::unique_ptr<SnpeLib> snpe_rt_;
};
}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime