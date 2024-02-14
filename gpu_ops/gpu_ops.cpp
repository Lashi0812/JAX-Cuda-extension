#include "kernels.h"
#include "kernel_nanobind_helpers.h"

namespace {
nanobind::dict RMSNormRegistrations() {
    nanobind::dict dict;
    dict["rms_forward_affine_mixed_dtype"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rms_forward_affine_mixed_dtypes);
    dict["rms_backward_affine"] = gpu_ops::EncapsulateFunction(gpu_ops::rms_backward_affine);
    return dict;
}

NB_MODULE(gpu_ops, m) {
    m.def("get_rms_norm_registrations", &RMSNormRegistrations);
    m.def(
      "create_rms_norm_descriptor",
      [](
        int                  n1,
        int                  n2,
        double               eps,
        gpu_ops::ElementType x_type,
        gpu_ops::ElementType w_type,
        int                  part_grad_size) {
          return gpu_ops::PackDescriptor(
            gpu_ops::RMSNormDescriptor{n1, n2, eps, x_type, w_type, part_grad_size});
      });

    nanobind::enum_<gpu_ops::ElementType>(m, "ElementType")
      .value("BF16", gpu_ops::ElementType::BF16)
      .value("F16", gpu_ops::ElementType::F16)
      .value("F32", gpu_ops::ElementType::F32)
      .value("F64", gpu_ops::ElementType::F64);
}
} // namespace