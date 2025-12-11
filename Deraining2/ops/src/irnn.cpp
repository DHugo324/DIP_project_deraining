#include <torch/extension.h>
#include <vector>

// 宣告 CUDA 函數
std::vector<torch::Tensor> irnn_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

std::vector<torch::Tensor> irnn_backward_cuda(
    torch::Tensor grad_output, torch::Tensor input, 
    torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

// 檢查器
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> irnn_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    return irnn_forward_cuda(input, weight, bias);
}

std::vector<torch::Tensor> irnn_backward(
    torch::Tensor grad_output, torch::Tensor input, 
    torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    return irnn_backward_cuda(grad_output, input, weight, bias, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &irnn_forward, "IRNN forward (CUDA)");
    m.def("backward", &irnn_backward, "IRNN backward (CUDA)");
}
