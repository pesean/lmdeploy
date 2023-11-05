#include "src/turbomind/triton_backend/llama/LlamaTritonModel.h"
#include <iostream>

using namespace std;

int main()
{
    std::string model_dir                = "./workspace/triton_models/weights";
    std::string data_type                = "fp16";
    int         tensor_para_size         = 1;
    int         pipeline_para_size       = 1;
    int         enable_custom_all_reduce = 0

        if (data_type == "half" || data_type == "fp16" || data_type == "int4")
    {
        auto model = std::make_shared<LlamaTritonModel<half>>(
            tensor_para_size, pipeline_para_size, enable_custom_all_reduce, model_dir);
        model->setFfiLock(gil_control);
    }
    else
    {
        auto model = std::make_shared<LlamaTritonModel<float>>(
            tensor_para_size, pipeline_para_size, enable_custom_all_reduce, model_dir);
        model->setFfiLock(gil_control);
    }
    cout << "create instance success..." << endl;
    return 0;
}