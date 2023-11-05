#include "3rdparty/INIReader.h"
#include "src/turbomind/kernels/gemm_s_f16/format.h"
#include <iostream>

using namespace std;
namespace ft = turbomind;

int main()
{
    std::string                     model_name_;
    size_t                          head_num_;
    size_t                          kv_head_num_;
    size_t                          size_per_head_;
    size_t                          inter_size_;
    size_t                          num_layer_;
    size_t                          vocab_size_;
    turbomind::LlamaAttentionParams attn_params_;
    float                           norm_eps_;
    int                             max_batch_size_;
    int                             max_context_token_num_;
    int                             session_len_;
    int                             step_length_;
    int                             start_id_;
    int                             end_id_;
    int                             cache_max_entry_count_;
    int                             cache_chunk_size_;
    int                             use_context_fmha_;
    size_t                          tensor_para_size_;
    size_t                          pipeline_para_size_;
    ft::WeightType                  weight_type_;
    bool                            attn_bias_;
    int                             quant_policy_;
    int                             group_size_;

    std::string       model_dir = "./workspace/triton_models/weights";
    const std::string inifile{model_dir + "/config.ini"};
    INIReader         reader = INIReader(inifile);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << inifile << "'\n";
        ft::FT_CHECK(false);
    }

    model_name_            = reader.Get("llama", "model_name");
    head_num_              = reader.GetInteger("llama", "head_num");
    kv_head_num_           = reader.GetInteger("llama", "kv_head_num", 0);
    size_per_head_         = reader.GetInteger("llama", "size_per_head");
    inter_size_            = reader.GetInteger("llama", "inter_size");
    num_layer_             = reader.GetInteger("llama", "num_layer");
    vocab_size_            = reader.GetInteger("llama", "vocab_size");
    norm_eps_              = reader.GetFloat("llama", "norm_eps");
    start_id_              = reader.GetInteger("llama", "start_id");
    end_id_                = reader.GetInteger("llama", "end_id");
    max_batch_size_        = reader.GetInteger("llama", "max_batch_size", 0);
    max_context_token_num_ = reader.GetInteger("llama", "max_context_token_num", 0);
    session_len_           = reader.GetInteger("llama", "session_len", 0);
    step_length_           = reader.GetInteger("llama", "step_length", 0);
    cache_max_entry_count_ = reader.GetInteger("llama", "cache_max_entry_count", 0);
    use_context_fmha_      = reader.GetInteger("llama", "use_context_fmha", 1);
    cache_chunk_size_      = reader.GetInteger("llama", "cache_chunk_size", 0);
    attn_bias_             = reader.GetInteger("llama", "attn_bias", 0);
    quant_policy_          = reader.GetInteger("llama", "quant_policy", 0);
    group_size_            = reader.GetInteger("llama", "group_size", 0);

    attn_params_.rotray_embedding_dim    = reader.GetInteger("llama", "rotary_embedding");
    attn_params_.rotary_embedding_base   = reader.GetFloat("llama", "rope_theta", 10000.0f);
    attn_params_.max_position_embeddings = reader.GetInteger("llama", "max_position_embeddings", 0);
    attn_params_.use_dynamic_ntk         = reader.GetInteger("llama", "use_dynamic_ntk", 0);
    attn_params_.use_logn_attn           = reader.GetInteger("llama", "use_logn_attn", 0);

    if (max_context_token_num_ <= max_batch_size_) {
        max_context_token_num_ *= session_len_;
    }

    shared_state_          = std::make_shared<typename ft::LlamaV2<T>::SharedState>();
    shared_state_->barrier = std::make_shared<ft::Barrier>(tensor_para_size);

    const auto device_count = ft::getDeviceCount();
    shared_instances_.resize(device_count);
    shared_mutexes_.resize(device_count);

    const std::string weight_type_str = reader.Get("llama", "weight_type");
    weight_type_                      = ft::WeightType::kFP16;

    cout << "create llama model success..." << endl;

    return 0;
}