#import <Metal/Metal.h>
#include "mllm/model_format.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <numeric>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" void* nanoedge_metal_bridge_create(const void*, size_t, uint32_t, float);
extern "C" void nanoedge_metal_bridge_destroy(void*);
extern "C" bool nanoedge_metal_bridge_forward(void*, uint32_t, uint32_t, bool, float*);
extern "C" bool nanoedge_metal_bridge_prefill(void*, const int32_t*, uint32_t, uint32_t);
extern "C" bool nanoedge_metal_bridge_copy_logits(void*, float*);
extern "C" int32_t nanoedge_metal_bridge_sample(void*, const int32_t*, uint32_t, float, float, float);
extern "C" void nanoedge_metal_bridge_set_prefill_hidden_limit(void*, uint32_t);
extern "C" bool nanoedge_metal_bridge_set_kv_precision(void*, uint32_t);
extern "C" size_t nanoedge_metal_bridge_kv_cache_bytes(void*);

namespace {
using Clock = std::chrono::steady_clock;

struct ModelFile {
    int fd{-1};
    size_t size{};
    void* bytes{MAP_FAILED};
    const mllm::ModelHeader* header{};
    explicit ModelFile(const char* path) {
        fd = open(path, O_RDONLY); struct stat st{};
        if (fd < 0 || fstat(fd, &st)) throw std::runtime_error("cannot open model");
        size = static_cast<size_t>(st.st_size); bytes = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (bytes == MAP_FAILED) throw std::runtime_error("cannot map model");
        header = static_cast<const mllm::ModelHeader*>(bytes);
    }
    ~ModelFile() { if (bytes != MAP_FAILED) munmap(bytes, size); if (fd >= 0) close(fd); }
};

struct Bridge {
    void* value{};
    Bridge(const ModelFile& model, uint32_t context) {
        const auto begin = Clock::now();
        value = nanoedge_metal_bridge_create(model.bytes, model.size, context, 100000.0f);
        setup_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        if (!value) throw std::runtime_error("Metal bridge rejected model");
    }
    ~Bridge() { nanoedge_metal_bridge_destroy(value); }
    double setup_ms{};
};

struct Stats { double mean{}, median{}, p95{}; };
Stats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size(),
            samples[samples.size()/2], samples[static_cast<size_t>(std::ceil(samples.size()*0.95))-1]};
}

template<class F> Stats measure(F&& operation, int warmup=5, int iterations=30) {
    for (int i=0;i<warmup;++i) if (!operation()) throw std::runtime_error("warmup failed");
    std::vector<double> samples; samples.reserve(iterations);
    for(int i=0;i<iterations;++i){const auto begin=Clock::now(); if(!operation()) throw std::runtime_error("measurement failed");
        samples.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());}
    return summarize(std::move(samples));
}

void print_stats(const char* label, const Stats& value, uint32_t tokens=1) {
    std::printf("%-24s %9.3f %9.3f %9.3f %11.1f\n", label, value.median, value.p95, value.mean,
                1000.0 * tokens / value.median);
}

Stats sequential_prefill(Bridge& bridge, const std::vector<int32_t>& prompt, std::vector<float>& logits) {
    return measure([&]{
        for(uint32_t i=0;i<prompt.size();++i)
            if(!nanoedge_metal_bridge_forward(bridge.value,prompt[i],i,i+1==prompt.size(),i+1==prompt.size()?logits.data():nullptr)) return false;
        return true;
    });
}

Stats batched_prefill(Bridge& bridge, const std::vector<int32_t>& prompt, std::vector<float>& logits) {
    return measure([&]{return nanoedge_metal_bridge_prefill(bridge.value,prompt.data(),prompt.size(),0) &&
                              nanoedge_metal_bridge_copy_logits(bridge.value,logits.data());});
}

Stats decode(Bridge& bridge, const std::vector<int32_t>& prompt) {
    if (!nanoedge_metal_bridge_prefill(bridge.value,prompt.data(),prompt.size(),0)) {
        for (uint32_t i=0;i<prompt.size();++i)
            if (!nanoedge_metal_bridge_forward(bridge.value,prompt[i],i,i+1==prompt.size(),nullptr))
                throw std::runtime_error("decode setup failed");
    }
    if (nanoedge_metal_bridge_sample(bridge.value,prompt.data(),prompt.size(),0.2f,1.15f,0.05f)<0)
        throw std::runtime_error("decode setup failed");
    const uint32_t position=prompt.size();
    return measure([&]{return nanoedge_metal_bridge_forward(bridge.value,42,position,true,nullptr) &&
        nanoedge_metal_bridge_sample(bridge.value,prompt.data(),prompt.size(),0.2f,1.15f,0.05f)>=0;});
}

struct Quality { double nll{}, perplexity{}; std::vector<uint32_t> greedy; };
Quality score_sequence(Bridge& bridge, const std::vector<int32_t>& tokens, uint32_t vocab) {
    std::vector<float> logits(vocab); double nll=0.0; std::vector<uint32_t> greedy;
    for(uint32_t pos=0;pos+1<tokens.size();++pos){
        if(!nanoedge_metal_bridge_forward(bridge.value,tokens[pos],pos,true,logits.data())) throw std::runtime_error("quality pass failed");
        const float maximum=*std::max_element(logits.begin(),logits.end()); double sum=0.0;
        for(float value:logits)sum+=std::exp(double(value-maximum));
        nll+=double(maximum)+std::log(sum)-logits[tokens[pos+1]];
        greedy.push_back(static_cast<uint32_t>(std::max_element(logits.begin(),logits.end())-logits.begin()));
    }
    nll/=tokens.size()-1; return {nll,std::exp(nll),std::move(greedy)};
}
}

int main(int argc,char** argv){
    if(argc<2){std::fprintf(stderr,"usage: runtime_comparison MODEL_Q4 [MODEL_MQ4]\n");return 2;}
    try{
        ModelFile model(argv[1]); const uint32_t context=std::min(512u,model.header->max_seq_len);
        std::vector<int32_t> prompt(32); std::iota(prompt.begin(),prompt.end(),1);
        for(auto& token:prompt) token%=static_cast<int32_t>(model.header->vocab_size);
        std::vector<float> sequential_logits(model.header->vocab_size),batched_logits(model.header->vocab_size);
        Bridge sequential(model,context),batched(model,context),decoder(model,context);
        nanoedge_metal_bridge_set_prefill_hidden_limit(batched.value,UINT32_MAX);
        const auto sequential_stats=sequential_prefill(sequential,prompt,sequential_logits);
        const auto batched_stats=batched_prefill(batched,prompt,batched_logits);
        const auto decode_stats=decode(decoder,std::vector<int32_t>(prompt.begin(),prompt.begin()+8));
        float max_abs=0.0f,mean_abs=0.0f;uint32_t greedy_a=0,greedy_b=0;
        for(uint32_t i=0;i<model.header->vocab_size;++i){const float d=std::abs(sequential_logits[i]-batched_logits[i]);
            max_abs=std::max(max_abs,d);mean_abs+=d;if(sequential_logits[i]>sequential_logits[greedy_a])greedy_a=i;
            if(batched_logits[i]>batched_logits[greedy_b])greedy_b=i;}
        mean_abs/=model.header->vocab_size;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        std::printf("\nDevice: %s | model: %ux%u, %u layers | trials: 30 + 5 warmup\n",
                    [[device name] UTF8String],model.header->hidden_dim,model.header->intermediate_dim,model.header->num_layers);
        std::printf("%-24s %9s %9s %9s %11s\n","Path","P50 ms","P95 ms","Mean ms","tokens/s");
        print_stats("Sequential prefill",sequential_stats,prompt.size());
        print_stats("Batched prefill",batched_stats,prompt.size());
        print_stats("Decode + GPU sample",decode_stats);
        std::printf("\nPrefill speedup: %.2fx | max abs: %.6f | mean abs: %.6f | greedy: %u/%u %s\n",
                    sequential_stats.median/batched_stats.median,max_abs,mean_abs,greedy_a,greedy_b,greedy_a==greedy_b?"PASS":"FAIL");
        const size_t fp16=static_cast<size_t>(model.header->num_layers)*2*model.header->num_kv_heads*context*
                            (model.header->hidden_dim/model.header->num_heads)*2;
        const size_t q8=static_cast<size_t>(model.header->num_layers)*2*model.header->num_kv_heads*context*
                          ((model.header->hidden_dim/model.header->num_heads+31)/32)*34;
        std::printf("KV cache @ %u: FP16 %.2f MiB | Q8 %.2f MiB | saved %.1f%%\n",context,fp16/1048576.0,q8/1048576.0,100.0*(fp16-q8)/fp16);
        std::printf("Pipeline setup: %.2f ms (includes runtime shader compilation)\n",decoder.setup_ms);
        const int32_t seed[]={1,9690,198,2683,359,253,5356,5646,11173,3365,3511,308,34519,28,7018,411,407,19712,8182,2,198,
                              1,4093,198,1780,314,260,3575,282,4649,47,2,198,1,520,9531,198};
        std::vector<int32_t> long_tokens; const uint32_t quality_length=context;
        while(long_tokens.size()<quality_length)for(int32_t token:seed){long_tokens.push_back(token%static_cast<int32_t>(model.header->vocab_size));if(long_tokens.size()==quality_length)break;}
        Bridge fp16_quality(model,context),q8_quality(model,context);
        if(!nanoedge_metal_bridge_set_kv_precision(fp16_quality.value,16))throw std::runtime_error("FP16 KV unavailable");
        const auto fp16_score=score_sequence(fp16_quality,long_tokens,model.header->vocab_size);
        const auto q8_score=score_sequence(q8_quality,long_tokens,model.header->vocab_size);
        size_t agreement=0;for(size_t i=0;i<fp16_score.greedy.size();++i)agreement+=fp16_score.greedy[i]==q8_score.greedy[i];
        std::printf("\nLong-context KV quality (%u tokens):\n",quality_length);
        std::printf("FP16 NLL %.6f | PPL %.3f | cache %.2f MiB\n",fp16_score.nll,fp16_score.perplexity,
                    nanoedge_metal_bridge_kv_cache_bytes(fp16_quality.value)/1048576.0);
        std::printf("Q8   NLL %.6f | PPL %.3f | cache %.2f MiB | PPL ratio %.6fx | greedy agreement %.1f%%\n",
                    q8_score.nll,q8_score.perplexity,nanoedge_metal_bridge_kv_cache_bytes(q8_quality.value)/1048576.0,
                    q8_score.perplexity/fp16_score.perplexity,100.0*agreement/fp16_score.greedy.size());
        if(argc>2){
            ModelFile alternative(argv[2]);
            if(alternative.header->hidden_dim!=model.header->hidden_dim||alternative.header->intermediate_dim!=model.header->intermediate_dim||
               alternative.header->num_layers!=model.header->num_layers||alternative.header->vocab_size!=model.header->vocab_size)
                throw std::runtime_error("alternative model dimensions do not match");
            Bridge alternative_decoder(alternative,context);
            const auto alternative_stats=decode(alternative_decoder,std::vector<int32_t>(prompt.begin(),prompt.begin()+8));
            std::printf("\nMatched-shape format comparison:\n");
            print_stats("Primary decode",decode_stats);
            print_stats("Alternative decode",alternative_stats);
            std::printf("Alternative/primary speed: %.2fx\n",decode_stats.median/alternative_stats.median);
        }
        return greedy_a==greedy_b&&max_abs<0.5f?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"benchmark failed: %s\n",e.what());return 1;}
}
