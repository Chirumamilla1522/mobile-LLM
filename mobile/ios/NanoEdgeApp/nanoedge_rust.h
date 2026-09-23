#ifndef NANOEDGE_RUST_H
#define NANOEDGE_RUST_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t max_tokens;
    float temperature;
    float repetition_penalty;
    float min_p;
} NanoEdgeGenConfig;

typedef bool (*NanoEdgeTokenCallback)(const char* token, double tokens_per_sec, void* user_data);
typedef void (*NanoEdgeCompleteCallback)(const char* full_text, double total_time_sec, double avg_tok_per_sec, double ttft_ms, void* user_data);

void* nanoedge_rust_init(const char* model_path, const char* vocab_path);

bool nanoedge_rust_generate(
    void* handle,
    const char* prompt,
    const char* system_prompt,
    const NanoEdgeGenConfig* config,
    NanoEdgeTokenCallback token_cb,
    NanoEdgeCompleteCallback complete_cb,
    void* user_data
);

void nanoedge_rust_cancel(void* handle);

void nanoedge_rust_set_execution_engine(void* handle, uint32_t engine);
bool nanoedge_rust_set_kv_precision(void* handle, uint32_t bits);

void nanoedge_rust_free(void* handle);

#ifdef __cplusplus
}
#endif

#endif // NANOEDGE_RUST_H
