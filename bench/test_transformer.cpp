#include "mllm/transformer_engine.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string model_path = "models/suite/smollm2_135m_q4.mllm";
    std::string vocab_path = "models/suite/smollm2_vocab.json";

    std::cout << "[Test] Loading " << model_path << "..." << std::endl;
    auto model = std::make_shared<mllm::MemoryMappedModel>(model_path);
    std::cout << "[Test] Model loaded. Tensors: " << model->descriptors().size() << std::endl;

    mllm::TransformerEngine engine(model);
    std::cout << "[Test] Loading vocab from " << vocab_path << "..." << std::endl;
    if (!engine.load_vocab(vocab_path)) {
        std::cerr << "[Test] Failed to load vocab!" << std::endl;
        return 1;
    }
    std::cout << "[Test] Vocab loaded: " << engine.vocab_size() << " tokens." << std::endl;

    mllm::GenerationConfig cfg;
    cfg.temperature = 0.0f;
    cfg.repetition_penalty = 1.0f;
    cfg.min_p = 0.0f;
    cfg.max_tokens = 25;

    std::string prompt = "What is the capital of France?";
    std::string system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

    std::cout << "\n[Test] Prompt: '" << prompt << "'" << std::endl;
    std::cout << "[Test] Generating tokens..." << std::endl;

    // Test with exact token prompt
    std::vector<int32_t> exact_tokens = {
        1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407, 19712, 8182, 2, 198,
        1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 2, 198,
        1, 520, 9531, 198
    };

    std::cout << "[Test] Prefilling " << exact_tokens.size() << " exact prompt tokens..." << std::endl;

    engine.generate_from_tokens(
        exact_tokens,
        cfg,
        [](const std::string& tok, double tok_s) {
            std::cout << "{" << tok << "}" << std::flush;
            return true;
        },
        [](const std::string& full, double total_s, double avg_s) {
            std::cout << "\n\n[Test] Complete! Generated in " << total_s << "s (" << avg_s << " tok/s)" << std::endl;
        }
    );

    return 0;
}
