import SwiftUI
import UniformTypeIdentifiers

public enum AITool: String, CaseIterable, Identifiable {
    case summarizer = "Summarize"
    case tonePolish = "Polish"
    case codeAssist = "Code"
    case jsonExtractor = "Schema"
    case ragVault = "Vault"
    
    public var id: String { rawValue }
    
    public var symbol: StudioSymbol {
        switch self {
        case .summarizer: return .fileText
        case .tonePolish: return .wand
        case .codeAssist: return .code
        case .jsonExtractor: return .braces
        case .ragVault: return .bookOpen
        }
    }
    
    public var accentColor: Color {
        StudioTheme.ember
    }
}

public struct AppsView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Binding var activeModelName: String
    @Binding var discoveredModels: [ModelFileItem]
    @Binding var physicalFootprintMB: Double
    
    @StateObject private var speechManager = SpeechManager.shared
    
    @State private var selectedTool: AITool = .summarizer
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isGenerating: Bool = false
    @State private var currentTokPerSec: Double = 0.0
    @State private var totalTokensGenerated: Int = 0
    @State private var copiedToClipboard: Bool = false
    @State private var showScannerSheet: Bool = false
    @State private var selectedEngine: NanoEdgeExecutionEngine = .metalGPU
    
    // Sub-modes
    @State private var summaryStyle: String = "Key Takeaways"
    @State private var toneStyle: String = "Professional & Crisp"
    @State private var codeMode: String = "Explain Logic"
    @State private var jsonSchema: String = "Receipt / Expense"
    
    public init(activeModelName: Binding<String>, discoveredModels: Binding<[ModelFileItem]>, physicalFootprintMB: Binding<Double>) {
        self._activeModelName = activeModelName
        self._discoveredModels = discoveredModels
        self._physicalFootprintMB = physicalFootprintMB
    }
    
    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Instruments")
                        .font(StudioTheme.heading(.largeTitle, weight: .bold))
                        .tracking(-1.5)
                        .padding(.bottom, 12)

                    toolPickerSection
                    
                    modelAndEngineBar
                    
                    toolConfigurationCard
                    
                    inputEditorCard
                    
                    actionButton
                    
                    if !outputText.isEmpty || isGenerating {
                        outputResultCard
                            .transition(reduceMotion ? .opacity : .opacity.combined(with: .move(edge: .bottom)))
                    }
                }
                .frame(maxWidth: 760, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20)
                .padding(.top, 32)
                .padding(.bottom, 40)
            }
            .toolbar(.hidden, for: .navigationBar)
            .background(StudioTheme.canvas)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.28), value: selectedTool)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.28), value: !outputText.isEmpty || isGenerating)
            .sheet(isPresented: $showScannerSheet) {
                DocumentScannerView { extracted in
                    self.inputText = extracted
                }
            }
            .onAppear {
                loadInitialPresetIfNeeded()
            }
        }
    }
    
    // MARK: - Subviews
    
    private var toolPickerSection: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 22) {
                ForEach(AITool.allCases) { tool in
                    let isSelected = (selectedTool == tool)
                    Button(action: {
                        if selectedTool != tool {
                            selectedTool = tool
                            loadInitialPresetIfNeeded()
                        }
                    }) {
                        Text(tool.rawValue)
                            .font(StudioTheme.body(.subheadline, weight: isSelected ? .semibold : .regular))
                            .foregroundStyle(isSelected ? .primary : .secondary)
                            .frame(minHeight: 48)
                            .fixedSize(horizontal: true, vertical: false)
                            .overlay(alignment: .bottom) {
                                (isSelected ? StudioTheme.ember : Color.clear).frame(height: 2)
                            }
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(isSelected ? .isSelected : [])
                }
            }
        }
        .background(alignment: .bottom) { StudioTheme.border.frame(height: 1) }
    }
    
    private var modelAndEngineBar: some View {
        HStack {
            Menu {
                ForEach(discoveredModels) { model in
                    Button(action: { switchModel(model) }) {
                        HStack {
                            Text(model.name)
                            if model.name == activeModelName {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                HStack(spacing: 5) {
                    Text(activeModelName.replacingOccurrences(of: "_instruct_q4", with: "").replacingOccurrences(of: "_q4", with: ""))
                        .font(StudioTheme.body(.caption, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    StudioIcon(.chevronDown)
                        .frame(width: 9, height: 9)
                        .foregroundStyle(.secondary)
                }
            }
            
            Button(action: toggleEngine) {
                HStack(spacing: 4) {
                    Text(selectedEngine == .metalGPU ? "Metal GPU" : "NEON CPU")
                        .font(StudioTheme.body(.caption, weight: .semibold))
                    Image(systemName: "arrow.left.arrow.right")
                        .font(StudioTheme.body(.caption2))
                }
                .foregroundStyle(.secondary)
            }
            .studioPress()
            .accessibilityLabel("Switch compute engine")
            
            Spacer()
            
            if isGenerating {
                Text(String(format: "%.1f tok/s", currentTokPerSec))
                    .font(StudioTheme.body(.caption, weight: .semibold))
                    .foregroundStyle(.secondary)
            } else {
                Text(String(format: "%.0f MB", physicalFootprintMB))
                    .font(StudioTheme.body(.caption, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 12)
        .overlay(alignment: .bottom) { StudioTheme.border.frame(height: 1) }
    }
    
    private var toolConfigurationCard: some View {
        VStack(spacing: 0) {
            HStack {
                Text(modeLabel)
                    .font(StudioTheme.body(.caption, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Menu {
                    ForEach(currentModeOptions, id: \.self) { option in
                        Button {
                            setModeOption(option)
                        } label: {
                            if isModeOptionSelected(option) {
                                Label(option, systemImage: "checkmark")
                            } else {
                                Text(option)
                            }
                        }
                    }
                } label: {
                    Label(currentModeOptions.first(where: isModeOptionSelected) ?? "Choose", systemImage: "chevron.down")
                        .labelStyle(.titleAndIcon)
                        .font(StudioTheme.body(.subheadline, weight: .semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                }
            }
            .frame(minHeight: 50)
            .overlay(alignment: .bottom) { StudioTheme.border.frame(height: 1) }

            HStack {
                Text("Example input")
                    .font(StudioTheme.body(.caption, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Menu {
                    ForEach(currentPresets, id: \.name) { preset in
                        Button(preset.name) { inputText = preset.content }
                    }
                } label: {
                    Label("Insert example", systemImage: "chevron.down")
                        .font(StudioTheme.body(.subheadline, weight: .semibold))
                        .foregroundStyle(.primary)
                }
            }
            .frame(minHeight: 50)
        }
        .overlay(alignment: .bottom) { StudioTheme.border.frame(height: 1) }
    }
    
    private var inputEditorCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Source content")
                    .font(StudioTheme.body(.caption, weight: .semibold))
                    .foregroundStyle(.secondary)
                
                if !inputText.isEmpty {
                    Text("• \(inputText.count) chars")
                        .font(StudioTheme.body(.caption2))
                        .foregroundStyle(.tertiary)
                }
                
                Spacer()
                
                Button(action: { showScannerSheet = true }) {
                    HStack(spacing: 5) {
                        StudioIcon(.camera)
                            .frame(width: 12, height: 12)
                        Text("Scan document")
                    }
                    .font(StudioTheme.body(.caption, weight: .semibold))
                    .foregroundStyle(.secondary)
                }
                .studioPress()
                
                if !inputText.isEmpty {
                    Button(action: { inputText = "" }) {
                        Text("Clear")
                            .font(StudioTheme.body(.caption, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .padding(.leading, 4)
                    }
                }
            }
            
            TextEditor(text: $inputText)
                .font(selectedTool == .codeAssist ? .system(.subheadline, design: .monospaced) : StudioTheme.body(.subheadline))
                .scrollContentBackground(.hidden)
                .frame(minHeight: 110, maxHeight: 180)
                .padding(10)
                .background(StudioTheme.surfaceInput)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(StudioTheme.border, lineWidth: 1)
                )
        }
        .padding(.vertical, 14)
    }
    
    private var actionButton: some View {
        Group {
            if isGenerating {
                Button(action: cancelGeneration) {
                    HStack(spacing: 8) {
                        StudioIcon(.square)
                            .frame(width: 14, height: 14)
                        Text("Stop Generation")
                            .font(StudioTheme.body(.subheadline, weight: .semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color.red)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .studioPress()
            } else {
                let isDisabled = inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                Button(action: runInference) {
                    HStack(spacing: 8) {
                        StudioIcon(actionButtonStudioSymbol)
                            .frame(width: 15, height: 15)
                        Text(actionButtonTitle)
                            .font(StudioTheme.body(.subheadline, weight: .semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(isDisabled ? StudioTheme.surfaceRaised : StudioTheme.ember)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .disabled(isDisabled)
                .studioPress()
            }
        }
    }
    
    private var outputResultCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Output")
                    .font(StudioTheme.body(.subheadline, weight: .semibold))
                
                Spacer()
                
                if !outputText.isEmpty {
                    Button(action: {
                        speechManager.toggleSpeech(for: outputText)
                    }) {
                        HStack(spacing: 4) {
                            StudioIcon(.volume2)
                                .frame(width: 12, height: 12)
                            Text(speechManager.isSpeaking ? "Speaking" : "Listen")
                                .font(StudioTheme.body(.caption, weight: .semibold))
                        }
                        .foregroundStyle(speechManager.isSpeaking ? selectedTool.accentColor : .primary)
                    }
                    .studioPress()
                    
                    Button(action: copyToClipboard) {
                        HStack(spacing: 4) {
                            StudioIcon(copiedToClipboard ? .check : .copy)
                                .frame(width: 12, height: 12)
                            Text(copiedToClipboard ? "Copied" : "Copy")
                                .font(StudioTheme.body(.caption, weight: .semibold))
                        }
                        .foregroundStyle(copiedToClipboard ? StudioTheme.phosphor : .primary)
                    }
                    .studioPress()
                }
            }
            
            Divider()
                .overlay(StudioTheme.border)
            
            Text(outputText)
                .font((selectedTool == .codeAssist || selectedTool == .jsonExtractor) ? .system(.subheadline, design: .monospaced) : StudioTheme.body(.subheadline))
                .lineSpacing(3)
                .foregroundStyle(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
            
            if isGenerating {
                HStack(spacing: 8) {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: selectedTool.accentColor))
                        .scaleEffect(0.8)
                    Text("Synthesizing on \(selectedEngine == .metalGPU ? "A18 Pro GPU" : "ARM NEON CPU")...")
                        .font(StudioTheme.body(.caption, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)
            }
            
            if !isGenerating && !outputText.isEmpty {
                Divider()
                    .overlay(StudioTheme.border)
                HStack {
                    HStack(spacing: 4) {
                        StudioIcon(.zap)
                            .frame(width: 11, height: 11)
                            .foregroundStyle(StudioTheme.phosphor)
                        Text(String(format: "Tokens: %d • Speed: %.1f tok/s", totalTokensGenerated, currentTokPerSec))
                            .font(StudioTheme.body(.caption, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    HStack(spacing: 4) {
                        StudioIcon(.shieldCheck)
                            .frame(width: 11, height: 11)
                            .foregroundStyle(StudioTheme.phosphor)
                        Text("100% On-Device")
                            .font(StudioTheme.body(.caption, weight: .semibold))
                            .foregroundStyle(StudioTheme.phosphor)
                    }
                }
            }
        }
        .padding(16)
        .studioCard()
    }
    
    // MARK: - Helper Methods
    
    private var modeLabel: String {
        switch selectedTool {
        case .summarizer: return "Summary Format"
        case .tonePolish: return "Target Tone"
        case .codeAssist: return "Analysis Mode"
        case .jsonExtractor: return "Target JSON Schema"
        case .ragVault: return "Knowledge Vault Scope"
        }
    }
    
    private var currentModeOptions: [String] {
        switch selectedTool {
        case .summarizer: return ["Key Takeaways", "1-Sentence TL;DR", "Executive Memo"]
        case .tonePolish: return ["Professional & Crisp", "Friendly & Casual", "Executive & Decisive", "Ultra-Concise"]
        case .codeAssist: return ["Explain Logic", "Optimize for Metal/NEON", "Find Bugs & Fix"]
        case .jsonExtractor: return ["Receipt / Expense", "Action Items", "Contact Card"]
        case .ragVault: return ["All Vault Docs", "Apple A18 Pro", "Quantization", "Jetsam Paging"]
        }
    }
    
    private func isModeOptionSelected(_ opt: String) -> Bool {
        switch selectedTool {
        case .summarizer: return summaryStyle == opt
        case .tonePolish: return toneStyle == opt
        case .codeAssist: return codeMode == opt
        case .jsonExtractor: return jsonSchema == opt
        case .ragVault: return opt == "All Vault Docs"
        }
    }
    
    private func setModeOption(_ opt: String) {
        switch selectedTool {
        case .summarizer: summaryStyle = opt
        case .tonePolish: toneStyle = opt
        case .codeAssist: codeMode = opt
        case .jsonExtractor: jsonSchema = opt
        case .ragVault: break
        }
    }
    
    private var actionButtonTitle: String {
        switch selectedTool {
        case .summarizer: return "Generate Summary"
        case .tonePolish: return "Polish Text"
        case .codeAssist: return "Analyze Code"
        case .jsonExtractor: return "Extract JSON Structure"
        case .ragVault: return "Query Knowledge Vault"
        }
    }
    
    private var actionButtonStudioSymbol: StudioSymbol {
        switch selectedTool {
        case .summarizer: return .fileText
        case .tonePolish: return .wand
        case .codeAssist: return .play
        case .jsonExtractor: return .braces
        case .ragVault: return .bookOpen
        }
    }
    
    private struct PresetItem {
        let name: String
        let content: String
    }
    
    private var currentPresets: [PresetItem] {
        switch selectedTool {
        case .summarizer:
            return [
                PresetItem(
                    name: "A18 Pro Architecture",
                    content: "The Apple A18 Pro system-on-chip features a 6-core GPU with next-generation neural accelerators, 16-core Neural Engine, and unified memory bandwidth exceeding 60 GB/s. Built on TSMC's second-generation 3nm (N3E) process, it integrates Scalable Matrix Extension (SME) units in the performance CPU cores and hardware-accelerated ray tracing in the GPU. These architectural advances enable high-throughput local LLM execution, zero-copy unified DRAM sharing between CPU and Metal GPU, and low-power INT4 matrix-vector multiplication without thermal throttling."
                ),
                PresetItem(
                    name: "Edge LLM Deployment",
                    content: "Deploying large language models on mobile edge devices requires addressing three major bottlenecks: memory bandwidth, physical footprint (Jetsam limits), and sustained thermal envelope. By employing asymmetric 4-bit block quantization (Q4_0 and Apple MQ4) and memory-mapped (mmap) zero-copy weight loading, models up to 7B parameters can decode tokens at interactive speeds (>30 tok/s) directly on unified device memory without triggering out-of-memory terminations."
                ),
                PresetItem(
                    name: "Sprint Sync Minutes",
                    content: "Project Sync September 17. The mobile inference team reviewed the 10-model suite rollout on iPhone 16 Pro. Sarah confirmed that Q4_0 and MQ4 kernels pass numerical parity tests against ARM NEON. David verified that physical memory footprint remains under 350 MB for the 1.5B model. Next milestone is shipping specialized productivity micro-apps (Summarizer, Tone Polish, Code Assist, JSON Extractor) before Q4 release."
                )
            ]
        case .tonePolish:
            return [
                PresetItem(
                    name: "Late Deliverable",
                    content: "hey sorry for the delay on the report i had some personal stuff come up and also the build was broken yesterday but i should be able to send it over soon hopefully by tomorrow afternoon"
                ),
                PresetItem(
                    name: "Bug Feedback",
                    content: "the new app is kind of lagging when we click the button and sometimes it feels like the model takes forever to respond, can you guys fix it asap because our users are getting mad"
                ),
                PresetItem(
                    name: "Demo Follow-up",
                    content: "thanks for hopping on the call earlier today to check out our demo. let me know what you think about moving forward or if you need any more info from our side"
                )
            ]
        case .codeAssist:
            return [
                PresetItem(
                    name: "Quantized GEMV",
                    content: "void matvec_naive(const float* A, const float* x, float* y, int M, int K) {\n    for (int i = 0; i < M; i++) {\n        float sum = 0.0f;\n        for (int j = 0; j < K; j++) {\n            sum += A[i * K + j] * x[j];\n        }\n        y[i] = sum;\n    }\n}"
                ),
                PresetItem(
                    name: "Zero-Copy Mmap",
                    content: "void* load_weights(const char* path, size_t* out_size) {\n    int fd = open(path, O_RDONLY);\n    struct stat sb;\n    fstat(fd, &sb);\n    *out_size = sb.st_size;\n    void* ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);\n    close(fd);\n    return ptr;\n}"
                ),
                PresetItem(
                    name: "Swift Concurrency",
                    content: "func fetchTokens() async -> [String] {\n    var tokens: [String] = []\n    for await token in tokenStream {\n        tokens.append(token)\n    }\n    return tokens\n}"
                )
            ]
        case .jsonExtractor:
            return [
                PresetItem(
                    name: "Store Receipt",
                    content: "Apple Store Fifth Ave, Date: Sept 17, 2026. Purchased 1x iPhone 16 Pro 256GB Black Titanium for $999.00, 1x MagSafe Case for $49.00. Subtotal: $1048.00. Tax (8.875%): $93.01. Total: $1141.01. Paid with Apple Card ending 4821."
                ),
                PresetItem(
                    name: "Sprint Tasks",
                    content: "Sprint 14 Planning: Sarah needs to optimize the 4-bit Metal shader by Friday (Priority: High). David is assigned to test Jetsam memory limits on iOS 18 by Thursday (Priority: Medium). Alex will review PR #42 for JSON schema validation by Wednesday (Priority: Low)."
                ),
                PresetItem(
                    name: "Business Card",
                    content: "Met with Dr. Elena Rostova at the AI Mobile Summit. She is VP of Engineering at SiliconCore Labs. Her email is elena.rostova@siliconcore.ai and mobile is +1-415-555-0199. Office is located in San Francisco, CA."
                )
            ]
        case .ragVault:
            return [
                PresetItem(
                    name: "A18 Pro Memory Bandwidth",
                    content: "How much memory bandwidth does the Apple A18 Pro unified memory bus provide, and how does 2-row tiling save DRAM traffic?"
                ),
                PresetItem(
                    name: "Jetsam Eviction Strategy",
                    content: "What paging techniques and memory allocations does NanoEdge use to prevent iOS Jetsam memory kills?"
                ),
                PresetItem(
                    name: "Quantization Formats",
                    content: "Compare Q4_0 block quantization with Apple-optimized MQ4_Apple formats."
                )
            ]
        }
    }
    
    private func loadInitialPresetIfNeeded() {
        if inputText.isEmpty, let first = currentPresets.first {
            inputText = first.content
        }
    }
    
    // MARK: - Actions
    
    private func runInference() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        
        outputText = ""
        isGenerating = true
        copiedToClipboard = false
        currentTokPerSec = 0.0
        totalTokensGenerated = 0
        speechManager.stopSpeaking()
        
        let systemPrompt: String
        let userPrompt: String
        
        switch selectedTool {
        case .summarizer:
            systemPrompt = "Format: \(summaryStyle)."
            userPrompt = "Summarize the following text using \(summaryStyle):\n\n\(text)"
        case .tonePolish:
            systemPrompt = "Target tone: \(toneStyle)."
            userPrompt = "Rewrite in a \(toneStyle) tone:\n\n\(text)"
        case .codeAssist:
            systemPrompt = "Action: \(codeMode)."
            userPrompt = "Perform \(codeMode) on the following code:\n\n\(text)"
        case .jsonExtractor:
            systemPrompt = "Extract strictly valid JSON for: \(jsonSchema)."
            userPrompt = "Extract key entities from the following text into JSON format:\n\n\(text)"
        case .ragVault:
            let hits = LocalRAGStore.shared.search(query: text, topK: 3)
            let context = hits.map { "• [\($0.documentTitle)] \($0.snippet)" }.joined(separator: "\n\n")
            systemPrompt = "100% Offline RAG Knowledge Vault Assistant. Ground answer on verified excerpts."
            userPrompt = """
            Context from 100% Offline Local Knowledge Vault:
            \(context)
            
            Question: \(text)
            Answer accurately based on the verified documents above.
            """
        }
        
        NanoEdgeBridge.sharedInstance().executionEngine = selectedEngine
        NanoEdgeBridge.sharedInstance().generateStreaming(
            withPrompt: userPrompt,
            systemPrompt: systemPrompt,
            maxTokens: 500,
            temperature: 0.5,
            onToken: { token, tokPerSec in
                self.outputText += token
                self.currentTokPerSec = tokPerSec
                self.totalTokensGenerated += 1
            },
            onComplete: { fullGeneratedText, totalTime, avgTokPerSec, ttftMs in
                if !fullGeneratedText.isEmpty {
                    self.outputText = fullGeneratedText
                }
                self.currentTokPerSec = avgTokPerSec
                self.isGenerating = false
            }
        )
    }
    
    private func cancelGeneration() {
        NanoEdgeBridge.sharedInstance().cancelGeneration()
        isGenerating = false
    }
    
    private func toggleEngine() {
        if selectedEngine == .metalGPU {
            selectedEngine = .neonCPU
        } else {
            selectedEngine = .metalGPU
        }
        NanoEdgeBridge.sharedInstance().executionEngine = selectedEngine
    }
    
    private func copyToClipboard() {
        UIPasteboard.general.string = outputText
        copiedToClipboard = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            self.copiedToClipboard = false
        }
    }
    
    private func switchModel(_ model: ModelFileItem) {
        do {
            try NanoEdgeBridge.sharedInstance().loadModel(fromPath: model.path)
            activeModelName = model.name
        } catch {
            print("Failed switching to model: \(error)")
        }
    }
}
