import SwiftUI

public struct ChatMessage: Identifiable, Equatable {
    public let id = UUID()
    public let isUser: Bool
    public var text: String
    public let timestamp: Date = Date()
    public var tokPerSec: Double = 0.0
    public var isStreaming: Bool = false
    public var engineUsed: String = "Metal GPU"
    public var toolResult: ToolCallResult? = nil
    public var ragCitations: [String] = []
}

public enum IntelligenceTier: String, CaseIterable, Identifiable {
    case fast1B = "Feather 1B (663 MB)"
    case deep3B = "Titanium 3B (1.7 GB)"
    
    public var id: String { rawValue }
    
    public var shortLabel: String {
        switch self {
        case .fast1B: return "Feather 1B"
        case .deep3B: return "Titanium 3B"
        }
    }
    
    public var memoryTierMB: Double {
        switch self {
        case .fast1B: return 730.0
        case .deep3B: return 1950.0
        }
    }
}

public struct ChatView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Binding var activeModelName: String
    @Binding var discoveredModels: [ModelFileItem]
    @Binding var physicalFootprintMB: Double
    @Binding var selectedTier: IntelligenceTier
    
    @StateObject private var speechManager = SpeechManager.shared
    
    @State private var messages: [ChatMessage] = []
    @State private var inputPrompt: String = ""
    @State private var isGenerating: Bool = false
    @State private var currentTokPerSec: Double = 0.0
    @State private var selectedPersona: String = "Helpful Assistant"
    @State private var selectedEngine: NanoEdgeExecutionEngine = .metalGPU
    @State private var intelligenceMode: String = "On-Device A18 Pro"
    @State private var copiedMessageId: UUID? = nil
    
    // Advanced Feature States
    @State private var showVoiceOrb: Bool = false
    @State private var showCameraVision: Bool = false
    @State private var isRAGEnabled: Bool = true
    
    let personas = [
        "Helpful Assistant",
        "Code Architect",
        "Concise Editor",
        "Creative Writer"
    ]
    
    let quickPrompts = [
        "Solve step-by-step: If a train travels at 90 km/h for 2h 40m, what's the distance?",
        "Write a thread-safe Swift actor with debounce",
        "Explain how on-device LLM inference works",
        "Compare unified memory bandwidth on A18 Pro vs M4"
    ]
    
    public init(
        activeModelName: Binding<String>,
        discoveredModels: Binding<[ModelFileItem]>,
        physicalFootprintMB: Binding<Double>,
        selectedTier: Binding<IntelligenceTier>
    ) {
        self._activeModelName = activeModelName
        self._discoveredModels = discoveredModels
        self._physicalFootprintMB = physicalFootprintMB
        self._selectedTier = selectedTier
    }
    
    public var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Live Hardware Status Bar
                topStatusBar
                
                Divider()
                    .overlay(StudioTheme.border)
                
                // Messages Scroll View
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 16) {
                            if messages.isEmpty {
                                emptyStateView
                            } else {
                                ForEach(messages) { msg in
                                    messageBubble(msg)
                                        .transition(reduceMotion ? .opacity : .opacity.combined(with: .move(edge: .bottom)))
                                }
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                    }
                    .onChange(of: messages.last?.id) { _, _ in
                        if let last = messages.last {
                            withAnimation(reduceMotion ? nil : .easeOut(duration: 0.25)) {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                    .onChange(of: messages.last?.text) { _, _ in
                        if let last = messages.last {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                
                // Audio Dictation HUD (if active)
                if speechManager.isRecording {
                    recordingHUD
                }
                
                // Floating Input Bar
                inputBar
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    modelDropdownMenu
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { messages.removeAll() }) {
                        StudioIcon(.trash2)
                            .frame(width: 15, height: 15)
                            .foregroundStyle(messages.isEmpty ? Color.secondary.opacity(0.3) : StudioTheme.ember)
                    }
                    .disabled(messages.isEmpty || isGenerating)
                    .accessibilityLabel("Clear conversation")
                    .studioPress()
                }
            }
            .background(StudioTheme.canvas)
            .sheet(isPresented: $showVoiceOrb) {
                VoiceOrbView(activeModelName: $activeModelName)
            }
            .sheet(isPresented: $showCameraVision) {
                LiveCameraVisionView { recognized in
                    self.inputPrompt = recognized
                }
            }
        }
    }
    
    // MARK: - Navigation Bar Model Selector
    
    private var modelDropdownMenu: some View {
        Menu {
            Section("Intelligence Tier") {
                ForEach(IntelligenceTier.allCases) { tier in
                    Button(action: { switchTier(tier) }) {
                        HStack {
                            Text(tier.shortLabel)
                            if tier == selectedTier {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            }
            
            Section("Execution Engine") {
                Button(action: toggleEngine) {
                    Label(selectedEngine == .metalGPU ? "Metal GPU (Active)" : "Switch to Metal GPU", systemImage: "bolt.fill")
                }
                Button(action: toggleEngine) {
                    Label(selectedEngine == .neonCPU ? "ARM NEON CPU (Active)" : "Switch to ARM NEON CPU", systemImage: "cpu.fill")
                }
            }
            
            Section("Persona Style") {
                ForEach(personas, id: \.self) { persona in
                    Button(action: { selectedPersona = persona }) {
                        HStack {
                            Text(persona)
                            if persona == selectedPersona {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            }
            
            Section("Knowledge Vault") {
                Button(action: { isRAGEnabled.toggle() }) {
                    Label(isRAGEnabled ? "Private Vault RAG: ON" : "Private Vault RAG: OFF", systemImage: isRAGEnabled ? "books.vertical.fill" : "books.vertical")
                }
            }
        } label: {
            HStack(spacing: 6) {
                Text(selectedTier.shortLabel)
                    .font(StudioTheme.body(.subheadline, weight: .semibold))
                    .foregroundStyle(.primary)
                
                StudioIcon(.chevronDown)
                    .frame(width: 10, height: 10)
                    .foregroundStyle(.secondary)
            }
        }
    }
    
    // MARK: - Top Status Bar
    
    private var topStatusBar: some View {
        HStack {
            HStack(spacing: 5) {
                StudioIcon(.activity)
                    .frame(width: 11, height: 11)
                    .foregroundStyle(StudioTheme.phosphor)
                
                Text(selectedEngine == .metalGPU ? "Metal GPU" : "NEON CPU")
                    .font(StudioTheme.body(.caption, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
            
            if isGenerating {
                HStack(spacing: 4) {
                    Text(String(format: "%.1f tok/s", currentTokPerSec))
                        .font(StudioTheme.body(.caption2, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
            } else {
                HStack(spacing: 4) {
                    StudioIcon(.cpu)
                        .frame(width: 10, height: 10)
                        .foregroundStyle(StudioTheme.titanium)
                    Text(String(format: "%.0f MB RAM", physicalFootprintMB > 0 ? physicalFootprintMB : selectedTier.memoryTierMB))
                        .font(StudioTheme.body(.caption2, weight: .bold))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(StudioTheme.surface)
    }
    
    // MARK: - Empty State
    
    private var emptyStateView: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("New conversation")
                .font(StudioTheme.heading(.title2, weight: .semibold))
                .foregroundStyle(.primary)
                .padding(.bottom, 8)
            Text("Prompts and responses stay on this device.")
                .font(StudioTheme.body(.body))
                .foregroundStyle(.secondary)
                .padding(.bottom, 38)

            Text("Try asking")
                .font(StudioTheme.heading(.headline))
                .padding(.bottom, 12)
            starterPrompt("Explain something", prompt: "Explain how on-device LLM inference works")
            Divider()
            starterPrompt("Write some code", prompt: "Write a thread-safe Swift actor with debounce")
            Divider()
            starterPrompt("Work through a problem", prompt: "Help me solve a problem step by step")
        }
        .frame(maxWidth: 620, alignment: .leading)
        .padding(.horizontal, 24)
        .padding(.top, 36)
        .padding(.bottom, 24)
    }

    private func starterPrompt(_ title: String, prompt: String) -> some View {
        Button {
            inputPrompt = prompt
        } label: {
            HStack(spacing: 12) {
                Text(title)
                    .font(StudioTheme.body(.body))
                    .foregroundStyle(.primary)
                Spacer()
                Image(systemName: "arrow.up.left")
                    .font(StudioTheme.body(.footnote))
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 17)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityHint("Fills the message field")
    }

    // MARK: - Message Bubble
    
    private func messageBubble(_ msg: ChatMessage) -> some View {
        HStack(alignment: .top, spacing: 10) {
            if !msg.isUser {
                ZStack {
                    Circle()
                        .fill(StudioTheme.ember.opacity(0.14))
                        .frame(width: 30, height: 30)
                        .overlay(
                            Circle().stroke(StudioTheme.ember.opacity(0.3), lineWidth: 1)
                        )
                    StudioIcon(.cpu)
                        .frame(width: 14, height: 14)
                        .foregroundStyle(StudioTheme.ember)
                }
            } else {
                Spacer(minLength: 40)
            }
            
            VStack(alignment: msg.isUser ? .trailing : .leading, spacing: 6) {
                // Tool Call Execution Card (if a local tool was invoked)
                if let tool = msg.toolResult {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 6) {
                            StudioIcon(.wrench)
                                .frame(width: 12, height: 12)
                                .foregroundStyle(StudioTheme.ember)
                            Text("Tool Executed: \(tool.toolName)")
                                .font(StudioTheme.body(.caption, weight: .semibold))
                                .foregroundStyle(StudioTheme.ember)
                            Spacer()
                            Text(String(format: "%.1f ms", tool.executionTimeMs))
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                        
                        Text(tool.output)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.primary)
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(StudioTheme.surfaceInput)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .padding(10)
                    .studioCard()
                }
                
                // RAG Source Citations
                if !msg.ragCitations.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(msg.ragCitations, id: \.self) { cit in
                                HStack(spacing: 4) {
                                    StudioIcon(.bookOpen)
                                        .frame(width: 10, height: 10)
                                        .foregroundStyle(StudioTheme.ember)
                                    Text(cit)
                                        .font(StudioTheme.body(.caption2, weight: .bold))
                                }
                                .padding(.horizontal, 9)
                                .padding(.vertical, 4)
                                .background(StudioTheme.ember.opacity(0.12))
                                .foregroundStyle(StudioTheme.ember)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                }
                
                if !msg.text.isEmpty {
                    Text(LocalizedStringKey(msg.text))
                        .font(StudioTheme.body(.subheadline))
                        .lineSpacing(3)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            msg.isUser ?
                            AnyShapeStyle(Color(red: 0.16, green: 0.20, blue: 0.30)) :
                            AnyShapeStyle(StudioTheme.surfaceRaised)
                        )
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .stroke(msg.isUser ? Color.white.opacity(0.10) : StudioTheme.border, lineWidth: 1)
                        )
                }
                
                // Assistant Actions Toolbar
                if !msg.isUser && !msg.text.isEmpty {
                    HStack(spacing: 8) {
                        if msg.tokPerSec > 0 {
                            HStack(spacing: 4) {
                                StudioIcon(.zap)
                                    .frame(width: 10, height: 10)
                                    .foregroundStyle(StudioTheme.phosphor)
                                Text(String(format: "%.1f tok/s • %@", msg.tokPerSec, msg.engineUsed))
                                    .font(StudioTheme.body(.caption2, weight: .bold))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        
                        Spacer()
                        
                        // Copy Button
                        Button(action: {
                            #if canImport(UIKit)
                            UIPasteboard.general.string = msg.text
                            #endif
                            copiedMessageId = msg.id
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                if copiedMessageId == msg.id { copiedMessageId = nil }
                            }
                        }) {
                            HStack(spacing: 3) {
                                if copiedMessageId == msg.id {
                                    StudioIcon(.check)
                                        .frame(width: 11, height: 11)
                                        .foregroundStyle(StudioTheme.phosphor)
                                } else {
                                    StudioIcon(.copy)
                                        .frame(width: 11, height: 11)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(6)
                            .background(StudioTheme.surface)
                            .clipShape(Circle())
                            .overlay(Circle().stroke(StudioTheme.border, lineWidth: 0.8))
                        }
                        .accessibilityLabel(copiedMessageId == msg.id ? "Copied" : "Copy response")
                        .studioPress()
                        
                        // TTS Speaker Button
                        Button(action: {
                            speechManager.toggleSpeech(for: msg.text)
                        }) {
                            StudioIcon(.volume2)
                                .frame(width: 11, height: 11)
                                .foregroundStyle(speechManager.isSpeaking ? StudioTheme.ember : .secondary)
                                .padding(6)
                                .background(StudioTheme.surface)
                                .clipShape(Circle())
                                .overlay(Circle().stroke(StudioTheme.border, lineWidth: 0.8))
                        }
                        .accessibilityLabel(speechManager.isSpeaking ? "Stop reading response" : "Read response aloud")
                        .studioPress()
                    }
                    .padding(.horizontal, 4)
                }
            }
            
            if msg.isUser {
                ZStack {
                    Circle()
                        .fill(Color(red: 0.16, green: 0.20, blue: 0.30))
                        .frame(width: 30, height: 30)
                        .overlay(Circle().stroke(Color.white.opacity(0.12), lineWidth: 1))
                    StudioIcon(.user)
                        .frame(width: 14, height: 14)
                        .foregroundStyle(.white)
                }
            } else {
                Spacer(minLength: 40)
            }
        }
    }
    
    // MARK: - Voice Recording HUD
    
    private var recordingHUD: some View {
        HStack(spacing: 8) {
            StudioIcon(.mic)
                .frame(width: 14, height: 14)
                .foregroundStyle(.red)
            Text("Listening... speak your prompt")
                .font(StudioTheme.body(.caption, weight: .semibold))
                .foregroundStyle(.red)
            Spacer()
            Button("Done") {
                speechManager.stopRecording()
            }
            .font(StudioTheme.body(.caption, weight: .semibold))
            .foregroundStyle(StudioTheme.ember)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.red.opacity(0.10))
    }
    
    // MARK: - Floating Input Composer
    
    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            // Plus Action Hub (Camera, Voice, Vault)
            Menu {
                Button(action: { showCameraVision = true }) {
                    Label("Scan with Camera OCR", systemImage: "camera")
                }
                Button(action: { showVoiceOrb = true }) {
                    Label("Voice Duplex Mode", systemImage: "waveform")
                }
                Button(action: { isRAGEnabled.toggle() }) {
                    Label(isRAGEnabled ? "Private Vault RAG: Enabled" : "Private Vault RAG: Disabled", systemImage: "books.vertical")
                }
                Divider()
                Button(role: .destructive, action: { messages.removeAll() }) {
                    Label("Clear Chat", systemImage: "trash")
                }
            } label: {
                StudioIcon(.plus)
                    .frame(width: 16, height: 16)
                    .foregroundStyle(.secondary)
                    .padding(10)
                    .background(StudioTheme.surfaceRaised)
                    .clipShape(Circle())
                    .overlay(Circle().stroke(StudioTheme.border, lineWidth: 0.8))
            }
            .accessibilityLabel("More actions")
            .studioPress()
            
            // Text Input Field Container
            HStack(alignment: .bottom, spacing: 8) {
                TextField("Ask anything...", text: $inputPrompt, axis: .vertical)
                    .lineLimit(1...5)
                    .font(StudioTheme.body(.subheadline))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .disabled(isGenerating)
                
                // Mic dictation button
                Button(action: toggleVoiceDictation) {
                    StudioIcon(.mic)
                        .frame(width: 15, height: 15)
                        .foregroundStyle(speechManager.isRecording ? .red : .secondary)
                        .padding(8)
                }
                .accessibilityLabel(speechManager.isRecording ? "Stop dictation" : "Dictate prompt")
                .studioPress()
            }
            .background(StudioTheme.surfaceInput)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(StudioTheme.border, lineWidth: 1)
            )
            
            // Send / Cancel Action Button
            if isGenerating {
                Button(action: cancelGeneration) {
                    StudioIcon(.square)
                        .frame(width: 14, height: 14)
                        .foregroundStyle(.white)
                        .padding(10)
                        .background(Color.red)
                        .clipShape(Circle())
                }
                .accessibilityLabel("Stop generating")
                .studioPress()
            } else {
                let hasText = !inputPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                Button(action: sendMessage) {
                    StudioIcon(.arrowUp)
                        .frame(width: 15, height: 15)
                        .foregroundStyle(hasText ? .white : Color.secondary.opacity(0.4))
                        .padding(10)
                        .background(hasText ? StudioTheme.ember : StudioTheme.surfaceRaised)
                        .clipShape(Circle())
                }
                .disabled(!hasText)
                .accessibilityLabel("Send message")
                .studioPress()
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(StudioTheme.surface)
    }
    
    // MARK: - Actions
    
    private func sendMessage() {
        let userText = inputPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !userText.isEmpty else { return }
        
        inputPrompt = ""
        speechManager.stopSpeaking()
        speechManager.stopRecording()
        
        let engineLabel = (selectedEngine == .metalGPU) ? "Metal GPU" : "ARM NEON CPU"
        let assistantMsg = ChatMessage(isUser: false, text: "", isStreaming: true, engineUsed: engineLabel)
        withAnimation(reduceMotion ? nil : .easeOut(duration: 0.25)) {
            messages.append(ChatMessage(isUser: true, text: userText))
            messages.append(assistantMsg)
        }
        let assistantIndex = messages.count - 1
        
        isGenerating = true
        currentTokPerSec = 0.0
        
        // 0. Knowledge Vault RAG Search (if enabled)
        var ragCitations: [String] = []
        var queryForBrain = userText
        if isRAGEnabled {
            let hits = LocalRAGStore.shared.search(query: userText, topK: 2)
            if !hits.isEmpty {
                ragCitations = hits.map { "\($0.documentTitle) [Chunk \($0.chunkIndex)]" }
                let context = hits.map { "• \($0.snippet)" }.joined(separator: "\n")
                queryForBrain = "[RAG CONTEXT]\n\(context)\n[USER QUESTION]\n\(userText)"
            }
        }
        
        // 1. Prepare prompt with RAG context if active
        let effectivePrompt = isRAGEnabled && !ragCitations.isEmpty ? queryForBrain : userText
        
        // 2. Stream tokens generated directly by the on-device LLM
        let onTok: (String, Double) -> Void = { token, tokPerSec in
            guard self.messages.indices.contains(assistantIndex) else { return }
            self.messages[assistantIndex].text += token
            self.messages[assistantIndex].tokPerSec = tokPerSec
            self.currentTokPerSec = tokPerSec
        }
        
        let onDone: (String, Double, Double, Double) -> Void = { fullGeneratedText, totalTime, avgTokPerSec, ttftMs in
            self.isGenerating = false
            guard self.messages.indices.contains(assistantIndex) else { return }
            let rawText = fullGeneratedText.isEmpty ? self.messages[assistantIndex].text : fullGeneratedText
            
            // Check if model invoked any on-device tool
            let (cleanedResponse, toolResult) = DeviceToolRegistry.shared.parseAndExecuteToolCall(in: rawText)
            
            self.messages[assistantIndex].text = cleanedResponse.isEmpty ? rawText : cleanedResponse
            self.messages[assistantIndex].toolResult = toolResult
            self.messages[assistantIndex].ragCitations = ragCitations
            self.messages[assistantIndex].tokPerSec = avgTokPerSec
            self.messages[assistantIndex].isStreaming = false
        }
        
        NanoEdgeBridge.sharedInstance().executionEngine = selectedEngine
        NanoEdgeBridge.sharedInstance().generateStreaming(
            withPrompt: effectivePrompt,
            systemPrompt: selectedPersona,
            maxTokens: 500,
            temperature: 0.7,
            onToken: onTok,
            onComplete: onDone
        )
    }
    
    private func cancelGeneration() {
        NanoEdgeBridge.sharedInstance().cancelGeneration()
        isGenerating = false
        if let last = messages.indices.last, !messages[last].isUser {
            messages[last].isStreaming = false
        }
    }
    
    private func toggleEngine() {
        if selectedEngine == .metalGPU {
            selectedEngine = .neonCPU
        } else {
            selectedEngine = .metalGPU
        }
        NanoEdgeBridge.sharedInstance().executionEngine = selectedEngine
    }
    
    private func toggleVoiceDictation() {
        speechManager.toggleRecording { transcribed in
            self.inputPrompt = transcribed
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
    
    private func switchTier(_ tier: IntelligenceTier) {
        switch tier {
        case .fast1B:
            if let m = discoveredModels.first(where: { $0.name.contains("1b") }) {
                switchModel(m)
            }
        case .deep3B:
            if let m = discoveredModels.first(where: { $0.name.contains("3b") }) {
                switchModel(m)
            }
        }
    }
}
