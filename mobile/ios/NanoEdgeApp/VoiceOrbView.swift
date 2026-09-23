import SwiftUI
import Combine

public enum VoiceChatState: String {
    case idle = "Ready to Talk"
    case listening = "Listening..."
    case thinking = "Reasoning on A18 Pro..."
    case speaking = "Speaking..."
}

public struct VoiceOrbView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var speechManager = SpeechManager.shared
    
    @Binding var activeModelName: String
    @State private var chatState: VoiceChatState = .idle
    @State private var userSpokenText: String = ""
    @State private var modelSpokenText: String = ""
    @State private var orbRotation: Double = 0.0
    @State private var pulseScale: CGFloat = 1.0
    
    public init(activeModelName: Binding<String>) {
        self._activeModelName = activeModelName
    }
    
    public var body: some View {
        ZStack {
            // Dark futuristic backdrop
            Color.black.ignoresSafeArea()
            
            // Ambient background glow
            RadialGradient(
                colors: [stateGlowColor.opacity(0.35), Color.clear],
                center: .center,
                startRadius: 40,
                endRadius: 280
            )
            .ignoresSafeArea()
            
            VStack(spacing: 24) {
                // Header Bar
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Live Voice Mode")
                            .font(.headline)
                            .fontWeight(.bold)
                            .foregroundStyle(.white)
                        HStack(spacing: 4) {
                            Circle().fill(Color.green).frame(width: 6, height: 6)
                            Text("Apple Silicon Duplex • \(activeModelName)")
                                .font(.caption2)
                                .foregroundStyle(.gray)
                        }
                    }
                    Spacer()
                    Button(action: {
                        speechManager.stopSpeaking()
                        speechManager.stopRecording()
                        dismiss()
                    }) {
                        StudioIcon(.x)
                            .frame(width: 18, height: 18)
                            .foregroundStyle(.white.opacity(0.7))
                            .padding(8)
                            .background(Color.white.opacity(0.12))
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal, 24)
                .padding(.top, 16)
                
                Spacer()
                
                // Animated Voice Orb
                ZStack {
                    // Outer Ripple 1
                    Circle()
                        .stroke(stateGlowColor.opacity(0.25), lineWidth: 2)
                        .frame(width: 220, height: 220)
                        .scaleEffect(1.0 + CGFloat(speechManager.currentAudioLevel) * 0.5)
                        .animation(.easeOut(duration: 0.15), value: speechManager.currentAudioLevel)
                    
                    // Outer Ripple 2
                    Circle()
                        .stroke(stateGlowColor.opacity(0.15), lineWidth: 1.5)
                        .frame(width: 260, height: 260)
                        .scaleEffect(1.0 + CGFloat(speechManager.currentAudioLevel) * 0.7)
                        .animation(.easeOut(duration: 0.2), value: speechManager.currentAudioLevel)
                    
                    // Core Fluid Glowing Orb
                    Circle()
                        .fill(
                            AngularGradient(
                                gradient: Gradient(colors: [
                                    Color.cyan,
                                    Color.blue,
                                    Color.indigo,
                                    Color.purple,
                                    Color.pink,
                                    Color.cyan
                                ]),
                                center: .center,
                                angle: .degrees(orbRotation)
                            )
                        )
                        .frame(width: 160, height: 160)
                        .blur(radius: 6)
                        .scaleEffect(1.0 + CGFloat(speechManager.currentAudioLevel) * 0.35)
                        .shadow(color: stateGlowColor.opacity(0.8), radius: 35, x: 0, y: 0)
                        .overlay(
                            Circle()
                                .stroke(Color.white.opacity(0.4), lineWidth: 1.5)
                        )
                        .onAppear {
                            withAnimation(.linear(duration: 8.0).repeatForever(autoreverses: false)) {
                                orbRotation = 360.0
                            }
                        }
                    
                    // Center specular highlight
                    Circle()
                        .fill(Color.white.opacity(0.25))
                        .frame(width: 60, height: 60)
                        .blur(radius: 8)
                }
                .frame(height: 280)
                
                // State Label & Waveform Status
                VStack(spacing: 8) {
                    Text(chatState.rawValue)
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                    
                    if chatState == .listening {
                        Text("Speak naturally. Will answer automatically on silence.")
                            .font(.caption)
                            .foregroundStyle(.gray)
                    } else if chatState == .speaking {
                        Text("Tap anywhere to interrupt.")
                            .font(.caption)
                            .foregroundStyle(.cyan)
                    }
                }
                
                // Live Conversation Cards
                VStack(spacing: 12) {
                    if !userSpokenText.isEmpty {
                        HStack {
                            Text("You: \(userSpokenText)")
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.9))
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(Color.white.opacity(0.12))
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                            Spacer()
                        }
                    }
                    
                    if !modelSpokenText.isEmpty {
                        HStack {
                            Spacer()
                            Text(modelSpokenText)
                                .font(.subheadline)
                                .foregroundStyle(.cyan)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(Color.cyan.opacity(0.15))
                                .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                    }
                }
                .padding(.horizontal, 24)
                .frame(minHeight: 90)
                
                Spacer()
                
                // Bottom Interactive Controls
                HStack(spacing: 32) {
                    // Interrupt / Stop Speaking Button
                    Button(action: {
                        speechManager.stopSpeaking()
                        chatState = .idle
                    }) {
                        StudioIcon(.square)
                            .frame(width: 18, height: 18)
                            .foregroundStyle(.white)
                            .frame(width: 56, height: 56)
                            .background(Color.red.opacity(0.8))
                            .clipShape(Circle())
                    }
                    .disabled(chatState != .speaking)
                    .opacity(chatState == .speaking ? 1.0 : 0.3)
                    
                    // Main Mic / Turn Toggle Button
                    Button(action: {
                        handleMicTap()
                    }) {
                        StudioIcon(chatState == .listening ? .audioWaveform : .mic)
                            .frame(width: 28, height: 28)
                            .foregroundStyle(.white)
                            .frame(width: 76, height: 76)
                            .background(chatState == .listening ? StudioTheme.phosphor : StudioTheme.ember)
                            .clipShape(Circle())
                            .shadow(color: (chatState == .listening ? StudioTheme.phosphor : StudioTheme.ember).opacity(0.5), radius: 15)
                    }
                    
                    // Reset / Clear Button
                    Button(action: {
                        speechManager.stopSpeaking()
                        speechManager.stopRecording()
                        userSpokenText = ""
                        modelSpokenText = ""
                        chatState = .idle
                    }) {
                        StudioIcon(.rotateCw)
                            .frame(width: 18, height: 18)
                            .foregroundStyle(.white)
                            .frame(width: 56, height: 56)
                            .background(Color.white.opacity(0.15))
                            .clipShape(Circle())
                    }
                }
                .padding(.bottom, 32)
            }
        }
        .onAppear {
            setupVoiceCallbacks()
            // Auto start listening on open
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                startListening()
            }
        }
        .onDisappear {
            speechManager.stopSpeaking()
            speechManager.stopRecording()
        }
    }
    
    private var stateGlowColor: Color {
        switch chatState {
        case .listening: return .green
        case .thinking: return .purple
        case .speaking: return .cyan
        case .idle: return .blue
        }
    }
    
    private func setupVoiceCallbacks() {
        speechManager.onSilenceDetected = { finalSpoken in
            guard !finalSpoken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
            self.userSpokenText = finalSpoken
            self.processSpokenQuery(finalSpoken)
        }
        
        speechManager.onSpeechFinished = {
            if self.chatState == .speaking {
                // Auto resume listening for seamless duplex hands-free conversation
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    self.startListening()
                }
            }
        }
    }
    
    private func handleMicTap() {
        if chatState == .listening {
            speechManager.stopRecording()
            if !userSpokenText.isEmpty {
                processSpokenQuery(userSpokenText)
            } else {
                chatState = .idle
            }
        } else if chatState == .speaking {
            speechManager.stopSpeaking()
            chatState = .idle
        } else {
            startListening()
        }
    }
    
    private func startListening() {
        speechManager.stopSpeaking()
        chatState = .listening
        userSpokenText = ""
        speechManager.startRecording { partial in
            self.userSpokenText = partial
        }
    }
    
    private func processSpokenQuery(_ query: String) {
        chatState = .thinking
        modelSpokenText = ""
        
        NanoEdgeBridge.sharedInstance().generateStreaming(
            withPrompt: query,
            systemPrompt: "You are a concise voice assistant running locally on iPhone Apple Silicon. Answer naturally in 1 to 2 clear sentences.",
            maxTokens: 100,
            temperature: 0.6,
            onToken: { token, _ in
                DispatchQueue.main.async {
                    if self.chatState == .thinking {
                        self.chatState = .speaking
                    }
                    self.modelSpokenText += token
                }
            },
            onComplete: { fullText, _, _, _ in
                DispatchQueue.main.async {
                    let finalText = fullText.isEmpty ? self.modelSpokenText : fullText
                    self.modelSpokenText = finalText
                    self.chatState = .speaking
                    self.speechManager.speak(text: finalText)
                }
            }
        )
    }
}
