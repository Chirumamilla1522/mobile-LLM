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
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @ObservedObject private var speechManager = SpeechManager.shared
    
    @Binding var activeModelName: String
    @State private var chatState: VoiceChatState = .idle
    @State private var userSpokenText: String = ""
    @State private var modelSpokenText: String = ""
    
    public init(activeModelName: Binding<String>) {
        self._activeModelName = activeModelName
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("VOICE / ON DEVICE")
                        .font(StudioTheme.body(.caption, weight: .bold))
                        .tracking(2)
                        .foregroundStyle(StudioTheme.ember)
                    Text(activeModelName)
                        .font(StudioTheme.body(.subheadline))
                        .foregroundStyle(StudioTheme.titanium)
                        .lineLimit(1)
                }
                Spacer()
                Button {
                    speechManager.stopSpeaking()
                    speechManager.stopRecording()
                    dismiss()
                } label: {
                    Image(systemName: "xmark")
                        .font(StudioTheme.body(.body, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .background(StudioTheme.surface)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(StudioTheme.border))
                }
                .accessibilityLabel("Close voice mode")
            }

            Spacer(minLength: 60)

            Text(chatState.rawValue)
                .font(StudioTheme.heading(.largeTitle, weight: .bold))
                .tracking(-1)
                .contentTransition(.opacity)
            Text(chatState == .listening ? "Speak naturally. Recording stops when you pause." :
                 chatState == .speaking ? "Tap the microphone to interrupt." :
                 "Your conversation stays on this device.")
                .font(StudioTheme.body(.body))
                .foregroundStyle(StudioTheme.titanium)
                .padding(.top, 10)

            HStack(alignment: .center, spacing: 5) {
                ForEach(0..<17, id: \.self) { index in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(chatState == .listening ? StudioTheme.ember : StudioTheme.borderActive)
                        .frame(maxWidth: .infinity)
                        .frame(height: 12 + CGFloat((index * 7) % 5) * 8 + CGFloat(speechManager.currentAudioLevel) * 45)
                }
            }
            .frame(height: 110)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.12), value: speechManager.currentAudioLevel)
            .padding(.vertical, 32)
            .accessibilityLabel(chatState == .listening ? "Microphone level" : "Voice activity")

            VStack(alignment: .leading, spacing: 16) {
                if !userSpokenText.isEmpty {
                    transcript("YOU", text: userSpokenText)
                }
                if !modelSpokenText.isEmpty {
                    transcript("NANOEDGE", text: modelSpokenText)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 110, alignment: .topLeading)

            Spacer(minLength: 32)

            HStack(spacing: 10) {
                Button(action: handleMicTap) {
                    Label(chatState == .listening ? "Finish speaking" : "Start speaking",
                          systemImage: chatState == .listening ? "stop.fill" : "mic.fill")
                        .font(StudioTheme.body(.body, weight: .semibold))
                        .frame(maxWidth: .infinity, minHeight: 54)
                        .foregroundStyle(.black)
                        .background(StudioTheme.ember)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                Button {
                    speechManager.stopSpeaking()
                    speechManager.stopRecording()
                    userSpokenText = ""
                    modelSpokenText = ""
                    chatState = .idle
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                        .frame(width: 54, height: 54)
                        .background(StudioTheme.surface)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(StudioTheme.border))
                }
                .accessibilityLabel("Reset conversation")
                Button {
                    speechManager.stopSpeaking()
                    chatState = .idle
                } label: {
                    Image(systemName: "speaker.slash")
                        .frame(width: 54, height: 54)
                        .background(StudioTheme.surface)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(StudioTheme.border))
                }
                .disabled(chatState != .speaking)
                .accessibilityLabel("Stop speaking")
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(StudioTheme.canvas.ignoresSafeArea())
        .tint(StudioTheme.ember)
        .onAppear {
            setupVoiceCallbacks()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { startListening() }
        }
        .onDisappear {
            speechManager.stopSpeaking()
            speechManager.stopRecording()
        }
    }

    private func transcript(_ speaker: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(speaker)
                .font(StudioTheme.body(.caption2, weight: .bold))
                .tracking(1.5)
                .foregroundStyle(StudioTheme.ember)
            Text(text)
                .font(StudioTheme.body(.body))
                .foregroundStyle(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.top, 12)
        .overlay(alignment: .top) { StudioTheme.border.frame(height: 1) }
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
