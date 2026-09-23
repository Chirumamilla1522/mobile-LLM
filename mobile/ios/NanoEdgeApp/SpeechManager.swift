import Foundation
import Speech
import AVFoundation

@MainActor
public class SpeechManager: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    public static let shared = SpeechManager()
    
    @Published public var isRecording: Bool = false
    @Published public var isSpeaking: Bool = false
    @Published public var recognizedText: String = ""
    @Published public var errorMessage: String? = nil
    @Published public var currentAudioLevel: Float = 0.0
    
    public var onSilenceDetected: ((String) -> Void)? = nil
    public var onSpeechFinished: (() -> Void)? = nil
    private var silenceTimer: Timer? = nil
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()
    
    public override init() {
        super.init()
        synthesizer.delegate = self
    }
    
    // MARK: - Speech Recognition (Dictation)
    
    public func startRecording(onTranscript: @escaping (String) -> Void) {
        // Stop any current audio
        stopSpeaking()
        stopRecording()
        
        recognizedText = ""
        errorMessage = nil
        currentAudioLevel = 0.0
        
        SFSpeechRecognizer.requestAuthorization { authStatus in
            Task { @MainActor in
                switch authStatus {
                case .authorized:
                    self.beginAudioEngineRecording(onTranscript: onTranscript)
                case .denied, .restricted:
                    self.errorMessage = "Speech recognition permission was denied."
                case .notDetermined:
                    self.errorMessage = "Speech recognition authorization pending."
                @unknown default:
                    break
                }
            }
        }
    }
    
    private func beginAudioEngineRecording(onTranscript: @escaping (String) -> Void) {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let request = recognitionRequest else { return }
            
            // Guarantee 100% on-device offline recognition
            if speechRecognizer?.supportsOnDeviceRecognition == true {
                request.requiresOnDeviceRecognition = true
            }
            request.shouldReportPartialResults = true
            
            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            
            inputNode.removeTap(onBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
                request.append(buffer)
                
                // Real-time audio RMS metering for Voice Orb
                guard let self = self, let channelData = buffer.floatChannelData?[0] else { return }
                let frames = Int(buffer.frameLength)
                if frames > 0 {
                    var sum: Float = 0.0
                    for i in 0..<frames {
                        sum += channelData[i] * channelData[i]
                    }
                    let rms = sqrt(sum / Float(frames))
                    let normalized = min(1.0, max(0.0, rms * 6.0))
                    Task { @MainActor in
                        self.currentAudioLevel = normalized
                    }
                }
            }
            
            audioEngine.prepare()
            try audioEngine.start()
            isRecording = true
            
            recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
                Task { @MainActor in
                    guard let self = self else { return }
                    if let result = result {
                        let text = result.bestTranscription.formattedString
                        self.recognizedText = text
                        onTranscript(text)
                        
                        // Reset VAD silence timer
                        self.silenceTimer?.invalidate()
                        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            self.silenceTimer = Timer.scheduledTimer(withTimeInterval: 1.4, repeats: false) { [weak self] _ in
                                guard let strongSelf = self, strongSelf.isRecording else { return }
                                strongSelf.stopRecording()
                                strongSelf.onSilenceDetected?(strongSelf.recognizedText)
                            }
                        }
                    }
                    if error != nil || result?.isFinal == true {
                        self.stopRecording()
                    }
                }
            }
        } catch {
            errorMessage = "Microphone error: \(error.localizedDescription)"
            stopRecording()
        }
    }
    
    public func stopRecording() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        isRecording = false
    }
    
    public func toggleRecording(onTranscript: @escaping (String) -> Void) {
        if isRecording {
            stopRecording()
        } else {
            startRecording(onTranscript: onTranscript)
        }
    }
    
    // MARK: - Text to Speech (TTS)
    
    public func speak(text: String) {
        stopSpeaking()
        stopRecording()
        
        let cleaned = cleanMarkdownForSpeech(text)
        guard !cleaned.isEmpty else { return }
        
        let utterance = AVSpeechUtterance(string: cleaned)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        
        if let voice = AVSpeechSynthesisVoice(language: "en-US") {
            utterance.voice = voice
        }
        
        isSpeaking = true
        synthesizer.speak(utterance)
    }
    
    public func stopSpeaking() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
    }
    
    public func toggleSpeech(for text: String) {
        if isSpeaking {
            stopSpeaking()
        } else {
            speak(text: text)
        }
    }
    
    // AVSpeechSynthesizerDelegate
    nonisolated public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
            self.onSpeechFinished?()
        }
    }
    
    nonisolated public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
            self.onSpeechFinished?()
        }
    }
    
    private func cleanMarkdownForSpeech(_ text: String) -> String {
        var res = text
        // Remove code blocks
        res = res.replacingOccurrences(of: "```[a-zA-Z]*\n[\\s\\S]*?\n```", with: "Code snippet omitted.", options: .regularExpression)
        // Remove markdown headers
        res = res.replacingOccurrences(of: "#{1,6}\\s+", with: "", options: .regularExpression)
        // Remove bold/italics
        res = res.replacingOccurrences(of: "\\*\\*", with: "")
        res = res.replacingOccurrences(of: "\\*", with: "")
        res = res.replacingOccurrences(of: "`", with: "")
        // Remove LaTeX math
        res = res.replacingOccurrences(of: "\\$\\$[\\s\\S]*?\\$\\$", with: "mathematical expression", options: .regularExpression)
        res = res.replacingOccurrences(of: "\\$[\\s\\S]*?\\$", with: "formula", options: .regularExpression)
        return res.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
