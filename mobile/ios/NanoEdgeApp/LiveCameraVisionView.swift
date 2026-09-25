import SwiftUI
import AVFoundation
import Vision
import UIKit

public enum VisionAnalysisMode: String, CaseIterable, Identifiable {
    case mathSolver = "Math & Formula"
    case codeDebugger = "Code on Screen"
    case receiptParser = "Receipt / Invoice"
    case translator = "Real-World Translate"
    
    public var id: String { rawValue }
    
    public var symbol: StudioSymbol {
        switch self {
        case .mathSolver: return .calculator
        case .codeDebugger: return .code
        case .receiptParser: return .fileText
        case .translator: return .languages
        }
    }
}

public struct LiveCameraVisionView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    public let onAnalyzeComplete: ((String) -> Void)?
    
    @State private var selectedMode: VisionAnalysisMode = .mathSolver
    @State private var recognizedLines: [String] = []
    @State private var isProcessingAnalysis: Bool = false
    @State private var aiAnalysisOutput: String = ""
    @State private var scanLineOffset: CGFloat = -140
    @State private var hasCameraPermission: Bool = false
    
    public init(onAnalyzeComplete: ((String) -> Void)? = nil) {
        self.onAnalyzeComplete = onAnalyzeComplete
    }
    
    public var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                // Camera Viewfinder Feed
                CameraPreviewRepresentable { lines in
                    self.recognizedLines = lines
                }
                .ignoresSafeArea()
                
                // Camera controls and live text
                VStack(spacing: 0) {
                    // Mode Selector Bar
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(VisionAnalysisMode.allCases) { mode in
                                Button(action: { selectedMode = mode }) {
                                    HStack(spacing: 6) {
                                        StudioIcon(mode.symbol)
                                            .frame(width: 12, height: 12)
                                        Text(mode.rawValue)
                                    }
                                    .font(StudioTheme.body(.caption, weight: .semibold))
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(selectedMode == mode ? StudioTheme.ember : Color.black.opacity(0.6))
                                    .foregroundStyle(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.2), lineWidth: 0.5)
                                    )
                                }
                            }
                        }
                        .padding(.horizontal, 16)
                    }
                    .padding(.top, 12)
                    
                    Spacer()
                    
                    // Center Targeting Reticle
                    ZStack {
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(StudioTheme.ember, lineWidth: 1)
                            .frame(maxWidth: .infinity, maxHeight: 240)
                            .padding(.horizontal, 24)
                        
                        // Animated Scanning Line
                        Rectangle()
                            .fill(StudioTheme.ember)
                            .frame(height: 1)
                            .offset(y: scanLineOffset)
                            .onAppear {
                                if !reduceMotion {
                                    withAnimation(.easeInOut(duration: 2.0).repeatForever(autoreverses: true)) {
                                        scanLineOffset = 140
                                    }
                                }
                            }
                    }
                    
                    Spacer()
                    
                    // Live Extracted Text & AI Analysis Drawer
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            HStack(spacing: 5) {
                                StudioIcon(.scan)
                                    .frame(width: 12, height: 12)
                                    .foregroundStyle(StudioTheme.phosphor)
                                Text("Live text")
                                    .font(StudioTheme.body(.caption, weight: .semibold))
                                    .foregroundStyle(.primary)
                            }
                            Spacer()
                            Text("\(recognizedLines.count) lines detected")
                                .font(StudioTheme.body(.caption2))
                                .foregroundStyle(.gray)
                        }
                        
                        if !aiAnalysisOutput.isEmpty {
                            ScrollView {
                                Text(aiAnalysisOutput)
                                    .font(StudioTheme.body(.caption))
                                    .foregroundStyle(.white)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .frame(maxHeight: 140)
                        } else {
                            Text(recognizedLines.isEmpty ? "Point camera at math, code, or receipts..." : recognizedLines.prefix(3).joined(separator: " • "))
                                .font(StudioTheme.body(.caption))
                                .foregroundStyle(.white.opacity(0.8))
                                .lineLimit(2)
                        }
                        
                        // Action Trigger Button
                        Button(action: runAIAnalysisOnFrame) {
                            HStack(spacing: 8) {
                                if isProcessingAnalysis {
                                    ProgressView().tint(.white)
                                } else {
                                    StudioIcon(.scan)
                                        .frame(width: 14, height: 14)
                                }
                                Text(isProcessingAnalysis ? "Analyzing on Neural Engine..." : "Analyze Current View")
                            }
                            .font(StudioTheme.body(.footnote, weight: .semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(StudioTheme.ember)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .disabled(isProcessingAnalysis || recognizedLines.isEmpty)
                    }
                    .padding(16)
                    .background(Color.black.opacity(0.85))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.white.opacity(0.15), lineWidth: 1)
                    )
                    .padding(.horizontal, 16)
                    .padding(.bottom, 20)
                }
            }
            .navigationTitle("Live Camera Vision")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { dismiss() }) {
                        StudioIcon(.x)
                            .frame(width: 16, height: 16)
                            .foregroundStyle(.white.opacity(0.8))
                            .padding(6)
                            .background(Color.white.opacity(0.12))
                            .clipShape(Circle())
                    }
                }
            }
        }
    }
    
    private func runAIAnalysisOnFrame() {
        let extracted = recognizedLines.joined(separator: "\n")
        guard !extracted.isEmpty else { return }
        
        isProcessingAnalysis = true
        
        var prompt = ""
        switch selectedMode {
        case .mathSolver:
            prompt = "Translate this handwritten math formula into clean LaTeX and provide the step-by-step mathematical solution:\n\(extracted)"
        case .codeDebugger:
            prompt = "Inspect this code detected from a screen. Identify any syntax bugs, logic errors, and provide the corrected code:\n\(extracted)"
        case .receiptParser:
            prompt = "Extract merchant name, date, subtotal, tax, and total expense from this receipt text into structured JSON:\n\(extracted)"
        case .translator:
            prompt = "Translate the following real-world text into clear English and explain context:\n\(extracted)"
        }
        
        NanoEdgeBridge.sharedInstance().generateStreaming(
            withPrompt: prompt,
            systemPrompt: "You are an intelligent on-device Vision AI assistant running on Apple Silicon. Analyze the extracted image text accurately and concisely.",
            maxTokens: 250,
            temperature: 0.5,
            onToken: { token, _ in
                DispatchQueue.main.async {
                    self.aiAnalysisOutput += token
                }
            },
            onComplete: { fullText, _, _, _ in
                DispatchQueue.main.async {
                    let finalText = fullText.isEmpty ? self.aiAnalysisOutput : fullText
                    self.aiAnalysisOutput = finalText
                    self.isProcessingAnalysis = false
                    self.onAnalyzeComplete?(finalText)
                }
            }
        )
    }
}

// Camera Feed Representable using AVFoundation
struct CameraPreviewRepresentable: UIViewControllerRepresentable {
    let onTextRecognized: ([String]) -> Void
    
    func makeUIViewController(context: Context) -> CameraFeedViewController {
        let vc = CameraFeedViewController()
        vc.onTextRecognized = onTextRecognized
        return vc
    }
    
    func updateUIViewController(_ uiViewController: CameraFeedViewController, context: Context) {}
}

class CameraFeedViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var onTextRecognized: (([String]) -> Void)?
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var lastFrameTime: CFAbsoluteTime = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCaptureSession()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }
    
    private func setupCaptureSession() {
        captureSession.sessionPreset = .high
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            return
        }
        
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        let queue = DispatchQueue(label: "camera.vision.queue")
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.videoGravity = .resizeAspectFill
        if let pl = previewLayer {
            view.layer.addSublayer(pl)
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Throttle to 1.5 FPS to conserve silicon battery & thermal headroom
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastFrameTime > 0.65 else { return }
        lastFrameTime = now
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request = VNRecognizeTextRequest { [weak self] req, err in
            guard let observations = req.results as? [VNRecognizedTextObservation], err == nil else { return }
            let lines = observations.compactMap { $0.topCandidates(1).first?.string }
            DispatchQueue.main.async {
                self?.onTextRecognized?(lines)
            }
        }
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        try? handler.perform([request])
    }
}
