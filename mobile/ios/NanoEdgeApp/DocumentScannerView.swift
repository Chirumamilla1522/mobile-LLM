import SwiftUI
import PhotosUI
import Vision
import UIKit

public struct DocumentScannerView: View {
    @Environment(\.dismiss) private var dismiss
    public let onTextExtracted: (String) -> Void
    
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var selectedImage: UIImage? = nil
    @State private var isProcessing: Bool = false
    @State private var recognizedText: String = ""
    @State private var statusMessage: String = "Select a document photo or receipt to scan text."
    
    public init(onTextExtracted: @escaping (String) -> Void) {
        self.onTextExtracted = onTextExtracted
    }
    
    public var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // Image Preview Area
                if let img = selectedImage {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 220)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(StudioTheme.border, lineWidth: 1)
                        )
                } else {
                    VStack(alignment: .leading, spacing: 12) {
                        Image(systemName: "doc.viewfinder")
                            .font(StudioTheme.heading(.title2))
                            .foregroundStyle(StudioTheme.ember)
                        Text("Scan a document")
                            .font(StudioTheme.heading(.title2))
                        Text("Choose a photo to extract text without uploading it.")
                            .font(StudioTheme.body(.body))
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 180, alignment: .leading)
                    .padding(20)
                    .studioCard()
                }
                
                // Photo Picker Button
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    HStack(spacing: 6) {
                        Image(systemName: "photo.on.rectangle.angled")
                        Text(selectedImage == nil ? "Choose Document Photo" : "Choose Another Photo")
                    }
                    .font(StudioTheme.body(.subheadline))
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(StudioTheme.ember)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .onChange(of: selectedItem) { _, newItem in
                    Task {
                        if let data = try? await newItem?.loadTransferable(type: Data.self),
                           let uiImg = UIImage(data: data) {
                            selectedImage = uiImg
                            recognizeText(from: uiImg)
                        }
                    }
                }
                
                // Recognized Text Preview
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Extracted Text")
                            .font(StudioTheme.body(.caption))
                            .fontWeight(.bold)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if isProcessing {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: StudioTheme.ember))
                                .scaleEffect(0.6)
                        }
                    }
                    
                    TextEditor(text: $recognizedText)
                        .font(StudioTheme.body(.footnote))
                        .frame(minHeight: 120)
                        .padding(6)
                        .background(StudioTheme.canvas)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color(.systemGray4), lineWidth: 0.5)
                        )
                }
                .padding(12)
                .background(StudioTheme.surface)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                
                Spacer()
                
                // Action: Use Extracted Text
                Button(action: {
                    let cleaned = recognizedText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !cleaned.isEmpty {
                        onTextExtracted(cleaned)
                        dismiss()
                    }
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.down.doc.fill")
                        Text("Insert Extracted Text")
                    }
                    .font(StudioTheme.heading(.headline))
                    .fontWeight(.bold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? StudioTheme.surfaceRaised : StudioTheme.ember)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .disabled(recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding()
            .navigationTitle("Scan document")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
    
    // MARK: - Vision OCR Processing
    
    private func recognizeText(from image: UIImage) {
        guard let cgImage = image.cgImage else { return }
        
        isProcessing = true
        statusMessage = "Analyzing document with Neural Vision..."
        recognizedText = ""
        
        DispatchQueue.global(qos: .userInitiated).async {
            let request = VNRecognizeTextRequest { req, err in
                DispatchQueue.main.async {
                    self.isProcessing = false
                    if let err = err {
                        self.statusMessage = "OCR error: \(err.localizedDescription)"
                        return
                    }
                    guard let observations = req.results as? [VNRecognizedTextObservation] else {
                        self.statusMessage = "No text detected."
                        return
                    }
                    
                    var lines: [String] = []
                    for obs in observations {
                        if let topCandidate = obs.topCandidates(1).first {
                            lines.append(topCandidate.string)
                        }
                    }
                    
                    self.recognizedText = lines.joined(separator: "\n")
                    self.statusMessage = "Successfully recognized \(lines.count) lines."
                }
            }
            
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            
            let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    self.statusMessage = "Scan failed: \(error.localizedDescription)"
                }
            }
        }
    }
}
