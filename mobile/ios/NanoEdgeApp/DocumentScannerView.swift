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
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                        )
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "doc.viewfinder.fill")
                            .font(.largeTitle)
                            .foregroundStyle(.blue.gradient)
                        Text("On-Device Vision OCR")
                            .font(.headline)
                            .fontWeight(.bold)
                        Text("Extract text from receipts, documents, or whiteboards with 100% offline neural recognition.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }
                    .frame(maxWidth: .infinity, minHeight: 180)
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                
                // Photo Picker Button
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    HStack(spacing: 6) {
                        Image(systemName: "photo.on.rectangle.angled")
                        Text(selectedImage == nil ? "Choose Document Photo" : "Choose Another Photo")
                    }
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(Color.blue)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
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
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundStyle(.secondary)
                        Spacer()
                        if isProcessing {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .blue))
                                .scaleEffect(0.6)
                        }
                    }
                    
                    TextEditor(text: $recognizedText)
                        .font(.footnote)
                        .frame(minHeight: 120)
                        .padding(6)
                        .background(Color(.systemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color(.systemGray4), lineWidth: 0.5)
                        )
                }
                .padding(12)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 14))
                
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
                    .font(.headline)
                    .fontWeight(.bold)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.gray.opacity(0.4) : Color.green)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                }
                .disabled(recognizedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding()
            .navigationTitle("Scan Document (OCR)")
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
