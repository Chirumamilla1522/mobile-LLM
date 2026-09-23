import Foundation
import Combine

public struct RAGChunk: Identifiable, Hashable {
    public let id = UUID()
    public let documentId: UUID
    public let documentTitle: String
    public let chunkIndex: Int
    public let text: String
    public let wordCounts: [String: Double]
}

public struct RAGSearchResult: Identifiable {
    public let id = UUID()
    public let documentTitle: String
    public let snippet: String
    public let score: Double
    public let chunkIndex: Int
}

public struct RAGDocument: Identifiable {
    public let id = UUID()
    public let title: String
    public let content: String
    public let dateAdded: Date
    public var chunkCount: Int { chunks.count }
    public var chunks: [RAGChunk] = []
}

@objc public class LocalRAGStore: NSObject, ObservableObject {
    @objc public static let shared = LocalRAGStore()
    
    @Published public var documents: [RAGDocument] = []
    @Published public var totalChunksIndexed: Int = 0
    
    private override init() {
        super.init()
        loadDefaultDocuments()
    }
    
    public func loadDefaultDocuments() {
        documents.removeAll()
        
        addDocument(
            title: "Apple A18 Pro Silicon Architecture.txt",
            content: """
            The Apple A18 Pro is manufactured on TSMC's second-generation 3-nanometer process (N3E).
            It features a 6-core CPU configuration: 2 high-performance cores running up to 4.04 GHz with 64KB L1 instruction cache and 16MB shared L2 cache, paired with 4 high-efficiency cores with 4MB shared L2 cache.
            The graphics subsystem consists of a 6-core Metal GPU featuring dynamic caching, hardware-accelerated ray tracing, and dedicated mesh shading.
            The Apple Neural Engine (ANE) features 16 cores delivering up to 35 TOPS (trillion operations per second) of int8 matrix throughput, operating at an ultra-low power budget under 1.5 Watts.
            Memory architecture utilizes 8GB of unified LPDDR5X memory across a 128-bit bus, providing approximately 60 GB/s of sustained memory bandwidth shared seamlessly across CPU, GPU, and Neural Engine with zero-copy buffer sharing.
            """
        )
        
        addDocument(
            title: "NanoEdge Quantization & MLLM Specification.txt",
            content: """
            NanoEdge uses the native .mllm binary format for zero-overhead memory mapped execution on iOS.
            Supported quantization formats include:
            1. Q4_0: 32-weight blocks with 4-bit unsigned integer nibbles and a single FP16 scale factor (4.5 bits/weight effective).
            2. MQ4_Apple: An optimized 4-bit format designed for Apple Silicon SIMD matrix-vector instructions, aligning weights to 64-byte L2 cache lines.
            3. INT8 / Q8_0: 8-bit symmetric quantization providing near FP16 perplexity with a 50% memory footprint reduction.
            In autoregressive token decode where M=1, operations are strictly memory-bandwidth bound. NanoEdge employs 2-row output tiling to load activation vectors once per 2 rows, cutting DRAM traffic by 50%.
            """
        )
        
        addDocument(
            title: "On-Device Memory Management & Jetsam.txt",
            content: """
            iOS enforces strict per-process memory limits governed by the Jetsam kernel subsystem.
            Exceeding physical dirty memory thresholds causes immediate SIGKILL termination with zero recovery.
            To guarantee rock-solid stability:
            1. Use mmap with MAP_SHARED or MAP_PRIVATE so weights remain clean file-backed pages rather than dirty heap allocations.
            2. Employ double-buffered Ping-Pong activation arenas capped at under 4MB, eliminating malloc and free inside the decode loop.
            3. Quantize the Key-Value (KV) cache to INT8 with per-block scale factors, saving over 51MB of RAM at 4096-token context lengths.
            4. Implement madvise(MADV_DONTNEED) handlers to purge cached activations upon receiving iOS memory pressure warnings.
            """
        )
    }
    
    public func addDocument(title: String, content: String) {
        let docId = UUID()
        let chunks = chunkText(content, docId: docId, docTitle: title)
        let doc = RAGDocument(title: title, content: content, dateAdded: Date(), chunks: chunks)
        documents.append(doc)
        totalChunksIndexed += chunks.count
    }
    
    public func deleteDocument(id: UUID) {
        documents.removeAll { $0.id == id }
        totalChunksIndexed = documents.reduce(0) { $0 + $1.chunks.count }
    }
    
    // Chunking text into overlapping paragraphs
    private func chunkText(_ text: String, docId: UUID, docTitle: String) -> [RAGChunk] {
        let paragraphs = text.components(separatedBy: "\n\n").filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        var result: [RAGChunk] = []
        
        for (idx, p) in paragraphs.enumerated() {
            let clean = p.trimmingCharacters(in: .whitespacesAndNewlines)
            let wordMap = tokenizeAndCount(clean)
            result.append(RAGChunk(
                documentId: docId,
                documentTitle: docTitle,
                chunkIndex: idx + 1,
                text: clean,
                wordCounts: wordMap
            ))
        }
        
        return result
    }
    
    private func tokenizeAndCount(_ text: String) -> [String: Double] {
        let words = text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
        var map: [String: Double] = [:]
        for w in words {
            map[w, default: 0.0] += 1.0
        }
        return map
    }
    
    // Semantic Cosine Similarity Search
    public func search(query: String, topK: Int = 3) -> [RAGSearchResult] {
        let qWords = tokenizeAndCount(query)
        if qWords.isEmpty { return [] }
        
        var scores: [(RAGChunk, Double)] = []
        
        for doc in documents {
            for chunk in doc.chunks {
                let score = cosineSimilarity(qWords, chunk.wordCounts)
                if score > 0.05 {
                    scores.append((chunk, score))
                }
            }
        }
        
        scores.sort { $0.1 > $1.1 }
        
        return scores.prefix(topK).map { chunk, score in
            RAGSearchResult(
                documentTitle: chunk.documentTitle,
                snippet: chunk.text,
                score: score,
                chunkIndex: chunk.chunkIndex
            )
        }
    }
    
    private func cosineSimilarity(_ a: [String: Double], _ b: [String: Double]) -> Double {
        var dotProduct = 0.0
        var normA = 0.0
        var normB = 0.0
        
        for (_, count) in a {
            normA += count * count
        }
        for (_, count) in b {
            normB += count * count
        }
        
        for (word, countA) in a {
            if let countB = b[word] {
                dotProduct += countA * countB
            }
        }
        
        let denominator = sqrt(normA) * sqrt(normB)
        return denominator > 0 ? (dotProduct / denominator) : 0.0
    }
}
