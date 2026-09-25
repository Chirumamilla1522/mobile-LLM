import SwiftUI
import Combine
import UniformTypeIdentifiers

// Model descriptor for UI
public struct ModelFileItem: Identifiable, Hashable {
    public var id: String { path }
    public let name: String
    public let path: String
    public let sizeMB: Double
    public let scaleLabel: String
    public let quantLabel: String
    public let isBundled: Bool
}

// Comparison history record
public struct BenchmarkRecord: Identifiable {
    public let id = UUID()
    public let modelName: String
    public let quantType: String
    public let dimensions: String
    public let weightMB: Double
    public let gpuLatencyUs: Double
    public let cpuLatencyUs: Double
    public let bandwidthGBs: Double
    public let tokensPerSec: Double
    public let speedup: Double
}

public struct ContentView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @State private var deviceName: String = "Detecting..."
    @State private var hasUnifiedMemory: Bool = true
    @State private var physicalFootprintMB: Double = 0.0
    @State private var residentRAMMB: Double = 0.0
    
    @State private var activeModelName: String = "None"
    @State private var activeModelPath: String = ""
    @State private var modelInfo: String = "No model active"
    @State private var isModelLoaded: Bool = false
    
    // Model Suite & Benchmark History
    @State private var discoveredModels: [ModelFileItem] = []
    @State private var benchmarkHistory: [BenchmarkRecord] = []
    
    @State private var selectedTab: Int = 0
    @State private var selectedTier: IntelligenceTier = .fast1B
    @State private var showGlobalVoiceOrb: Bool = false
    @State private var showGlobalCamera: Bool = false
    
    let timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    
    public init() {}

    public var body: some View {
        Group {
            if horizontalSizeClass == .regular {
                regularLayout
            } else {
                compactLayout
            }
        }
        .tint(StudioTheme.ember)
        .environment(\.font, StudioTheme.body(.body))
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showGlobalVoiceOrb) {
            VoiceOrbView(activeModelName: $activeModelName)
        }
        .sheet(isPresented: $showGlobalCamera) {
            LiveCameraVisionView { _ in selectedTab = 1 }
        }
        .onAppear {
            refreshDeviceInfo()
            refreshMemoryStats()
            scanDiscoveredModels()
        }
        .onReceive(timer) { _ in refreshMemoryStats() }
        .onOpenURL { handleDeepLink($0) }
    }

    private var compactLayout: some View {
        VStack(spacing: 0) {
            TabView(selection: $selectedTab) {
                destination(0).tabItem { Label("Studio", systemImage: "square.grid.2x2") }.tag(0)
                    .toolbar(.hidden, for: .tabBar)
                destination(1).tabItem { Label("Notebook", systemImage: "square.and.pencil") }.tag(1)
                    .toolbar(.hidden, for: .tabBar)
                destination(2).tabItem { Label("Instruments", systemImage: "slider.horizontal.3") }.tag(2)
                    .toolbar(.hidden, for: .tabBar)
                destination(3).tabItem { Label("Telemetry", systemImage: "chart.xyaxis.line") }.tag(3)
                    .toolbar(.hidden, for: .tabBar)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            HStack(spacing: 0) {
                tabButton("Studio", symbol: "square.grid.2x2", tab: 0)
                tabButton("Notebook", symbol: "square.and.pencil", tab: 1)
                tabButton("Instruments", symbol: "slider.horizontal.3", tab: 2)
                tabButton("Telemetry", symbol: "chart.xyaxis.line", tab: 3)
            }
            .background(StudioTheme.canvas)
            .overlay(alignment: .top) { StudioTheme.border.frame(height: 1) }
        }
        .background(StudioTheme.canvas.ignoresSafeArea())
    }

    private func tabButton(_ title: String, symbol: String, tab: Int) -> some View {
        Button { selectedTab = tab } label: {
            VStack(spacing: 5) {
                Image(systemName: symbol)
                    .font(.system(size: 19, weight: .regular))
                Text(title)
                    .font(StudioTheme.body(.caption2, weight: selectedTab == tab ? .semibold : .regular))
            }
            .foregroundStyle(selectedTab == tab ? StudioTheme.ember : StudioTheme.titanium)
            .frame(maxWidth: .infinity, minHeight: 62)
            .contentShape(Rectangle())
            .overlay(alignment: .top) {
                (selectedTab == tab ? StudioTheme.ember : Color.clear)
                    .frame(height: 2)
                    .padding(.horizontal, 16)
            }
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selectedTab == tab ? .isSelected : [])
    }

    private var regularLayout: some View {
        NavigationSplitView {
            List {
                sidebarButton("Studio", systemImage: "slider.horizontal.3", tab: 0)
                sidebarButton("Notebook", systemImage: "square.and.pencil", tab: 1)
                sidebarButton("Instruments", systemImage: "wrench.and.screwdriver", tab: 2)
                sidebarButton("Telemetry", systemImage: "chart.xyaxis.line", tab: 3)
            }
            .navigationTitle("NanoEdge")
            .listStyle(.sidebar)
        } detail: {
            destination(selectedTab)
                .frame(maxWidth: 900, alignment: .leading)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .background(StudioTheme.canvas)
        }
    }

    private func sidebarButton(_ title: String, systemImage: String, tab: Int) -> some View {
        Button { selectedTab = tab } label: {
            Label(title, systemImage: systemImage)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
        }
        .foregroundStyle(selectedTab == tab ? StudioTheme.ember : .primary)
        .accessibilityAddTraits(selectedTab == tab ? .isSelected : [])
    }

    @ViewBuilder
    private func destination(_ tab: Int) -> some View {
        switch tab {
        case 1:
            ChatView(
                activeModelName: $activeModelName,
                discoveredModels: $discoveredModels,
                physicalFootprintMB: $physicalFootprintMB,
                selectedTier: $selectedTier
            )
        case 2:
            AppsView(
                activeModelName: $activeModelName,
                discoveredModels: $discoveredModels,
                physicalFootprintMB: $physicalFootprintMB
            )
        case 3:
            BenchmarkView(
                deviceName: $deviceName,
                hasUnifiedMemory: $hasUnifiedMemory,
                physicalFootprintMB: $physicalFootprintMB,
                residentRAMMB: $residentRAMMB,
                activeModelName: $activeModelName,
                activeModelPath: $activeModelPath,
                modelInfo: $modelInfo,
                isModelLoaded: $isModelLoaded,
                discoveredModels: $discoveredModels,
                benchmarkHistory: $benchmarkHistory
            )
        default:
            FeaturesView(
                selectedTab: $selectedTab,
                activeModelName: $activeModelName,
                discoveredModels: $discoveredModels,
                physicalFootprintMB: $physicalFootprintMB,
                onOpenVoiceOrb: { showGlobalVoiceOrb = true },
                onOpenLiveCamera: { showGlobalCamera = true }
            )
        }
    }
    
    private func handleDeepLink(_ url: URL) {
        guard url.scheme?.lowercased() == "nanoedge" else { return }
        switch url.host?.lowercased() {
        case "studio":
            selectedTab = 0
        case "voice":
            showGlobalVoiceOrb = true
        case "camera":
            showGlobalCamera = true
        case "chat":
            selectedTab = 1
        case "apps":
            selectedTab = 2
        case "benchmark":
            selectedTab = 3
        default:
            break
        }
    }
    
    // MARK: - Discovery & Hardware Diagnostics
    
    private func refreshDeviceInfo() {
        let bridge = NanoEdgeBridge.sharedInstance()
        deviceName = bridge.deviceName()
        hasUnifiedMemory = bridge.hasUnifiedMemory()
    }
    
    private func refreshMemoryStats() {
        let stats = NanoEdgeBridge.sharedInstance().queryMemoryFootprint()
        physicalFootprintMB = stats.physicalFootprintMB
        residentRAMMB = stats.residentSizeMB
    }
    
    private func scanDiscoveredModels() {
        var items: [ModelFileItem] = []
        
        // 1. Scan App Bundle
        let bundlePaths = Bundle.main.paths(forResourcesOfType: "mllm", inDirectory: nil)
        for path in bundlePaths {
            items.append(createModelItem(fromPath: path, isBundled: true))
        }
        
        // Scan imported models and legacy Documents models.
        for directory in [FileManager.SearchPathDirectory.applicationSupportDirectory, .documentDirectory] {
            if let root = FileManager.default.urls(for: directory, in: .userDomainMask).first,
               let files = try? FileManager.default.contentsOfDirectory(
                    at: directory == .applicationSupportDirectory ? root.appendingPathComponent("Models") : root,
                    includingPropertiesForKeys: [.fileSizeKey]
               ) {
                for fileURL in files where fileURL.pathExtension.lowercased() == "mllm" {
                    items.append(createModelItem(fromPath: fileURL.path, isBundled: false))
                }
            }
        }
        
        // Deduplicate
        var seen = Set<String>()
        var unique: [ModelFileItem] = []
        for item in items {
            if !seen.contains(item.name) {
                seen.insert(item.name)
                unique.append(item)
            }
        }
        
        self.discoveredModels = unique.sorted { a, b in
            if a.sizeMB != b.sizeMB {
                return a.sizeMB < b.sizeMB
            }
            return a.name < b.name
        }
        
        // Auto-load preferred model if not loaded
        if !isModelLoaded {
            if let preferred = discoveredModels.first(where: { $0.name.contains("llama3_2_1b") }) ??
                               discoveredModels.first(where: { $0.name.contains("llama3_2_3b") }) ??
                               discoveredModels.first(where: { $0.name.contains("smollm2") }) ??
                               discoveredModels.first {
                loadModelItem(preferred)
            }
        }
    }
    
    private func createModelItem(fromPath path: String, isBundled: Bool) -> ModelFileItem {
        let name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        var sizeMB = 0.0
        if let attrs = try? FileManager.default.attributesOfItem(atPath: path),
           let size = attrs[.size] as? Int64 {
            sizeMB = Double(size) / (1024.0 * 1024.0)
        }
        
        var scale = "Custom"
        if name.contains("qwen2.5_0.5b") { scale = "Qwen 0.5B" }
        else if name.contains("qwen2.5_1.5b") { scale = "Qwen 1.5B" }
        else if name.contains("qwen2.5_3b") { scale = "Qwen 3B" }
        else if name.contains("llama3.2_1b") { scale = "Llama 1B" }
        else if name.contains("llama3.2_3b") { scale = "Llama 3B" }
        else if name.contains("llama3.1_8b") { scale = "Llama 8B" }
        else if name.contains("mistral_7b") { scale = "Mistral 7B" }
        else if name.contains("gemma2_2b") { scale = "Gemma 2B" }
        else if name.contains("nano") { scale = "0.5B Nano" }
        else if name.contains("small") || name.contains("mobile_demo") { scale = "1.5B Small" }
        else if name.contains("medium") { scale = "3B Medium" }
        else if name.contains("large") || name.contains("llama_dim") { scale = "7B Large" }
        
        var quant = "Q4_0"
        if name.contains("mq4") { quant = "MQ4_Apple" }
        else if name.contains("int8") || name.contains("q8") { quant = "INT8" }
        else if name.contains("q4") { quant = "Q4_0" }
        
        return ModelFileItem(
            name: name,
            path: path,
            sizeMB: sizeMB,
            scaleLabel: scale,
            quantLabel: quant,
            isBundled: isBundled
        )
    }
    
    private func loadModelItem(_ item: ModelFileItem) {
        do {
            try NanoEdgeBridge.sharedInstance().loadModel(fromPath: item.path)
            isModelLoaded = true
            activeModelName = item.name
            activeModelPath = item.path
            modelInfo = NanoEdgeBridge.sharedInstance().loadedModelInfo()
            refreshMemoryStats()
        } catch {
            print("Failed loading model: \(error.localizedDescription)")
        }
    }
    
}
