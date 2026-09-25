import SwiftUI

public struct FeaturesView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Binding var selectedTab: Int
    @Binding var activeModelName: String
    @Binding var discoveredModels: [ModelFileItem]
    @Binding var physicalFootprintMB: Double

    var onOpenVoiceOrb: () -> Void
    var onOpenLiveCamera: () -> Void

    public init(
        selectedTab: Binding<Int>,
        activeModelName: Binding<String>,
        discoveredModels: Binding<[ModelFileItem]>,
        physicalFootprintMB: Binding<Double>,
        onOpenVoiceOrb: @escaping () -> Void,
        onOpenLiveCamera: @escaping () -> Void
    ) {
        self._selectedTab = selectedTab
        self._activeModelName = activeModelName
        self._discoveredModels = discoveredModels
        self._physicalFootprintMB = physicalFootprintMB
        self.onOpenVoiceOrb = onOpenVoiceOrb
        self.onOpenLiveCamera = onOpenLiveCamera
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 36) {
                introduction
                notebookButton
                quickActions
                modelStatus
            }
            .frame(maxWidth: 680, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 24)
            .padding(.top, 32)
            .padding(.bottom, 48)
        }
        .background(StudioTheme.canvas)
    }

    private var introduction: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Studio")
                .font(StudioTheme.heading(.largeTitle, weight: .bold))
                .tracking(-1.5)
                .foregroundStyle(.primary)
            Text(activeModelName == "None"
                 ? "Models and tools that run on this device. Add a model to begin."
                 : "Running \(activeModelName) on this device.")
                .font(StudioTheme.body(.body))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var notebookButton: some View {
        Button {
            selectedTab = activeModelName == "None" ? 3 : 1
        } label: {
            HStack {
                Label(activeModelName == "None" ? "Browse models" : "Open Notebook", systemImage: "square.and.pencil")
                    .font(StudioTheme.body(.body, weight: .semibold))
                Spacer()
                Image(systemName: "arrow.right")
                    .font(StudioTheme.body(.body, weight: .medium))
                    .foregroundStyle(StudioTheme.ember)
            }
            .foregroundStyle(.primary)
            .padding(.vertical, 18)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .overlay(alignment: .top) { Rectangle().fill(StudioTheme.border).frame(height: 1) }
        .overlay(alignment: .bottom) { Rectangle().fill(StudioTheme.border).frame(height: 1) }
    }

    private var quickActions: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Quick actions")
                .font(StudioTheme.heading(.headline))
            VStack(spacing: 0) {
                quickAction("Speak", symbol: "waveform", action: onOpenVoiceOrb)
                quickAction("Use camera", symbol: "camera", action: onOpenLiveCamera)
            }
        }
    }

    private func quickAction(_ title: String, symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: symbol)
                    .font(StudioTheme.body(.body))
                    .foregroundStyle(StudioTheme.ember)
                    .frame(width: 24)
                Text(title)
                    .font(StudioTheme.body(.body))
                    .foregroundStyle(.primary)
                Spacer()
                Image(systemName: "arrow.up.right")
                    .font(StudioTheme.body(.footnote))
                    .foregroundStyle(.secondary)
            }
            .frame(minHeight: 48)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var modelStatus: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("On this device")
                    .font(StudioTheme.heading(.headline))
                Spacer()
                Button("Manage") { selectedTab = 3 }
                    .font(StudioTheme.body(.subheadline))
            }
            .padding(.bottom, 18)

            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "cpu")
                    .font(StudioTheme.body(.body))
                    .foregroundStyle(.secondary)
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 5) {
                    Text(activeModelName == "None" ? "No model loaded" : activeModelName)
                        .font(StudioTheme.body(.body, weight: .medium))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                    Text("\(discoveredModels.count) models available locally")
                        .font(StudioTheme.body(.subheadline))
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, 16)
            .overlay(alignment: .top) { Rectangle().fill(StudioTheme.border).frame(height: 1) }

            if physicalFootprintMB > 0 {
                Text("Memory in use  \(physicalFootprintMB, specifier: "%.0f") MB")
                    .font(StudioTheme.body(.footnote).monospacedDigit())
                    .foregroundStyle(.secondary)
                    .contentTransition(.numericText())
                    .animation(reduceMotion ? nil : .easeOut(duration: 0.2), value: Int(physicalFootprintMB))
                    .padding(.top, 8)
            }
        }
    }
}
