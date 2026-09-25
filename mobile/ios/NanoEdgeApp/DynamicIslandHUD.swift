import SwiftUI

public struct DynamicIslandHUD: View {
    public let isVisible: Bool
    public let modelName: String
    public let engineName: String
    public let tokensPerSec: Double
    public let progressPct: Double
    public let onStopTap: () -> Void
    
    @State private var isExpanded: Bool = false
    
    public var body: some View {
        if isVisible {
            VStack {
                HStack(spacing: 12) {
                    // Left: Model & Engine Indicator
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 8, height: 8)
                        
                        VStack(alignment: .leading, spacing: 1) {
                            Text(modelName)
                                .font(StudioTheme.body(.caption, weight: .semibold))
                                .foregroundStyle(.white)
                                .lineLimit(1)
                            Text(engineName)
                                .font(StudioTheme.body(.caption2, weight: .bold))
                                .foregroundStyle(.gray)
                        }
                    }
                    
                    Spacer()
                    
                    // Center: Real-time Decode Speed
                    HStack(spacing: 4) {
                        Image(systemName: "bolt.fill")
                            .font(StudioTheme.body(.caption2))
                            .foregroundStyle(.yellow)
                        Text(String(format: "%.1f tok/s", tokensPerSec))
                            .font(StudioTheme.body(.caption, weight: .semibold))
                            .foregroundStyle(.white)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    
                    // Right: Stop / Action Control
                    Button(action: onStopTap) {
                        Image(systemName: "stop.fill")
                            .font(StudioTheme.body(.caption2, weight: .bold))
                            .foregroundStyle(.white)
                            .frame(width: 24, height: 24)
                            .background(Color.red.opacity(0.8))
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.black.opacity(0.92))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.white.opacity(0.2), lineWidth: 1)
                        )
                )
                .padding(.horizontal, 16)
                .padding(.top, 4)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
    }
}
