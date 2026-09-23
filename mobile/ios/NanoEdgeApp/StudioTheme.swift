import SwiftUI

public enum StudioTheme {
    // MARK: - Monolithic Velvet & Obsidian Surfaces
    public static let canvas = Color(red: 0.04, green: 0.04, blue: 0.06)
    public static let surface = Color(red: 0.08, green: 0.09, blue: 0.12)
    public static let surfaceRaised = Color(red: 0.12, green: 0.13, blue: 0.17)
    public static let surfaceInput = Color(red: 0.07, green: 0.08, blue: 0.11)
    
    // MARK: - Precision Borders
    public static let border = Color.white.opacity(0.08)
    public static let borderSubtle = Color.white.opacity(0.04)
    public static let borderActive = Color.white.opacity(0.20)
    
    // MARK: - Semantic Human & Silicon Accents
    public static let ember = Color(red: 0.96, green: 0.42, blue: 0.28)      // Warm human touch & drafting
    public static let titanium = Color(red: 0.65, green: 0.69, blue: 0.76)   // Precision Apple Silicon
    public static let phosphor = Color(red: 0.13, green: 0.77, blue: 0.49)   // Health & live speed (clean emerald)
    public static let cobalt = Color(red: 0.31, green: 0.54, blue: 0.98)     // Deep logic, math & A18 Pro GPU
    public static let amber = Color(red: 0.98, green: 0.66, blue: 0.18)
    public static let purple = Color(red: 0.62, green: 0.45, blue: 0.95)     // Knowledge & vault
    
    // MARK: - Dynamic Gradients
    public static let emberGradient = LinearGradient(
        colors: [ember, Color(red: 1.0, green: 0.55, blue: 0.35)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
    
    public static let cobaltGradient = LinearGradient(
        colors: [cobalt, Color(red: 0.45, green: 0.68, blue: 1.0)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
    
    public static let cardBackground = Color(red: 0.09, green: 0.10, blue: 0.13)
    
    // Typography Styles
    public static func monoCaption(_ text: String) -> Text {
        Text(text)
            .font(.system(.caption, design: .rounded, weight: .semibold))
    }
}

public enum StudioSymbol {
    case activity, arrowRight, arrowUp, arrowUpRight, audioWaveform, bookOpen
    case brain, braces, calculator, camera, check, chevronDown, code, copy, cpu
    case fileText, languages, layers, messageSquare, mic, play, plus, rotateCw
    case scan, shieldCheck, sparkles, square, trash2, user, volume2, wand, wrench, x, zap

    fileprivate var systemName: String {
        switch self {
        case .activity: return "waveform.path.ecg"
        case .arrowRight: return "arrow.right"
        case .arrowUp: return "arrow.up"
        case .arrowUpRight: return "arrow.up.right"
        case .audioWaveform: return "waveform"
        case .bookOpen: return "books.vertical"
        case .brain: return "brain.head.profile"
        case .braces: return "curlybraces"
        case .calculator: return "function"
        case .camera: return "camera"
        case .check: return "checkmark"
        case .chevronDown: return "chevron.down"
        case .code: return "chevron.left.forwardslash.chevron.right"
        case .copy: return "doc.on.doc"
        case .cpu: return "cpu"
        case .fileText: return "doc.text"
        case .languages: return "character.bubble"
        case .layers: return "square.stack.3d.up"
        case .messageSquare: return "bubble.left.and.bubble.right"
        case .mic: return "mic"
        case .play: return "play.fill"
        case .plus: return "plus"
        case .rotateCw: return "arrow.clockwise"
        case .scan: return "viewfinder"
        case .shieldCheck: return "checkmark.shield"
        case .sparkles: return "sparkles"
        case .square: return "stop.fill"
        case .trash2: return "trash"
        case .user: return "person.crop.circle"
        case .volume2: return "speaker.wave.2"
        case .wand: return "wand.and.sparkles"
        case .wrench: return "wrench.and.screwdriver"
        case .x: return "xmark"
        case .zap: return "bolt.fill"
        }
    }
}

public struct StudioIcon: View {
    private let symbol: StudioSymbol

    public init(_ symbol: StudioSymbol) {
        self.symbol = symbol
    }

    public var body: some View {
        Image(systemName: symbol.systemName)
            .resizable()
            .scaledToFit()
            .accessibilityHidden(true)
    }
}

// MARK: - View Modifiers & Styles

public struct StudioCardModifier: ViewModifier {
    var cornerRadius: CGFloat = 18
    var borderColor: Color = StudioTheme.border
    var isRaised: Bool = false
    
    public func body(content: Content) -> some View {
        content
            .background(isRaised ? StudioTheme.surfaceRaised : StudioTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(borderColor, lineWidth: 1)
            )
    }
}

public struct StudioPillModifier: ViewModifier {
    var accentColor: Color = StudioTheme.phosphor
    
    public func body(content: Content) -> some View {
        content
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(accentColor.opacity(0.12))
            .foregroundStyle(accentColor)
            .clipShape(Capsule())
            .overlay(
                Capsule().stroke(accentColor.opacity(0.25), lineWidth: 0.8)
            )
    }
}

public struct BouncyButtonStyle: ButtonStyle {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public init() {}
    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(minHeight: 44)
            .contentShape(Rectangle())
            .scaleEffect(!reduceMotion && configuration.isPressed ? 0.98 : 1.0)
            .opacity(configuration.isPressed ? 0.8 : 1.0)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.16), value: configuration.isPressed)
    }
}

public extension View {
    func studioCard(cornerRadius: CGFloat = 18, borderColor: Color = StudioTheme.border, isRaised: Bool = false) -> some View {
        modifier(StudioCardModifier(cornerRadius: cornerRadius, borderColor: borderColor, isRaised: isRaised))
    }
    
    func studioPill(accent: Color = StudioTheme.phosphor) -> some View {
        modifier(StudioPillModifier(accentColor: accent))
    }
    
    func bouncyPress() -> some View {
        buttonStyle(BouncyButtonStyle())
    }
}
