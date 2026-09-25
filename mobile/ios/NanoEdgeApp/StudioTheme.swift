import SwiftUI

public enum StudioTheme {
    public static let canvas = Color(red: 0.067, green: 0.067, blue: 0.059)
    public static let surface = Color(red: 0.105, green: 0.105, blue: 0.094)
    public static let surfaceRaised = Color(red: 0.15, green: 0.15, blue: 0.133)
    public static let surfaceInput = Color(red: 0.085, green: 0.085, blue: 0.075)
    public static let border = Color.white.opacity(0.14)
    public static let borderActive = Color.white.opacity(0.32)
    public static let ember = Color(red: 0.96, green: 0.37, blue: 0.16)
    public static let titanium = Color(red: 0.68, green: 0.67, blue: 0.63)
    public static let phosphor = Color(red: 0.58, green: 0.73, blue: 0.53) // Status only

    public static func heading(_ style: Font.TextStyle, weight: Font.Weight = .semibold) -> Font {
        .custom("Sora-Regular", size: pointSize(for: style), relativeTo: style).weight(weight)
    }

    public static func body(_ style: Font.TextStyle, weight: Font.Weight = .regular) -> Font {
        .custom("SourceSans3-Roman", size: pointSize(for: style), relativeTo: style).weight(weight)
    }

    private static func pointSize(for style: Font.TextStyle) -> CGFloat {
        switch style {
        case .largeTitle: return 34
        case .title: return 28
        case .title2: return 22
        case .title3: return 20
        case .headline, .body: return 17
        case .subheadline, .callout: return 16
        case .footnote: return 14
        case .caption: return 13
        case .caption2: return 12
        @unknown default: return 17
        }
    }

}

public enum StudioSymbol {
    case activity, arrowRight, arrowUp, arrowUpRight, audioWaveform, bookOpen
    case brain, braces, calculator, camera, check, chevronDown, code, copy, cpu
    case fileText, languages, layers, messageSquare, mic, play, plus, rotateCw
    case scan, shieldCheck, square, trash2, user, volume2, wand, wrench, x, zap

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
        case .square: return "stop.fill"
        case .trash2: return "trash"
        case .user: return "person.crop.circle"
        case .volume2: return "speaker.wave.2"
        case .wand: return "text.badge.checkmark"
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
    public func body(content: Content) -> some View {
        content
            .background(StudioTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(StudioTheme.border, lineWidth: 1)
            )
    }
}

public struct StudioButtonStyle: ButtonStyle {
    public init() {}
    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(minHeight: 44)
            .contentShape(Rectangle())
            .opacity(configuration.isPressed ? 0.8 : 1.0)
    }
}

public extension View {
    func studioCard() -> some View {
        modifier(StudioCardModifier())
    }
    
    func studioPress() -> some View {
        buttonStyle(StudioButtonStyle())
    }
}
