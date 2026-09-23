// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NanoEdgeMobile",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "NanoEdgeBridge",
            targets: ["NanoEdgeBridge"]
        ),
    ],
    targets: [
        .target(
            name: "NanoEdgeBridge",
            path: "Sources/NanoEdgeBridge",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("core"),
                .headerSearchPath("core/mllm"),
                .unsafeFlags(["-std=c++20", "-fobjc-arc"])
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation")
            ]
        ),
    ],
    cxxLanguageStandard: .cxx20
)
