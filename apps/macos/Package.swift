// swift-tools-version: 6.2
// Package manifest for the Maistro macOS companion (menu bar app + IPC library).

import PackageDescription

let package = Package(
    name: "Maistro",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "MaistroIPC", targets: ["MaistroIPC"]),
        .library(name: "MaistroDiscovery", targets: ["MaistroDiscovery"]),
        .executable(name: "Maistro", targets: ["Maistro"]),
        .executable(name: "maistro-mac", targets: ["MaistroMacCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/orchetect/MenuBarExtraAccess", exact: "1.2.2"),
        .package(url: "https://github.com/swiftlang/swift-subprocess.git", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.8.0"),
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.8.1"),
        .package(url: "https://github.com/steipete/Peekaboo.git", branch: "main"),
        .package(path: "../shared/MaistroKit"),
        .package(path: "../../Swabble"),
    ],
    targets: [
        .target(
            name: "MaistroIPC",
            dependencies: [],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .target(
            name: "MaistroDiscovery",
            dependencies: [
                .product(name: "MaistroKit", package: "MaistroKit"),
            ],
            path: "Sources/MaistroDiscovery",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .executableTarget(
            name: "Maistro",
            dependencies: [
                "MaistroIPC",
                "MaistroDiscovery",
                .product(name: "MaistroKit", package: "MaistroKit"),
                .product(name: "MaistroChatUI", package: "MaistroKit"),
                .product(name: "MaistroProtocol", package: "MaistroKit"),
                .product(name: "SwabbleKit", package: "swabble"),
                .product(name: "MenuBarExtraAccess", package: "MenuBarExtraAccess"),
                .product(name: "Subprocess", package: "swift-subprocess"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Sparkle", package: "Sparkle"),
                .product(name: "PeekabooBridge", package: "Peekaboo"),
                .product(name: "PeekabooAutomationKit", package: "Peekaboo"),
            ],
            exclude: [
                "Resources/Info.plist",
            ],
            resources: [
                .copy("Resources/Maistro.icns"),
                .copy("Resources/DeviceModels"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .executableTarget(
            name: "MaistroMacCLI",
            dependencies: [
                "MaistroDiscovery",
                .product(name: "MaistroKit", package: "MaistroKit"),
                .product(name: "MaistroProtocol", package: "MaistroKit"),
            ],
            path: "Sources/MaistroMacCLI",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .testTarget(
            name: "MaistroIPCTests",
            dependencies: [
                "MaistroIPC",
                "Maistro",
                "MaistroDiscovery",
                .product(name: "MaistroProtocol", package: "MaistroKit"),
                .product(name: "SwabbleKit", package: "swabble"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
                .enableExperimentalFeature("SwiftTesting"),
            ]),
    ])
