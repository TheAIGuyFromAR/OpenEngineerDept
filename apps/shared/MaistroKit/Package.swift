// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MaistroKit",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "MaistroProtocol", targets: ["MaistroProtocol"]),
        .library(name: "MaistroKit", targets: ["MaistroKit"]),
        .library(name: "MaistroChatUI", targets: ["MaistroChatUI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/steipete/ElevenLabsKit", exact: "0.1.0"),
        .package(url: "https://github.com/gonzalezreal/textual", exact: "0.3.1"),
    ],
    targets: [
        .target(
            name: "MaistroProtocol",
            path: "Sources/MaistroProtocol",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .target(
            name: "MaistroKit",
            dependencies: [
                "MaistroProtocol",
                .product(name: "ElevenLabsKit", package: "ElevenLabsKit"),
            ],
            path: "Sources/MaistroKit",
            resources: [
                .process("Resources"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .target(
            name: "MaistroChatUI",
            dependencies: [
                "MaistroKit",
                .product(
                    name: "Textual",
                    package: "textual",
                    condition: .when(platforms: [.macOS, .iOS])),
            ],
            path: "Sources/MaistroChatUI",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .testTarget(
            name: "MaistroKitTests",
            dependencies: ["MaistroKit", "MaistroChatUI"],
            path: "Tests/MaistroKitTests",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
                .enableExperimentalFeature("SwiftTesting"),
            ]),
    ])
