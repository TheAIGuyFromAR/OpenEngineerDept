import Foundation

public enum MaistroCameraCommand: String, Codable, Sendable {
    case list = "camera.list"
    case snap = "camera.snap"
    case clip = "camera.clip"
}

public enum MaistroCameraFacing: String, Codable, Sendable {
    case back
    case front
}

public enum MaistroCameraImageFormat: String, Codable, Sendable {
    case jpg
    case jpeg
}

public enum MaistroCameraVideoFormat: String, Codable, Sendable {
    case mp4
}

public struct MaistroCameraSnapParams: Codable, Sendable, Equatable {
    public var facing: MaistroCameraFacing?
    public var maxWidth: Int?
    public var quality: Double?
    public var format: MaistroCameraImageFormat?
    public var deviceId: String?
    public var delayMs: Int?

    public init(
        facing: MaistroCameraFacing? = nil,
        maxWidth: Int? = nil,
        quality: Double? = nil,
        format: MaistroCameraImageFormat? = nil,
        deviceId: String? = nil,
        delayMs: Int? = nil)
    {
        self.facing = facing
        self.maxWidth = maxWidth
        self.quality = quality
        self.format = format
        self.deviceId = deviceId
        self.delayMs = delayMs
    }
}

public struct MaistroCameraClipParams: Codable, Sendable, Equatable {
    public var facing: MaistroCameraFacing?
    public var durationMs: Int?
    public var includeAudio: Bool?
    public var format: MaistroCameraVideoFormat?
    public var deviceId: String?

    public init(
        facing: MaistroCameraFacing? = nil,
        durationMs: Int? = nil,
        includeAudio: Bool? = nil,
        format: MaistroCameraVideoFormat? = nil,
        deviceId: String? = nil)
    {
        self.facing = facing
        self.durationMs = durationMs
        self.includeAudio = includeAudio
        self.format = format
        self.deviceId = deviceId
    }
}
