import Foundation

public enum MaistroDeviceCommand: String, Codable, Sendable {
    case status = "device.status"
    case info = "device.info"
}

public enum MaistroBatteryState: String, Codable, Sendable {
    case unknown
    case unplugged
    case charging
    case full
}

public enum MaistroThermalState: String, Codable, Sendable {
    case nominal
    case fair
    case serious
    case critical
}

public enum MaistroNetworkPathStatus: String, Codable, Sendable {
    case satisfied
    case unsatisfied
    case requiresConnection
}

public enum MaistroNetworkInterfaceType: String, Codable, Sendable {
    case wifi
    case cellular
    case wired
    case other
}

public struct MaistroBatteryStatusPayload: Codable, Sendable, Equatable {
    public var level: Double?
    public var state: MaistroBatteryState
    public var lowPowerModeEnabled: Bool

    public init(level: Double?, state: MaistroBatteryState, lowPowerModeEnabled: Bool) {
        self.level = level
        self.state = state
        self.lowPowerModeEnabled = lowPowerModeEnabled
    }
}

public struct MaistroThermalStatusPayload: Codable, Sendable, Equatable {
    public var state: MaistroThermalState

    public init(state: MaistroThermalState) {
        self.state = state
    }
}

public struct MaistroStorageStatusPayload: Codable, Sendable, Equatable {
    public var totalBytes: Int64
    public var freeBytes: Int64
    public var usedBytes: Int64

    public init(totalBytes: Int64, freeBytes: Int64, usedBytes: Int64) {
        self.totalBytes = totalBytes
        self.freeBytes = freeBytes
        self.usedBytes = usedBytes
    }
}

public struct MaistroNetworkStatusPayload: Codable, Sendable, Equatable {
    public var status: MaistroNetworkPathStatus
    public var isExpensive: Bool
    public var isConstrained: Bool
    public var interfaces: [MaistroNetworkInterfaceType]

    public init(
        status: MaistroNetworkPathStatus,
        isExpensive: Bool,
        isConstrained: Bool,
        interfaces: [MaistroNetworkInterfaceType])
    {
        self.status = status
        self.isExpensive = isExpensive
        self.isConstrained = isConstrained
        self.interfaces = interfaces
    }
}

public struct MaistroDeviceStatusPayload: Codable, Sendable, Equatable {
    public var battery: MaistroBatteryStatusPayload
    public var thermal: MaistroThermalStatusPayload
    public var storage: MaistroStorageStatusPayload
    public var network: MaistroNetworkStatusPayload
    public var uptimeSeconds: Double

    public init(
        battery: MaistroBatteryStatusPayload,
        thermal: MaistroThermalStatusPayload,
        storage: MaistroStorageStatusPayload,
        network: MaistroNetworkStatusPayload,
        uptimeSeconds: Double)
    {
        self.battery = battery
        self.thermal = thermal
        self.storage = storage
        self.network = network
        self.uptimeSeconds = uptimeSeconds
    }
}

public struct MaistroDeviceInfoPayload: Codable, Sendable, Equatable {
    public var deviceName: String
    public var modelIdentifier: String
    public var systemName: String
    public var systemVersion: String
    public var appVersion: String
    public var appBuild: String
    public var locale: String

    public init(
        deviceName: String,
        modelIdentifier: String,
        systemName: String,
        systemVersion: String,
        appVersion: String,
        appBuild: String,
        locale: String)
    {
        self.deviceName = deviceName
        self.modelIdentifier = modelIdentifier
        self.systemName = systemName
        self.systemVersion = systemVersion
        self.appVersion = appVersion
        self.appBuild = appBuild
        self.locale = locale
    }
}
