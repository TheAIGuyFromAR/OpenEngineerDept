import CoreLocation
import Foundation
import MaistroKit
import UIKit

protocol CameraServicing: Sendable {
    func listDevices() async -> [CameraController.CameraDeviceInfo]
    func snap(params: MaistroCameraSnapParams) async throws -> (format: String, base64: String, width: Int, height: Int)
    func clip(params: MaistroCameraClipParams) async throws -> (format: String, base64: String, durationMs: Int, hasAudio: Bool)
}

protocol ScreenRecordingServicing: Sendable {
    func record(
        screenIndex: Int?,
        durationMs: Int?,
        fps: Double?,
        includeAudio: Bool?,
        outPath: String?) async throws -> String
}

@MainActor
protocol LocationServicing: Sendable {
    func authorizationStatus() -> CLAuthorizationStatus
    func accuracyAuthorization() -> CLAccuracyAuthorization
    func ensureAuthorization(mode: MaistroLocationMode) async -> CLAuthorizationStatus
    func currentLocation(
        params: MaistroLocationGetParams,
        desiredAccuracy: MaistroLocationAccuracy,
        maxAgeMs: Int?,
        timeoutMs: Int?) async throws -> CLLocation
    func startLocationUpdates(
        desiredAccuracy: MaistroLocationAccuracy,
        significantChangesOnly: Bool) -> AsyncStream<CLLocation>
    func stopLocationUpdates()
    func startMonitoringSignificantLocationChanges(onUpdate: @escaping @Sendable (CLLocation) -> Void)
    func stopMonitoringSignificantLocationChanges()
}

protocol DeviceStatusServicing: Sendable {
    func status() async throws -> MaistroDeviceStatusPayload
    func info() -> MaistroDeviceInfoPayload
}

protocol PhotosServicing: Sendable {
    func latest(params: MaistroPhotosLatestParams) async throws -> MaistroPhotosLatestPayload
}

protocol ContactsServicing: Sendable {
    func search(params: MaistroContactsSearchParams) async throws -> MaistroContactsSearchPayload
    func add(params: MaistroContactsAddParams) async throws -> MaistroContactsAddPayload
}

protocol CalendarServicing: Sendable {
    func events(params: MaistroCalendarEventsParams) async throws -> MaistroCalendarEventsPayload
    func add(params: MaistroCalendarAddParams) async throws -> MaistroCalendarAddPayload
}

protocol RemindersServicing: Sendable {
    func list(params: MaistroRemindersListParams) async throws -> MaistroRemindersListPayload
    func add(params: MaistroRemindersAddParams) async throws -> MaistroRemindersAddPayload
}

protocol MotionServicing: Sendable {
    func activities(params: MaistroMotionActivityParams) async throws -> MaistroMotionActivityPayload
    func pedometer(params: MaistroPedometerParams) async throws -> MaistroPedometerPayload
}

extension CameraController: CameraServicing {}
extension ScreenRecordService: ScreenRecordingServicing {}
extension LocationService: LocationServicing {}
