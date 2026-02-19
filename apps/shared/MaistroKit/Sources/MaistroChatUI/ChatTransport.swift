import Foundation

public enum MaistroChatTransportEvent: Sendable {
    case health(ok: Bool)
    case tick
    case chat(MaistroChatEventPayload)
    case agent(MaistroAgentEventPayload)
    case seqGap
}

public protocol MaistroChatTransport: Sendable {
    func requestHistory(sessionKey: String) async throws -> MaistroChatHistoryPayload
    func sendMessage(
        sessionKey: String,
        message: String,
        thinking: String,
        idempotencyKey: String,
        attachments: [MaistroChatAttachmentPayload]) async throws -> MaistroChatSendResponse

    func abortRun(sessionKey: String, runId: String) async throws
    func listSessions(limit: Int?) async throws -> MaistroChatSessionsListResponse

    func requestHealth(timeoutMs: Int) async throws -> Bool
    func events() -> AsyncStream<MaistroChatTransportEvent>

    func setActiveSessionKey(_ sessionKey: String) async throws
}

extension MaistroChatTransport {
    public func setActiveSessionKey(_: String) async throws {}

    public func abortRun(sessionKey _: String, runId _: String) async throws {
        throw NSError(
            domain: "MaistroChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "chat.abort not supported by this transport"])
    }

    public func listSessions(limit _: Int?) async throws -> MaistroChatSessionsListResponse {
        throw NSError(
            domain: "MaistroChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "sessions.list not supported by this transport"])
    }
}
