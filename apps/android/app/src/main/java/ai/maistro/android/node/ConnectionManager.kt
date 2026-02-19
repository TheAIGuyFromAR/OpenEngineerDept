package ai.maistro.android.node

import android.os.Build
import ai.maistro.android.BuildConfig
import ai.maistro.android.SecurePrefs
import ai.maistro.android.gateway.GatewayClientInfo
import ai.maistro.android.gateway.GatewayConnectOptions
import ai.maistro.android.gateway.GatewayEndpoint
import ai.maistro.android.gateway.GatewayTlsParams
import ai.maistro.android.protocol.MaistroCanvasA2UICommand
import ai.maistro.android.protocol.MaistroCanvasCommand
import ai.maistro.android.protocol.MaistroCameraCommand
import ai.maistro.android.protocol.MaistroLocationCommand
import ai.maistro.android.protocol.MaistroScreenCommand
import ai.maistro.android.protocol.MaistroSmsCommand
import ai.maistro.android.protocol.MaistroCapability
import ai.maistro.android.LocationMode
import ai.maistro.android.VoiceWakeMode

class ConnectionManager(
  private val prefs: SecurePrefs,
  private val cameraEnabled: () -> Boolean,
  private val locationMode: () -> LocationMode,
  private val voiceWakeMode: () -> VoiceWakeMode,
  private val smsAvailable: () -> Boolean,
  private val hasRecordAudioPermission: () -> Boolean,
  private val manualTls: () -> Boolean,
) {
  companion object {
    internal fun resolveTlsParamsForEndpoint(
      endpoint: GatewayEndpoint,
      storedFingerprint: String?,
      manualTlsEnabled: Boolean,
    ): GatewayTlsParams? {
      val stableId = endpoint.stableId
      val stored = storedFingerprint?.trim().takeIf { !it.isNullOrEmpty() }
      val isManual = stableId.startsWith("manual|")

      if (isManual) {
        if (!manualTlsEnabled) return null
        if (!stored.isNullOrBlank()) {
          return GatewayTlsParams(
            required = true,
            expectedFingerprint = stored,
            allowTOFU = false,
            stableId = stableId,
          )
        }
        return GatewayTlsParams(
          required = true,
          expectedFingerprint = null,
          allowTOFU = false,
          stableId = stableId,
        )
      }

      // Prefer stored pins. Never let discovery-provided TXT override a stored fingerprint.
      if (!stored.isNullOrBlank()) {
        return GatewayTlsParams(
          required = true,
          expectedFingerprint = stored,
          allowTOFU = false,
          stableId = stableId,
        )
      }

      val hinted = endpoint.tlsEnabled || !endpoint.tlsFingerprintSha256.isNullOrBlank()
      if (hinted) {
        // TXT is unauthenticated. Do not treat the advertised fingerprint as authoritative.
        return GatewayTlsParams(
          required = true,
          expectedFingerprint = null,
          allowTOFU = false,
          stableId = stableId,
        )
      }

      return null
    }
  }

  fun buildInvokeCommands(): List<String> =
    buildList {
      add(MaistroCanvasCommand.Present.rawValue)
      add(MaistroCanvasCommand.Hide.rawValue)
      add(MaistroCanvasCommand.Navigate.rawValue)
      add(MaistroCanvasCommand.Eval.rawValue)
      add(MaistroCanvasCommand.Snapshot.rawValue)
      add(MaistroCanvasA2UICommand.Push.rawValue)
      add(MaistroCanvasA2UICommand.PushJSONL.rawValue)
      add(MaistroCanvasA2UICommand.Reset.rawValue)
      add(MaistroScreenCommand.Record.rawValue)
      if (cameraEnabled()) {
        add(MaistroCameraCommand.Snap.rawValue)
        add(MaistroCameraCommand.Clip.rawValue)
      }
      if (locationMode() != LocationMode.Off) {
        add(MaistroLocationCommand.Get.rawValue)
      }
      if (smsAvailable()) {
        add(MaistroSmsCommand.Send.rawValue)
      }
      if (BuildConfig.DEBUG) {
        add("debug.logs")
        add("debug.ed25519")
      }
      add("app.update")
    }

  fun buildCapabilities(): List<String> =
    buildList {
      add(MaistroCapability.Canvas.rawValue)
      add(MaistroCapability.Screen.rawValue)
      if (cameraEnabled()) add(MaistroCapability.Camera.rawValue)
      if (smsAvailable()) add(MaistroCapability.Sms.rawValue)
      if (voiceWakeMode() != VoiceWakeMode.Off && hasRecordAudioPermission()) {
        add(MaistroCapability.VoiceWake.rawValue)
      }
      if (locationMode() != LocationMode.Off) {
        add(MaistroCapability.Location.rawValue)
      }
    }

  fun resolvedVersionName(): String {
    val versionName = BuildConfig.VERSION_NAME.trim().ifEmpty { "dev" }
    return if (BuildConfig.DEBUG && !versionName.contains("dev", ignoreCase = true)) {
      "$versionName-dev"
    } else {
      versionName
    }
  }

  fun resolveModelIdentifier(): String? {
    return listOfNotNull(Build.MANUFACTURER, Build.MODEL)
      .joinToString(" ")
      .trim()
      .ifEmpty { null }
  }

  fun buildUserAgent(): String {
    val version = resolvedVersionName()
    val release = Build.VERSION.RELEASE?.trim().orEmpty()
    val releaseLabel = if (release.isEmpty()) "unknown" else release
    return "MaistroAndroid/$version (Android $releaseLabel; SDK ${Build.VERSION.SDK_INT})"
  }

  fun buildClientInfo(clientId: String, clientMode: String): GatewayClientInfo {
    return GatewayClientInfo(
      id = clientId,
      displayName = prefs.displayName.value,
      version = resolvedVersionName(),
      platform = "android",
      mode = clientMode,
      instanceId = prefs.instanceId.value,
      deviceFamily = "Android",
      modelIdentifier = resolveModelIdentifier(),
    )
  }

  fun buildNodeConnectOptions(): GatewayConnectOptions {
    return GatewayConnectOptions(
      role = "node",
      scopes = emptyList(),
      caps = buildCapabilities(),
      commands = buildInvokeCommands(),
      permissions = emptyMap(),
      client = buildClientInfo(clientId = "maistro-android", clientMode = "node"),
      userAgent = buildUserAgent(),
    )
  }

  fun buildOperatorConnectOptions(): GatewayConnectOptions {
    return GatewayConnectOptions(
      role = "operator",
      scopes = listOf("operator.read", "operator.write", "operator.talk.secrets"),
      caps = emptyList(),
      commands = emptyList(),
      permissions = emptyMap(),
      client = buildClientInfo(clientId = "maistro-control-ui", clientMode = "ui"),
      userAgent = buildUserAgent(),
    )
  }

  fun resolveTlsParams(endpoint: GatewayEndpoint): GatewayTlsParams? {
    val stored = prefs.loadGatewayTlsFingerprint(endpoint.stableId)
    return resolveTlsParamsForEndpoint(endpoint, storedFingerprint = stored, manualTlsEnabled = manualTls())
  }
}
