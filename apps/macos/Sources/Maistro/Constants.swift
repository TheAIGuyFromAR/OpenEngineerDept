import Foundation

// Stable identifier used for both the macOS LaunchAgent label and Nix-managed defaults suite.
// nix-maistro writes app defaults into this suite to survive app bundle identifier churn.
let launchdLabel = "ai.maistro.mac"
let gatewayLaunchdLabel = "ai.maistro.gateway"
let onboardingVersionKey = "maistro.onboardingVersion"
let onboardingSeenKey = "maistro.onboardingSeen"
let currentOnboardingVersion = 7
let pauseDefaultsKey = "maistro.pauseEnabled"
let iconAnimationsEnabledKey = "maistro.iconAnimationsEnabled"
let swabbleEnabledKey = "maistro.swabbleEnabled"
let swabbleTriggersKey = "maistro.swabbleTriggers"
let voiceWakeTriggerChimeKey = "maistro.voiceWakeTriggerChime"
let voiceWakeSendChimeKey = "maistro.voiceWakeSendChime"
let showDockIconKey = "maistro.showDockIcon"
let defaultVoiceWakeTriggers = ["maistro"]
let voiceWakeMaxWords = 32
let voiceWakeMaxWordLength = 64
let voiceWakeMicKey = "maistro.voiceWakeMicID"
let voiceWakeMicNameKey = "maistro.voiceWakeMicName"
let voiceWakeLocaleKey = "maistro.voiceWakeLocaleID"
let voiceWakeAdditionalLocalesKey = "maistro.voiceWakeAdditionalLocaleIDs"
let voicePushToTalkEnabledKey = "maistro.voicePushToTalkEnabled"
let talkEnabledKey = "maistro.talkEnabled"
let iconOverrideKey = "maistro.iconOverride"
let connectionModeKey = "maistro.connectionMode"
let remoteTargetKey = "maistro.remoteTarget"
let remoteIdentityKey = "maistro.remoteIdentity"
let remoteProjectRootKey = "maistro.remoteProjectRoot"
let remoteCliPathKey = "maistro.remoteCliPath"
let canvasEnabledKey = "maistro.canvasEnabled"
let cameraEnabledKey = "maistro.cameraEnabled"
let systemRunPolicyKey = "maistro.systemRunPolicy"
let systemRunAllowlistKey = "maistro.systemRunAllowlist"
let systemRunEnabledKey = "maistro.systemRunEnabled"
let locationModeKey = "maistro.locationMode"
let locationPreciseKey = "maistro.locationPreciseEnabled"
let peekabooBridgeEnabledKey = "maistro.peekabooBridgeEnabled"
let deepLinkKeyKey = "maistro.deepLinkKey"
let modelCatalogPathKey = "maistro.modelCatalogPath"
let modelCatalogReloadKey = "maistro.modelCatalogReload"
let cliInstallPromptedVersionKey = "maistro.cliInstallPromptedVersion"
let heartbeatsEnabledKey = "maistro.heartbeatsEnabled"
let debugPaneEnabledKey = "maistro.debugPaneEnabled"
let debugFileLogEnabledKey = "maistro.debug.fileLogEnabled"
let appLogLevelKey = "maistro.debug.appLogLevel"
let voiceWakeSupported: Bool = ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26
