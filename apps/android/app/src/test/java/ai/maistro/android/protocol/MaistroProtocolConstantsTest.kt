package ai.maistro.android.protocol

import org.junit.Assert.assertEquals
import org.junit.Test

class MaistroProtocolConstantsTest {
  @Test
  fun canvasCommandsUseStableStrings() {
    assertEquals("canvas.present", MaistroCanvasCommand.Present.rawValue)
    assertEquals("canvas.hide", MaistroCanvasCommand.Hide.rawValue)
    assertEquals("canvas.navigate", MaistroCanvasCommand.Navigate.rawValue)
    assertEquals("canvas.eval", MaistroCanvasCommand.Eval.rawValue)
    assertEquals("canvas.snapshot", MaistroCanvasCommand.Snapshot.rawValue)
  }

  @Test
  fun a2uiCommandsUseStableStrings() {
    assertEquals("canvas.a2ui.push", MaistroCanvasA2UICommand.Push.rawValue)
    assertEquals("canvas.a2ui.pushJSONL", MaistroCanvasA2UICommand.PushJSONL.rawValue)
    assertEquals("canvas.a2ui.reset", MaistroCanvasA2UICommand.Reset.rawValue)
  }

  @Test
  fun capabilitiesUseStableStrings() {
    assertEquals("canvas", MaistroCapability.Canvas.rawValue)
    assertEquals("camera", MaistroCapability.Camera.rawValue)
    assertEquals("screen", MaistroCapability.Screen.rawValue)
    assertEquals("voiceWake", MaistroCapability.VoiceWake.rawValue)
  }

  @Test
  fun screenCommandsUseStableStrings() {
    assertEquals("screen.record", MaistroScreenCommand.Record.rawValue)
  }
}
