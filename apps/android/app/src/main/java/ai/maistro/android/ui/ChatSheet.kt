package ai.maistro.android.ui

import androidx.compose.runtime.Composable
import ai.maistro.android.MainViewModel
import ai.maistro.android.ui.chat.ChatSheetContent

@Composable
fun ChatSheet(viewModel: MainViewModel) {
  ChatSheetContent(viewModel = viewModel)
}
