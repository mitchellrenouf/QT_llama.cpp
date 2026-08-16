import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtWebSockets
import QtMultimedia

ApplicationWindow {
    id: window
    width: 1024
    height: 768
    minimumWidth: 800
    minimumHeight: 600
    visible: true
    title: "Gemma 4 AI Assistant (Qt6 / Flatpak)"
    color: "#121316"

    property string currentMode: "general"
    property int estimatedTokens: 0
    property bool isThinking: false
    property bool speechEnabled: false
    property string statusText: "Connecting to agent..."
    property string currentModelName: "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0"
    property bool isModelLoaded: false
    property bool isDownloading: false
    property real downloadProgress: 0.0
    property int currentFileIndex: 1
    property int totalFilesCount: 1
    property string downloadStatusMessage: ""

    function findTargetPort() {
        if (Qt.application.arguments) {
            for (var i = Qt.application.arguments.length - 1; i >= 0; i--) {
                var arg = Qt.application.arguments[i]
                if (/^\d+$/.test(arg)) {
                    return arg
                }
            }
        }
        return "9876"
    }

    // --- WebSocket IPC Connection to in-process Rust Agent ---
    WebSocket {
        id: wsClient
        property string targetPort: findTargetPort()
        url: "ws://127.0.0.1:" + targetPort
        active: true

        onStatusChanged: function() {
            if (wsClient.status === WebSocket.Open) {
                console.log("WebSocket connected to Gemma Agent backend on port " + targetPort)
            } else if (wsClient.status === WebSocket.Error) {
                console.log("WebSocket error: " + wsClient.errorString)
                window.statusText = "Backend connection error: " + wsClient.errorString
            } else if (wsClient.status === WebSocket.Closed) {
                console.log("WebSocket closed")
            }
        }

        onTextMessageReceived: function(message) {
            try {
                var data = JSON.parse(message)
                handleWsEvent(data)
            } catch (err) {
                console.log("Failed to parse backend message: " + err)
            }
        }
    }

    function sendWsCommand(cmd) {
        if (wsClient.status === WebSocket.Open) {
            wsClient.sendTextMessage(JSON.stringify(cmd))
        } else {
            console.log("Cannot send command, WebSocket is not open")
        }
    }

    function handleWsEvent(evt) {
        switch (evt.type) {
            case "init_state":
                window.isModelLoaded = evt.model_loaded || false
                window.currentModelName = evt.model_name || window.currentModelName
                window.currentMode = evt.mode || "general"
                window.speechEnabled = evt.speech_enabled || false
                window.estimatedTokens = evt.tokens || 0
                window.statusText = window.isModelLoaded ? "Ready" : "Model not loaded"
                if (!window.isModelLoaded) {
                    modelSetupDialog.open()
                } else {
                    modelSetupDialog.close()
                }
                break

            case "stream_thought":
                window.isThinking = true
                window.statusText = "Gemma is reasoning..."
                appendThoughtToken(evt.thought)
                break

            case "stream_token":
                window.isThinking = false
                window.statusText = "Gemma is responding..."
                appendContentToken(evt.token)
                break

            case "tool_started":
                appendMessage("assistant", "", "", false, evt.name, evt.args, "", false)
                window.statusText = "Executing tool: " + evt.name
                break

            case "tool_finished":
                updateLastToolResult(evt.name, evt.result)
                window.statusText = "Tool finished: " + evt.name
                break

            case "turn_done":
                window.isThinking = false
                window.statusText = "Ready"
                window.estimatedTokens = evt.tokens || 0
                finalizeCurrentAssistantMessage(evt.content, evt.thought)
                break

            case "download_progress":
                window.isDownloading = true
                window.downloadStatusMessage = evt.message
                window.downloadProgress = evt.progress
                window.currentFileIndex = evt.file_idx
                window.totalFilesCount = evt.total_files
                break

            case "model_loaded":
                window.isDownloading = false
                window.isModelLoaded = true
                window.currentModelName = evt.model_name
                window.statusText = "Ready (In-Process llama.cpp)"
                modelSetupDialog.close()
                break

            case "error":
                window.isThinking = false
                window.statusText = "Error: " + evt.message
                appendMessage("assistant", "⚠️ " + evt.message, "", false, "", "", "", false)
                break

            case "mode_changed":
                window.currentMode = evt.mode
                break

            case "speech_toggled":
                window.speechEnabled = evt.enabled
                break

            case "history_cleared":
                chatModel.clear()
                window.estimatedTokens = evt.tokens || 0
                window.statusText = "Ready"
                break
        }
    }

    function appendMessage(role, content, thought, thoughtExpanded, toolName, toolArgs, toolResult, toolExpanded) {
        chatModel.append({
            "role": role,
            "content": content || "",
            "thought": thought || "",
            "thoughtExpanded": thoughtExpanded || false,
            "toolName": toolName || "",
            "toolArgs": toolArgs || "",
            "toolResult": toolResult || "",
            "toolExpanded": toolExpanded || false
        })
        chatListView.positionViewAtEnd()
    }

    function updateLastToolResult(toolName, result) {
        if (chatModel.count > 0) {
            for (var i = chatModel.count - 1; i >= 0; i--) {
                var item = chatModel.get(i)
                if (item.toolName === toolName) {
                    chatModel.setProperty(i, "toolResult", result)
                    break
                }
            }
        }
        chatListView.positionViewAtEnd()
    }

    function appendThoughtToken(token) {
        if (chatModel.count > 0) {
            var lastIdx = chatModel.count - 1
            var lastItem = chatModel.get(lastIdx)
            if (lastItem.role === "assistant" && lastItem.toolName === "") {
                var updatedThought = lastItem.thought + token
                chatModel.setProperty(lastIdx, "thought", updatedThought)
                chatListView.positionViewAtEnd()
                return
            }
        }
        appendMessage("assistant", "", token, false, "", "", "", false)
    }

    function appendContentToken(token) {
        if (chatModel.count > 0) {
            var lastIdx = chatModel.count - 1
            var lastItem = chatModel.get(lastIdx)
            if (lastItem.role === "assistant" && lastItem.toolName === "") {
                var updatedContent = lastItem.content + token
                chatModel.setProperty(lastIdx, "content", updatedContent)
                chatListView.positionViewAtEnd()
                return
            }
        }
        appendMessage("assistant", token, "", false, "", "", "", false)
    }

    function finalizeCurrentAssistantMessage(content, thought) {
        if (chatModel.count > 0) {
            var lastIdx = chatModel.count - 1
            var lastItem = chatModel.get(lastIdx)
            if (lastItem.role === "assistant" && lastItem.toolName === "") {
                if (content && content.length > 0) {
                    chatModel.setProperty(lastIdx, "content", content)
                }
                if (thought && thought.length > 0) {
                    chatModel.setProperty(lastIdx, "thought", thought)
                }
            }
        }
        chatListView.positionViewAtEnd()
    }

    function formatDuration(ms) {
        if (!ms || ms < 0 || isNaN(ms)) return "0:00"
        var totalSec = Math.floor(ms / 1000)
        var mins = Math.floor(totalSec / 60)
        var secs = totalSec % 60
        return mins + ":" + (secs < 10 ? "0" : "") + secs
    }

    function extractImages(text) {
        if (!text) return []
        var images = []
        var mdImgRegex = /!\[([^\]]*)\]\(([^)]+)\)/g
        var match
        while ((match = mdImgRegex.exec(text)) !== null) {
            var url = match[2].trim()
            if (!url.startsWith("http://") && !url.startsWith("https://") && !url.startsWith("file://") && !url.startsWith("qrc:/") && !url.startsWith("data:")) {
                if (url.startsWith("/")) {
                    url = "file://" + url
                }
            }
            images.push({ "alt": match[1] || "Image", "src": url })
        }
        var urlRegex = /(?:^|\s)(https?:\/\/[^\s]+\.(?:png|jpg|jpeg|gif|webp|svg)|file:\/\/\/[^\s]+\.(?:png|jpg|jpeg|gif|webp|svg))(?:\s|$)/gi
        while ((match = urlRegex.exec(text)) !== null) {
            var directUrl = match[1].trim()
            var exists = false
            for (var i = 0; i < images.length; i++) {
                if (images[i].src === directUrl) { exists = true; break; }
            }
            if (!exists) {
                images.push({ "alt": "Image", "src": directUrl })
            }
        }
        return images
    }

    function extractVideos(text) {
        if (!text) return []
        var videos = []
        var vidRegex = /(?:\[([^\]]*)\]\(([^)]+\.(?:mp4|webm|mkv|mov|avi))\)|(?:^|\s)(https?:\/\/[^\s]+\.(?:mp4|webm|mkv|mov|avi)|file:\/\/\/[^\s]+\.(?:mp4|webm|mkv|mov|avi))(?:\s|$))/gi
        var match
        while ((match = vidRegex.exec(text)) !== null) {
            var src = match[2] ? match[2].trim() : (match[3] ? match[3].trim() : "")
            var title = match[1] || "Video Playback"
            if (!src.startsWith("http://") && !src.startsWith("https://") && !src.startsWith("file://") && !src.startsWith("qrc:/")) {
                if (src.startsWith("/")) src = "file://" + src
            }
            if (src.length > 0) {
                videos.push({ "title": title, "src": src })
            }
        }
        return videos
    }

    function extractAudios(text) {
        if (!text) return []
        var audios = []
        var audRegex = /(?:\[([^\]]*)\]\(([^)]+\.(?:mp3|wav|ogg|flac|m4a|aac))\)|(?:^|\s)(https?:\/\/[^\s]+\.(?:mp3|wav|ogg|flac|m4a|aac)|file:\/\/\/[^\s]+\.(?:mp3|wav|ogg|flac|m4a|aac))(?:\s|$))/gi
        var match
        while ((match = audRegex.exec(text)) !== null) {
            var src = match[2] ? match[2].trim() : (match[3] ? match[3].trim() : "")
            var title = match[1] || "Audio Track"
            if (!src.startsWith("http://") && !src.startsWith("https://") && !src.startsWith("file://") && !src.startsWith("qrc:/")) {
                if (src.startsWith("/")) src = "file://" + src
            }
            if (src.length > 0) {
                audios.push({ "title": title, "src": src })
            }
        }
        return audios
    }

    function isThoughtValid(thought) {
        if (!thought) return false
        var trimmed = thought.trim()
        if (trimmed.length === 0) return false
        var lower = trimmed.toLowerCase()
        if (lower === "thought" || lower === "thought process" || lower === "reasoning" || lower === "<|channel>thought<channel|>") return false
        return true
    }

    function startModelDownload(repo, quant) {
        var fullSpec = repo + ":" + quant
        window.currentModelName = fullSpec
        window.isDownloading = true
        window.downloadProgress = 0.0
        window.downloadStatusMessage = "⏳ Resolving Hugging Face repository & shards..."
        sendWsCommand({ "type": "load_hf_model", "spec": fullSpec })
    }

    ListModel {
        id: chatModel
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // --- Top Header Bar ---
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 64
            color: "#1a1c23"
            border.color: "#282b36"
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                spacing: 16

                Label {
                    text: "🚀 Gemma 4"
                    font.bold: true
                    font.pixelSize: 20
                    color: "#f8f9fa"
                }

                // Model Specifier Button / Badge
                Button {
                    id: modelBadgeBtn
                    text: "📦 " + window.currentModelName + (window.isDownloading ? " (" + Math.round(window.downloadProgress * 100) + "%)" : "")
                    background: Rectangle {
                        color: window.isDownloading ? "#1e3a8a" : "#252834"
                        radius: 6
                        border.color: window.isDownloading ? "#60a5fa" : "#3b82f6"
                    }
                    contentItem: Text {
                        text: modelBadgeBtn.text
                        color: "#93c5fd"
                        font.pixelSize: 12
                        font.bold: true
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: 8
                        rightPadding: 8
                    }
                    onClicked: {
                        modelSetupDialog.open()
                    }
                }

                Item { Layout.fillWidth: true }

                // Mode Selector
                ComboBox {
                    id: modeCombo
                    model: ["General", "Code", "Desktop"]
                    currentIndex: window.currentMode === "code" ? 1 : (window.currentMode === "desktop" ? 2 : 0)
                    Layout.preferredWidth: 140
                    background: Rectangle {
                        color: "#252834"
                        radius: 6
                        border.color: "#383d4f"
                    }
                    contentItem: Text {
                        text: modeCombo.displayText
                        color: "#e2e8f0"
                        font.pixelSize: 13
                        font.bold: true
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: 10
                    }
                    onActivated: {
                        var m = modeCombo.currentText.toLowerCase()
                        window.currentMode = m
                        sendWsCommand({ "type": "switch_mode", "mode": m })
                    }
                }

                // Speech Toggle Button
                Button {
                    id: speechBtn
                    text: window.speechEnabled ? "🔊 Voice: ON" : "🔇 Voice: OFF"
                    background: Rectangle {
                        color: window.speechEnabled ? "#1e3a8a" : "#252834"
                        radius: 6
                        border.color: window.speechEnabled ? "#3b82f6" : "#383d4f"
                    }
                    contentItem: Text {
                        text: speechBtn.text
                        color: window.speechEnabled ? "#93c5fd" : "#94a3b8"
                        font.pixelSize: 12
                        font.bold: true
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: {
                        sendWsCommand({ "type": "toggle_speech" })
                    }
                }

                // Clear History Button
                Button {
                    id: clearBtn
                    text: "🗑️ Clear"
                    background: Rectangle {
                        color: "#252834"
                        radius: 6
                        border.color: "#383d4f"
                    }
                    contentItem: Text {
                        text: clearBtn.text
                        color: "#cbd5e1"
                        font.pixelSize: 12
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: {
                        sendWsCommand({ "type": "clear_history" })
                    }
                }
            }
        }

        // --- Chat Messages Area ---
        ListView {
            id: chatListView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: chatModel
            spacing: 14
            topMargin: 16
            bottomMargin: 16
            leftMargin: 20
            rightMargin: 20

            delegate: ColumnLayout {
                id: delegateRoot
                width: chatListView.width - 40
                spacing: 6

                // Role Header Badge
                RowLayout {
                    spacing: 8
                    Rectangle {
                        width: 10
                        height: 10
                        radius: 5
                        color: model.role === "user" ? "#38bdf8" : (model.toolName !== "" ? "#fbbf24" : "#a855f7")
                    }
                    Label {
                        text: model.role === "user" ? "You" : (model.toolName !== "" ? ("Tool: " + model.toolName) : "Gemma 4")
                        font.bold: true
                        font.pixelSize: 13
                        color: model.role === "user" ? "#38bdf8" : (model.toolName !== "" ? "#fbbf24" : "#c084fc")
                    }
                }

                // Message Bubble
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: bubbleContentColumn.implicitHeight + 28
                    radius: 8
                    color: model.role === "user" ? "#1e293b" : (model.toolName !== "" ? "#181a22" : "#181a20")
                    border.color: model.role === "user" ? "#334155" : (model.toolName !== "" ? "#374151" : "#27272a")
                    border.width: 1

                    ColumnLayout {
                        id: bubbleContentColumn
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10

                        // Collapsible Tool Call Card (Collapsed by default!)
                        Rectangle {
                            visible: model.toolName !== undefined && model.toolName !== null && model.toolName !== ""
                            Layout.fillWidth: true
                            implicitHeight: toolColumn.implicitHeight + 16
                            color: "#12141c"
                            border.color: "#374151"
                            border.width: 1
                            radius: 6

                            ColumnLayout {
                                id: toolColumn
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 8

                                // Clickable toggle header
                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 28
                                    color: toolMouseArea.containsMouse ? "#202534" : "transparent"
                                    radius: 4

                                    MouseArea {
                                        id: toolMouseArea
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            chatModel.setProperty(index, "toolExpanded", !model.toolExpanded)
                                        }
                                    }

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.leftMargin: 6
                                        anchors.rightMargin: 6
                                        spacing: 8

                                        Text {
                                            text: model.toolExpanded ? ("▼ 🔧 Tool Call: " + model.toolName) : ("▶ 🔧 Tool Call: " + model.toolName + " (click to expand)")
                                            color: "#fbbf24"
                                            font.bold: true
                                            font.pixelSize: 12
                                        }

                                        Item { Layout.fillWidth: true }

                                        Rectangle {
                                            Layout.preferredHeight: 20
                                            Layout.preferredWidth: toolStatusText.implicitWidth + 12
                                            radius: 10
                                            color: (model.toolResult && model.toolResult.length > 0) ? "#14532d" : "#713f12"
                                            Text {
                                                id: toolStatusText
                                                anchors.centerIn: parent
                                                text: (model.toolResult && model.toolResult.length > 0) ? "✓ Done" : "⏳ Executing"
                                                color: (model.toolResult && model.toolResult.length > 0) ? "#86efac" : "#fde047"
                                                font.pixelSize: 10
                                                font.bold: true
                                            }
                                        }

                                        Text {
                                            text: model.toolExpanded ? "Hide" : "Show"
                                            color: "#94a3b8"
                                            font.pixelSize: 11
                                        }
                                    }
                                }

                                // Collapsible Tool Details (Arguments and Result)
                                ColumnLayout {
                                    visible: model.toolExpanded === true
                                    Layout.fillWidth: true
                                    spacing: 6

                                    Label {
                                        text: "📥 Arguments:"
                                        color: "#9ca3af"
                                        font.bold: true
                                        font.pixelSize: 11
                                    }

                                    Rectangle {
                                        Layout.fillWidth: true
                                        implicitHeight: toolArgsBox.implicitHeight + 12
                                        color: "#0a0c10"
                                        border.color: "#282c3c"
                                        border.width: 1
                                        radius: 4

                                        TextEdit {
                                            id: toolArgsBox
                                            anchors.fill: parent
                                            anchors.margins: 6
                                            text: model.toolArgs || "{}"
                                            color: "#cbd5e1"
                                            font.pixelSize: 11
                                            font.family: "Monospace"
                                            wrapMode: TextEdit.Wrap
                                            readOnly: true
                                            selectByMouse: true
                                        }
                                    }

                                    Label {
                                        text: "📤 Result:"
                                        color: "#86efac"
                                        font.bold: true
                                        font.pixelSize: 11
                                        visible: model.toolResult !== undefined && model.toolResult !== null && model.toolResult.length > 0
                                    }

                                    Rectangle {
                                        visible: model.toolResult !== undefined && model.toolResult !== null && model.toolResult.length > 0
                                        Layout.fillWidth: true
                                        implicitHeight: Math.min(250, toolResBox.implicitHeight + 12)
                                        color: "#0a0c10"
                                        border.color: "#282c3c"
                                        border.width: 1
                                        radius: 4

                                        ScrollView {
                                            anchors.fill: parent
                                            anchors.margins: 6
                                            clip: true

                                            TextEdit {
                                                id: toolResBox
                                                width: parent.width
                                                text: model.toolResult || ""
                                                color: "#86efac"
                                                font.pixelSize: 11
                                                font.family: "Monospace"
                                                wrapMode: TextEdit.Wrap
                                                readOnly: true
                                                selectByMouse: true
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Collapsible Thought Process Box (Collapsed by default!)
                        Rectangle {
                            visible: isThoughtValid(model.thought)
                            Layout.fillWidth: true
                            implicitHeight: thoughtColumn.implicitHeight + 16
                            color: "#13151b"
                            border.color: "#33384a"
                            border.width: 1
                            radius: 6

                            ColumnLayout {
                                id: thoughtColumn
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 6

                                // Clickable toggle header
                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 26
                                    color: thoughtMouseArea.containsMouse ? "#202433" : "transparent"
                                    radius: 4

                                    MouseArea {
                                        id: thoughtMouseArea
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            chatModel.setProperty(index, "thoughtExpanded", !model.thoughtExpanded)
                                        }
                                    }

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.leftMargin: 6
                                        anchors.rightMargin: 6
                                        spacing: 8

                                        Text {
                                            text: model.thoughtExpanded ? "▼ 🧠 Thought Process" : "▶ 🧠 Thought Process (click to expand)"
                                            color: "#eab308"
                                            font.bold: true
                                            font.pixelSize: 12
                                        }

                                        Item { Layout.fillWidth: true }

                                        Text {
                                            text: model.thoughtExpanded ? "Hide" : "Show"
                                            color: "#94a3b8"
                                            font.pixelSize: 11
                                        }
                                    }
                                }

                                // Thought text content (visible only when expanded)
                                TextEdit {
                                    visible: model.thoughtExpanded === true
                                    Layout.fillWidth: true
                                    text: model.thought || ""
                                    color: "#d4d4d8"
                                    font.pixelSize: 12
                                    font.italic: true
                                    font.family: "Monospace"
                                    wrapMode: TextEdit.Wrap
                                    readOnly: true
                                    selectByMouse: true
                                }
                            }
                        }

                        // Main Text Content with Rich Markdown Rendering
                        TextEdit {
                            visible: model.content !== undefined && model.content !== null && model.content.length > 0
                            Layout.fillWidth: true
                            text: model.content || ""
                            textFormat: TextEdit.MarkdownText
                            color: "#f1f5f9"
                            font.pixelSize: 14
                            font.family: "Sans Serif"
                            wrapMode: TextEdit.Wrap
                            readOnly: true
                            selectByMouse: true
                            onLinkActivated: function(link) {
                                Qt.openUrlExternally(link)
                            }
                        }

                        // Embedded Images
                        Repeater {
                            model: extractImages(model.content)
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.maximumWidth: 520
                                Layout.preferredHeight: 300
                                color: "#111318"
                                radius: 8
                                clip: true
                                border.color: "#2a2e3d"
                                border.width: 1

                                Image {
                                    anchors.fill: parent
                                    anchors.margins: 4
                                    source: modelData.src
                                    fillMode: Image.PreserveAspectFit
                                    smooth: true
                                    asynchronous: true
                                }

                                Rectangle {
                                    anchors.bottom: parent.bottom
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    height: 26
                                    color: "#cc181a24"
                                    Label {
                                        anchors.centerIn: parent
                                        text: "🔍 Click to open full image: " + modelData.alt
                                        color: "#38bdf8"
                                        font.pixelSize: 11
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        Qt.openUrlExternally(modelData.src)
                                    }
                                }
                            }
                        }

                        // Embedded Video Player
                        Repeater {
                            model: extractVideos(model.content)
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.maximumWidth: 560
                                Layout.preferredHeight: 340
                                color: "#0d0f14"
                                radius: 8
                                border.color: "#2a2e3d"
                                border.width: 1
                                clip: true

                                MediaPlayer {
                                    id: vidPlayer
                                    source: modelData.src
                                    audioOutput: AudioOutput { volume: 1.0 }
                                    videoOutput: vidOut
                                }

                                VideoOutput {
                                    id: vidOut
                                    anchors.top: parent.top
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.bottom: vidControlBar.top
                                    fillMode: VideoOutput.PreserveAspectFit
                                }

                                Rectangle {
                                    id: vidControlBar
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.bottom: parent.bottom
                                    height: 42
                                    color: "#181a24"

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.margins: 6
                                        spacing: 8

                                        Button {
                                            text: vidPlayer.playbackState === MediaPlayer.PlayingState ? "⏸" : "▶"
                                            Layout.preferredWidth: 38
                                            Layout.preferredHeight: 30
                                            background: Rectangle { color: "#2563eb"; radius: 4 }
                                            contentItem: Text { text: parent.text; color: "#ffffff"; font.bold: true; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                                            onClicked: {
                                                if (vidPlayer.playbackState === MediaPlayer.PlayingState) {
                                                    vidPlayer.pause()
                                                } else {
                                                    vidPlayer.play()
                                                }
                                            }
                                        }

                                        Slider {
                                            Layout.fillWidth: true
                                            from: 0
                                            to: vidPlayer.duration > 0 ? vidPlayer.duration : 100
                                            value: vidPlayer.position
                                            onMoved: {
                                                vidPlayer.position = value
                                            }
                                        }

                                        Label {
                                            text: formatDuration(vidPlayer.position) + " / " + formatDuration(vidPlayer.duration)
                                            color: "#94a3b8"
                                            font.pixelSize: 11
                                            font.family: "Monospace"
                                        }
                                    }
                                }
                            }
                        }

                        // Embedded Audio Player
                        Repeater {
                            model: extractAudios(model.content)
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.maximumWidth: 500
                                Layout.preferredHeight: 64
                                color: "#161822"
                                radius: 8
                                border.color: "#383d4f"
                                border.width: 1

                                MediaPlayer {
                                    id: sndPlayer
                                    source: modelData.src
                                    audioOutput: AudioOutput { volume: 1.0 }
                                }

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.margins: 10
                                    spacing: 12

                                    Button {
                                        text: sndPlayer.playbackState === MediaPlayer.PlayingState ? "⏸" : "▶ 🔊"
                                        Layout.preferredWidth: 46
                                        Layout.preferredHeight: 36
                                        background: Rectangle {
                                            color: "#8b5cf6"
                                            radius: 6
                                        }
                                        contentItem: Text {
                                            text: parent.text
                                            color: "#ffffff"
                                            font.bold: true
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter
                                        }
                                        onClicked: {
                                            if (sndPlayer.playbackState === MediaPlayer.PlayingState) {
                                                sndPlayer.pause()
                                            } else {
                                                sndPlayer.play()
                                            }
                                        }
                                    }

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 2

                                        Label {
                                            text: "🎵 " + modelData.title
                                            color: "#f8f9fa"
                                            font.bold: true
                                            font.pixelSize: 12
                                        }

                                        RowLayout {
                                            Layout.fillWidth: true
                                            spacing: 6

                                            Slider {
                                                Layout.fillWidth: true
                                                from: 0
                                                to: sndPlayer.duration > 0 ? sndPlayer.duration : 100
                                                value: sndPlayer.position
                                                onMoved: {
                                                    sndPlayer.position = value
                                                }
                                            }

                                            Label {
                                                text: formatDuration(sndPlayer.position) + " / " + formatDuration(sndPlayer.duration)
                                                color: "#94a3b8"
                                                font.pixelSize: 10
                                                font.family: "Monospace"
                                            }
                                        }
                                    }
                                }
                            }
                        }

                    }
                }
            }

            ScrollBar.vertical: ScrollBar {
                active: true
                policy: ScrollBar.AsNeeded
            }
        }

        // --- Status Bar & Progress Tracker ---
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 32
            color: "#16181f"
            border.color: "#222530"
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20

                Label {
                    text: window.isDownloading ? window.downloadStatusMessage : window.statusText
                    font.pixelSize: 12
                    color: window.isDownloading ? "#38bdf8" : (window.isThinking ? "#fbbf24" : "#94a3b8")
                }

                Item { Layout.fillWidth: true }

                // Custom Self-Contained Progress Bar
                Rectangle {
                    visible: window.isDownloading
                    Layout.preferredWidth: 200
                    Layout.preferredHeight: 10
                    color: "#252834"
                    radius: 5

                    Rectangle {
                        height: parent.height
                        width: Math.max(4, parent.width * Math.min(1.0, Math.max(0.0, window.downloadProgress)))
                        color: "#3b82f6"
                        radius: 5
                    }
                }

                Label {
                    text: "Tokens: ~" + window.estimatedTokens
                    font.pixelSize: 12
                    color: "#64748b"
                }
            }
        }

        // --- Bottom Input Area ---
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 96
            color: "#1a1c23"
            border.color: "#282b36"
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 12

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#121316"
                    radius: 8
                    border.color: messageInput.activeFocus ? "#3b82f6" : "#2d3139"
                    border.width: 1

                    ScrollView {
                        anchors.fill: parent
                        anchors.margins: 8
                        clip: true

                        TextArea {
                            id: messageInput
                            placeholderText: "Type your task or question here (Enter to send, Shift+Enter for newline)..."
                            placeholderTextColor: "#64748b"
                            color: "#f8f9fa"
                            font.pixelSize: 14
                            wrapMode: TextEdit.Wrap
                            selectByMouse: true
                            background: null

                            Keys.onReturnPressed: function(event) {
                                if (!(event.modifiers & Qt.ShiftModifier)) {
                                    event.accepted = true
                                    sendAction()
                                }
                            }
                        }
                    }
                }

                Button {
                    id: sendBtn
                    Layout.preferredWidth: 90
                    Layout.fillHeight: true
                    text: "Send ➔"
                    background: Rectangle {
                        color: messageInput.text.trim().length > 0 ? "#2563eb" : "#1e293b"
                        radius: 8
                    }
                    contentItem: Text {
                        text: sendBtn.text
                        color: "#ffffff"
                        font.bold: true
                        font.pixelSize: 14
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    onClicked: sendAction()
                }
            }
        }
    }

    // --- First-Launch / HuggingFace Model Setup Modal Dialog ---
    Dialog {
        id: modelSetupDialog
        title: "🤗 Hugging Face Model & Shards Setup"
        modal: true
        anchors.centerIn: parent
        width: 640
        height: 520
        visible: false

        background: Rectangle {
            color: "#1a1c23"
            radius: 12
            border.color: "#383d4f"
            border.width: 1
        }

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 20
            spacing: 14

            Label {
                text: "✨ In-Process Gemma 4 Model & Projector Setup"
                font.bold: true
                font.pixelSize: 18
                color: "#f8f9fa"
            }

            Label {
                text: "Specify any Hugging Face GGUF repository. Multi-file split shards (e.g. 4 shards for Q8_0) and vision mmproj files are automatically detected, downloaded with resume support, and loaded into memory."
                font.pixelSize: 12
                color: "#94a3b8"
                wrapMode: Text.Wrap
                Layout.fillWidth: true
            }

            // Quick Preset Selection
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 6

                Label {
                    text: "Popular Quantizations & Presets:"
                    color: "#cbd5e1"
                    font.bold: true
                    font.pixelSize: 12
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Button {
                        text: "Gemma 4 26B (Q4_0 1-File)"
                        background: Rectangle { color: "#252834"; radius: 6; border.color: "#383d4f" }
                        contentItem: Text { text: parent.text; color: "#38bdf8"; font.pixelSize: 11; font.bold: true }
                        onClicked: {
                            hfRepoInput.text = "ggml-org/gemma-4-26B-A4B-it-GGUF"
                            hfQuantInput.text = "Q4_0"
                        }
                    }

                    Button {
                        text: "Gemma 4 26B (Q8_0 4-Shards)"
                        background: Rectangle { color: "#252834"; radius: 6; border.color: "#383d4f" }
                        contentItem: Text { text: parent.text; color: "#a78bfa"; font.pixelSize: 11; font.bold: true }
                        onClicked: {
                            hfRepoInput.text = "ggml-org/gemma-4-26B-A4B-it-GGUF"
                            hfQuantInput.text = "Q8_0"
                        }
                    }

                    Button {
                        text: "Gemma 2 9B (Q4_K_M)"
                        background: Rectangle { color: "#252834"; radius: 6; border.color: "#383d4f" }
                        contentItem: Text { text: parent.text; color: "#34d399"; font.pixelSize: 11; font.bold: true }
                        onClicked: {
                            hfRepoInput.text = "google/gemma-2-9b-it-GGUF"
                            hfQuantInput.text = "Q4_K_M"
                        }
                    }
                }
            }

            // HuggingFace Repo Field
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4

                Label {
                    text: "Hugging Face Repository:"
                    font.bold: true
                    color: "#cbd5e1"
                    font.pixelSize: 12
                }

                TextField {
                    id: hfRepoInput
                    text: "ggml-org/gemma-4-26B-A4B-it-GGUF"
                    Layout.fillWidth: true
                    color: "#f8f9fa"
                    font.pixelSize: 13
                    background: Rectangle {
                        color: "#121316"
                        radius: 6
                        border.color: "#383d4f"
                    }
                }
            }

            // Quantization Field
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4

                Label {
                    text: "Quantization Level (e.g. Q4_0, Q8_0, Q4_K_M):"
                    font.bold: true
                    color: "#cbd5e1"
                    font.pixelSize: 12
                }

                TextField {
                    id: hfQuantInput
                    text: "Q4_0"
                    Layout.fillWidth: true
                    color: "#f8f9fa"
                    font.pixelSize: 13
                    background: Rectangle {
                        color: "#121316"
                        radius: 6
                        border.color: "#383d4f"
                    }
                }
            }

            // Multi-File Live Progress Tracker
            ColumnLayout {
                visible: window.isDownloading
                Layout.fillWidth: true
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: window.downloadStatusMessage
                        color: "#38bdf8"
                        font.pixelSize: 12
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }
                    Label {
                        text: Math.round(window.downloadProgress * 100) + "%"
                        color: "#60a5fa"
                        font.bold: true
                        font.pixelSize: 12
                    }
                }

                // Custom Self-Contained Progress Bar
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 12
                    color: "#252834"
                    radius: 6

                    Rectangle {
                        height: parent.height
                        width: Math.max(4, parent.width * Math.min(1.0, Math.max(0.0, window.downloadProgress)))
                        color: "#3b82f6"
                        radius: 6
                    }
                }
            }

            Item { Layout.fillHeight: true }

            // Action Buttons
            RowLayout {
                Layout.fillWidth: true
                spacing: 12

                Button {
                    text: "Cancel"
                    Layout.preferredWidth: 100
                    background: Rectangle { color: "#252834"; radius: 6 }
                    contentItem: Text { text: parent.text; color: "#94a3b8"; font.pixelSize: 13; horizontalAlignment: Text.AlignHCenter }
                    onClicked: {
                        window.isDownloading = false
                        modelSetupDialog.close()
                    }
                }

                Item { Layout.fillWidth: true }

                Button {
                    id: loadModelBtn
                    text: "🚀 Download & Load Model"
                    Layout.preferredWidth: 220
                    background: Rectangle { color: "#2563eb"; radius: 6 }
                    contentItem: Text { text: loadModelBtn.text; color: "#ffffff"; font.bold: true; font.pixelSize: 13; horizontalAlignment: Text.AlignHCenter }
                    onClicked: {
                        startModelDownload(hfRepoInput.text.trim(), hfQuantInput.text.trim())
                    }
                }
            }
        }
    }

    function sendAction() {
        var txt = messageInput.text.trim()
        if (txt.length === 0) return

        appendMessage("user", txt, "", false, "", "", "", false)
        messageInput.text = ""
        window.statusText = "Gemma is thinking..."
        window.isThinking = true
        sendWsCommand({ "type": "send_message", "message": txt })
    }
}
