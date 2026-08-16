import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

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
    property string statusText: "Ready"

    signal sendMessageRequested(string message)
    signal switchModeRequested(string mode)
    signal toggleSpeechRequested()
    signal clearHistoryRequested()

    function appendMessage(role, content, isThinkingBlock, toolName, toolArgs, toolResult) {
        chatModel.append({
            "role": role,
            "content": content,
            "isThinking": isThinkingBlock || false,
            "toolName": toolName || "",
            "toolArgs": toolArgs || "",
            "toolResult": toolResult || ""
        })
        chatListView.positionViewAtEnd()
    }

    function appendStreamToken(token) {
        if (chatModel.count > 0) {
            var lastItem = chatModel.get(chatModel.count - 1)
            if (lastItem.role === "assistant" && !lastItem.isThinking) {
                lastItem.content += token
                chatListView.positionViewAtEnd()
                return
            }
        }
        appendMessage("assistant", token, false, "", "", "")
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

                Label {
                    text: "Freedesktop Linux / Qt6"
                    font.pixelSize: 12
                    color: "#8a909e"
                    Layout.alignment: Qt.AlignVCenter
                }

                Item { Layout.fillWidth: true }

                // Mode Selector
                ComboBox {
                    id: modeCombo
                    model: ["General", "Coder", "Automatic"]
                    currentIndex: 0
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
                        window.switchModeRequested(m)
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
                        window.speechEnabled = !window.speechEnabled
                        window.toggleSpeechRequested()
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
                        chatModel.clear()
                        window.clearHistoryRequested()
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
                width: chatListView.width - 40
                spacing: 4

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
                        text: model.role === "user" ? "You" : (model.toolName !== "" ? "Tool: " + model.toolName : "Gemma 4")
                        font.bold: true
                        font.pixelSize: 13
                        color: model.role === "user" ? "#38bdf8" : (model.toolName !== "" ? "#fbbf24" : "#c084fc")
                    }
                }

                // Message Bubble
                Rectangle {
                    Layout.fillWidth: true
                    radius: 8
                    color: model.role === "user" ? "#1e293b" : (model.isThinking ? "#1c1917" : (model.toolName !== "" ? "#1e1e24" : "#181a20"))
                    border.color: model.role === "user" ? "#334155" : (model.isThinking ? "#44403c" : (model.toolName !== "" ? "#3f3f46" : "#27272a"))
                    border.width: 1

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 8

                        // Thought block tag if thinking
                        Label {
                            visible: model.isThinking
                            text: "🧠 Thinking Chain"
                            color: "#a8a29e"
                            font.italic: true
                            font.pixelSize: 11
                        }

                        // Main Text Content
                        TextEdit {
                            Layout.fillWidth: true
                            text: model.content
                            color: model.isThinking ? "#d6d3d1" : "#f1f5f9"
                            font.pixelSize: 14
                            font.family: "Sans Serif"
                            wrapMode: TextEdit.Wrap
                            readOnly: true
                            selectByMouse: true
                            textFormat: TextEdit.RichText
                        }

                        // Tool Arguments & Output if applicable
                        ColumnLayout {
                            visible: model.toolName !== ""
                            Layout.fillWidth: true
                            spacing: 4

                            Label {
                                text: "Arguments: " + model.toolArgs
                                color: "#9ca3af"
                                font.pixelSize: 11
                                font.family: "Monospace"
                                wrapMode: Text.Wrap
                            }

                            Label {
                                text: "Result: " + model.toolResult
                                color: "#86efac"
                                font.pixelSize: 11
                                font.family: "Monospace"
                                wrapMode: Text.Wrap
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

        // --- Status Bar ---
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 28
            color: "#16181f"
            border.color: "#222530"
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20

                Label {
                    text: window.statusText
                    font.pixelSize: 12
                    color: window.isThinking ? "#fbbf24" : "#94a3b8"
                }

                Item { Layout.fillWidth: true }

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

                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    TextArea {
                        id: messageInput
                        placeholderText: "Type your task or question here (Enter to send, Shift+Enter for newline)..."
                        placeholderTextColor: "#64748b"
                        color: "#f8f9fa"
                        font.pixelSize: 14
                        wrapMode: TextArea.Wrap
                        background: Rectangle {
                            color: "#121316"
                            radius: 8
                            border.color: messageInput.activeFocus ? "#3b82f6" : "#2d3139"
                            border.width: 1
                        }

                        Keys.onReturnPressed: function(event) {
                            if (!(event.modifiers & Qt.ShiftModifier)) {
                                event.accepted = true
                                sendAction()
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

    function sendAction() {
        var txt = messageInput.text.trim()
        if (txt.length === 0) return

        appendMessage("user", txt, false, "", "", "")
        messageInput.text = ""
        window.statusText = "Gemma is processing..."
        window.isThinking = true
        window.sendMessageRequested(txt)
    }
}
