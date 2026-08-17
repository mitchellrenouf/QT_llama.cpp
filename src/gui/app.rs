use qtrs::prelude::*;
use serde_json::{json, Value};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::Duration,
};
use tungstenite::{connect, stream::MaybeTlsStream, Error as WsError, Message};

pub fn run(port: u16) -> anyhow::Result<()> {
    let app = Application::new();
    let (tx, commands) = mpsc::channel();
    let (events_tx, events) = mpsc::channel();
    thread::spawn(move || worker(port, commands, events_tx));
    let window = Rc::new(
        MainWindow::new()
            .window_title("RustLlama")
            .size(1360, 860)
            .build(),
    );
    let root = Widget::new().parent(&*window).build();
    root.set_style_sheet(APP_STYLE);
    let mut shell_layout = HBoxLayout::with_parent(&root);
    shell_layout.set_contents_margins(0, 0, 0, 0);
    shell_layout.set_spacing(0);

    let sidebar = Widget::new().parent(&root).build();
    sidebar.set_min_size(248, 0);
    sidebar.set_max_size(280, 100_000);
    sidebar.set_style_sheet(SIDEBAR_STYLE);
    let mut sidebar_layout = VBoxLayout::with_parent(&sidebar);
    sidebar_layout.set_contents_margins(16, 18, 16, 16);
    sidebar_layout.set_spacing(12);
    sidebar_layout.add(
        Label::new(
            "<div style='font-size:20px;font-weight:650;color:#f8fafc'>RustLlama</div>\
             <div style='font-size:12px;color:#8b95a7;margin-top:3px'>Local AI workspace</div>",
        )
        .parent(&sidebar)
        .build(),
    );
    let mut new_chat = PushButton::new("＋  New conversation")
        .parent(&sidebar)
        .build();
    {
        let tx = tx.clone();
        new_chat.connect_clicked(move || {
            let _ = tx.send(json!({"type":"clear_history"}));
        });
    }
    sidebar_layout.add(new_chat);
    sidebar_layout.add(Label::new("CONVERSATIONS").parent(&sidebar).build());
    sidebar_layout.add(
        ListWidget::new()
            .items(&["Current conversation"])
            .parent(&sidebar)
            .build(),
    );
    sidebar_layout.add(Label::new("SESSION").parent(&sidebar).build());
    let mut mode = ComboBox::new()
        .items(&["General", "Coding", "Autonomous"])
        .parent(&sidebar)
        .build();
    {
        let tx = tx.clone();
        mode.connect_current_index_changed(move |index| {
            let mode = match index {
                1 => "coder",
                2 => "automatic",
                _ => "general",
            };
            let _ = tx.send(json!({"type":"switch_mode","mode":mode}));
        });
    }
    sidebar_layout.add(mode);
    let mut backend = ComboBox::new()
        .items(&["Auto backend", "CUDA", "Vulkan", "CPU"])
        .parent(&sidebar)
        .build();
    {
        let tx = tx.clone();
        backend.connect_current_index_changed(move |index| {
            let backend = match index {
                1 => "cuda",
                2 => "vulkan",
                3 => "cpu",
                _ => "auto",
            };
            let _ = tx.send(json!({"type":"switch_backend","backend":backend}));
        });
    }
    sidebar_layout.add(backend);
    sidebar_layout.add(Label::new("GENERATION").parent(&sidebar).build());
    let mut temperature = DoubleSpinBox::new()
        .range(0.0, 2.0)
        .value(0.7)
        .decimals(2)
        .suffix("  temperature")
        .parent(&sidebar)
        .build();
    {
        let tx = tx.clone();
        temperature.connect_value_changed(move |value| {
            if let Ok(temperature) = value.trim().parse::<f64>() {
                let _ = tx.send(json!({
                    "type":"update_generation_settings",
                    "temperature":temperature,
                }));
            }
        });
    }
    sidebar_layout.add(temperature);
    let max_tokens_tx = tx.clone();
    let max_tokens = SpinBox::new()
        .range(64, 32_768)
        .value(8192)
        .suffix("  max tokens")
        .on_value_changed(move |value| {
            let _ = max_tokens_tx.send(json!({
                "type":"update_generation_settings",
                "max_tokens":value,
            }));
        })
        .parent(&sidebar)
        .build();
    sidebar_layout.add(max_tokens);
    let mut speech = CheckBox::new("Read replies aloud").parent(&sidebar).build();
    {
        let tx = tx.clone();
        speech.connect_toggled(move |_| {
            let _ = tx.send(json!({"type":"toggle_speech"}));
        });
    }
    sidebar_layout.add(speech);
    let mut local_model = PushButton::new("Open local GGUF…").parent(&sidebar).build();
    {
        let tx = tx.clone();
        local_model.connect_clicked(move || {
            if let Some(path) =
                FileDialog::open_file(None, "Open a GGUF model", "", "GGUF model (*.gguf)")
            {
                let _ = tx.send(json!({"type":"load_local_model","path":path}));
            }
        });
    }
    sidebar_layout.add(local_model);
    sidebar_layout.add(Label::new("⌁  Local model · CUDA").parent(&sidebar).build());
    shell_layout.add(sidebar);

    let workspace = Widget::new().parent(&root).build();
    let mut layout = VBoxLayout::with_parent(&workspace);
    layout.set_contents_margins(32, 24, 32, 18);
    layout.set_spacing(12);
    layout.add(
        Label::new(
            "<div style='font-size:14px;font-weight:600;color:#f8fafc'>New conversation</div>\
             <div style='font-size:12px;color:#8b95a7;margin-top:3px'>General · Local execution</div>",
        )
        .parent(&workspace)
        .build(),
    );
    let chat = Rc::new(RefCell::new(
        TextBrowser::new()
            .html(welcome())
            .parent(&workspace)
            .build(),
    ));
    chat.borrow().set_open_external_links(true);
    chat.borrow_mut().set_has_parent();
    let chat_area = ScrollArea::new().set_widget_resizable(true).build();
    chat_area.set_widget(&*chat.borrow());
    layout.add(chat_area);
    let composer = Widget::new().parent(&workspace).build();
    composer.set_min_size(0, 106);
    composer.set_max_size(100_000, 132);
    composer.set_style_sheet(
        "QWidget { background: #17191f; border: 1px solid #303744; border-radius: 14px; }",
    );
    let mut composer_layout = HBoxLayout::with_parent(&composer);
    composer_layout.set_contents_margins(12, 10, 12, 10);
    composer_layout.set_spacing(10);
    let input = Rc::new(RefCell::new(PlainTextEdit::new().parent(&composer).build()));
    input.borrow().set_placeholder_text("Message RustLlama…");
    input.borrow_mut().set_has_parent();
    let input_area = ScrollArea::new().set_widget_resizable(true).build();
    input_area.set_widget(&*input.borrow());
    composer_layout.add(input_area);
    let mut send = PushButton::new("↑").parent(&composer).build();
    let mut download = PushButton::new("⌄").parent(&composer).build();
    layout.add(composer);
    let status_box = Widget::new().parent(&workspace).build();
    status_box.set_min_size(0, 26);
    status_box.set_max_size(100_000, 30);
    let mut status_layout = HBoxLayout::with_parent(&status_box);
    status_layout.set_contents_margins(4, 0, 4, 0);
    let status = Rc::new(RefCell::new(
        Label::new("Connecting to local agent…")
            .parent(&status_box)
            .build(),
    ));
    status.borrow_mut().set_has_parent();
    let status_area = ScrollArea::new().set_widget_resizable(true).build();
    status_area.set_widget(&*status.borrow());
    status_layout.add(status_area);
    layout.add(status_box);
    shell_layout.add(workspace);
    window.set_central_widget(&root);
    let transcript = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let streaming = Rc::new(RefCell::new(String::new()));
    let busy = Rc::new(RefCell::new(false));
    {
        let input = Rc::clone(&input);
        let chat = Rc::clone(&chat);
        let transcript = Rc::clone(&transcript);
        let streaming = Rc::clone(&streaming);
        let busy = Rc::clone(&busy);
        let status = Rc::clone(&status);
        let tx = tx.clone();
        send.connect_clicked(move || {
            let text = input.borrow().plain_text().trim().to_owned();
            if text.is_empty() || *busy.borrow() {
                return;
            }
            transcript.borrow_mut().push(("You".into(), text.clone()));
            input.borrow().clear();
            streaming.borrow_mut().clear();
            *busy.borrow_mut() = true;
            chat.borrow().set_html(&render(&transcript.borrow(), ""));
            status.borrow_mut().set_text("RustLlama is thinking…");
            let _ = tx.send(json!({"type":"send_message","message":text}));
        });
    }
    {
        let input = Rc::clone(&input);
        let status = Rc::clone(&status);
        let tx = tx.clone();
        download.connect_clicked(move || {
            let spec = input.borrow().plain_text().trim().to_owned();
            if spec.is_empty() {
                status
                    .borrow_mut()
                    .set_text("Enter a Hugging Face model spec first.");
            } else {
                let _ = tx.send(json!({"type":"load_hf_model","spec":spec}));
                status.borrow_mut().set_text("Preparing model download…");
            }
        });
    }
    composer_layout.add(send);
    composer_layout.add(download);
    let _timer = Timer::new(40)
        .on_timeout({
            let chat = Rc::clone(&chat);
            let transcript = Rc::clone(&transcript);
            let streaming = Rc::clone(&streaming);
            let busy = Rc::clone(&busy);
            let status = Rc::clone(&status);
            move || {
                while let Ok(e) = events.try_recv() {
                    match e["type"].as_str().unwrap_or("") {
                        "init_state" => status.borrow_mut().set_text(
                            if e["model_loaded"].as_bool().unwrap_or(false) {
                                "Ready."
                            } else {
                                "No model loaded. Enter a Hugging Face model spec to begin."
                            },
                        ),
                        "stream_token" => {
                            streaming
                                .borrow_mut()
                                .push_str(e["token"].as_str().unwrap_or(""));
                            chat.borrow()
                                .set_html(&render(&transcript.borrow(), &streaming.borrow()));
                        }
                        "stream_thought" => status
                            .borrow_mut()
                            .set_text(format!("Thinking: {}", e["thought"].as_str().unwrap_or(""))),
                        "tool_started" => status
                            .borrow_mut()
                            .set_text(format!("Running {}…", e["name"].as_str().unwrap_or("tool"))),
                        "download_progress" => {
                            status.borrow_mut().set_text(format!(
                                "{} · {:.0}%",
                                e["message"].as_str().unwrap_or("Downloading model"),
                                e["progress"].as_f64().unwrap_or(0.0) * 100.0
                            ));
                        }
                        "turn_done" => {
                            let answer = if streaming.borrow().is_empty() {
                                e["content"].as_str().unwrap_or("").to_owned()
                            } else {
                                std::mem::take(&mut *streaming.borrow_mut())
                            };
                            if !answer.is_empty() {
                                transcript.borrow_mut().push(("RustLlama".into(), answer));
                            }
                            *busy.borrow_mut() = false;
                            chat.borrow().set_html(&render(&transcript.borrow(), ""));
                            status.borrow_mut().set_text(format!(
                                "Ready · ~{} tokens",
                                e["tokens"].as_u64().unwrap_or(0)
                            ));
                        }
                        "history_cleared" => {
                            transcript.borrow_mut().clear();
                            chat.borrow().set_html(&welcome());
                        }
                        "model_loaded" => status.borrow_mut().set_text(format!(
                            "Model loaded: {}",
                            e["model_name"].as_str().unwrap_or("")
                        )),
                        "error" => {
                            *busy.borrow_mut() = false;
                            status.borrow_mut().set_text(format!(
                                "Error: {}",
                                e["message"].as_str().unwrap_or("")
                            ));
                        }
                        _ => {}
                    }
                }
            }
        })
        .build();
    let file = Menu::new("File")
        .action("Clear conversation", {
            let tx = tx.clone();
            move || {
                let _ = tx.send(json!({"type":"clear_history"}));
            }
        })
        .action("Quit", || std::process::exit(0))
        .parent(&*window)
        .build();
    let help = Menu::new("Help")
        .action("About RustLlama", || {
            qtrs::messagebox::information(
                None,
                "RustLlama",
                "RustLlama\nA native Qt 6 desktop AI agent, authored in Rust.",
            )
        })
        .parent(&*window)
        .build();
    let menu = MenuBar::new()
        .add_menu(file)
        .add_menu(help)
        .parent(&*window)
        .build();
    window.set_menu_bar(&menu);
    window.show();
    app.exec();
    Ok(())
}
fn worker(port: u16, commands: Receiver<Value>, events: Sender<Value>) {
    let Ok((mut s, _)) = connect(format!("ws://127.0.0.1:{port}")) else {
        let _ = events.send(json!({"type":"error","message":"Could not connect to local agent"}));
        return;
    };
    if let MaybeTlsStream::Plain(tcp) = s.get_mut() {
        if let Err(e) = tcp.set_nonblocking(true) {
            let _ = events.send(json!({"type":"error","message":format!("Could not configure local connection: {e}")}));
            return;
        }
    }
    loop {
        while let Ok(c) = commands.try_recv() {
            if s.send(Message::Text(c.to_string().into())).is_err() {
                return;
            }
        }
        match s.read() {
            Ok(Message::Text(t)) => {
                if let Ok(v) = serde_json::from_str(&t) {
                    let _ = events.send(v);
                }
            }
            Ok(Message::Close(_)) => return,
            Err(WsError::Io(e))
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(_) => return,
            _ => {}
        }
        thread::sleep(Duration::from_millis(8));
    }
}
fn welcome() -> String {
    "<div style='max-width:680px;margin:90px auto;color:#e5e7eb'>\
       <div style='font-size:28px;font-weight:600;margin-bottom:12px'>What can I help you with?</div>\
       <div style='font-size:15px;color:#94a3b8;line-height:1.55'>RustLlama is running locally. Ask a question, write some code, or load another GGUF model from Hugging Face.</div>\
     </div>".into()
}
fn render(items: &[(String, String)], live: &str) -> String {
    let mut h = String::new();
    for (a, b) in items {
        let (accent, heading, alignment) = if a == "You" {
            ("#1d4ed8", "You", "margin-left:18%;margin-right:2%")
        } else {
            ("#202633", "RustLlama", "margin-right:18%;margin-left:2%")
        };
        h.push_str(&format!(
            "<div style='{alignment};margin-top:14px;margin-bottom:14px;padding:14px 16px;border-radius:12px;background:{accent};color:#f8fafc;line-height:1.55'>\
               <div style='font-size:12px;font-weight:600;color:#cbd5e1;margin-bottom:6px'>{heading}</div>{}</div>",
            b.replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('\n', "<br>")
        ));
    }
    if !live.is_empty() {
        h.push_str(&format!(
            "<div style='margin-right:18%;margin-left:2%;margin-top:14px;padding:14px 16px;border-radius:12px;background:#202633;color:#f8fafc;line-height:1.55'>\
               <div style='font-size:12px;font-weight:600;color:#cbd5e1;margin-bottom:6px'>RustLlama · generating</div>{}</div>",
            live.replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('\n', "<br>")
        ));
    }
    h
}

const APP_STYLE: &str = r#"
    QWidget { background: #111318; color: #e5e7eb; font-family: "Segoe UI", "SF Pro Text", sans-serif; }
    QScrollArea { border: 0; background: transparent; }
    QTextBrowser { background: #111318; border: 0; padding: 14px 54px; color: #e5e7eb; selection-background-color: #355ca8; }
    QPlainTextEdit { background: transparent; border: 0; color: #f3f4f6; padding: 7px; font-size: 14px; }
    QPushButton { background:#242934; color:#dbe4f0; border:1px solid #3a4352; border-radius:9px; padding:9px 13px; }
    QPushButton:hover { background:#303846; border-color:#566174; }
    QPushButton:disabled { background:#334155; color:#94a3b8; }
    QScrollBar:vertical { background:#17191f; width:10px; margin:4px; }
    QScrollBar::handle:vertical { background:#3a4352; border-radius:5px; min-height:28px; }
    QMenuBar { background:#17191f; color:#e5e7eb; padding:4px 8px; }
    QMenuBar::item:selected, QMenu::item:selected { background:#252b36; border-radius:5px; }
    QMenu { background:#17191f; color:#e5e7eb; border:1px solid #303744; }
    QLabel { color:#cbd5e1; }
"#;

const SIDEBAR_STYLE: &str = r#"
    QWidget { background:#1b1d23; color:#e5e7eb; }
    QPushButton { text-align:left; background:#252831; color:#f1f5f9; border:1px solid #353a46; border-radius:8px; padding:10px 12px; font-weight:600; }
    QPushButton:hover { background:#30343e; }
    QListWidget { background:transparent; color:#d6dbe4; border:0; outline:0; padding:0; }
    QListWidget::item { padding:9px 10px; border-radius:7px; }
    QListWidget::item:selected, QListWidget::item:hover { background:#30323a; }
    QComboBox { background:#22252d; color:#dbe4f0; border:1px solid #343945; border-radius:7px; padding:7px 9px; }
    QComboBox:hover { border-color:#566174; }
    QComboBox QAbstractItemView { background:#252831; color:#e5e7eb; selection-background-color:#353945; }
    QCheckBox { color:#c6cedb; padding:4px 2px; }
    QCheckBox::indicator { width:15px; height:15px; border:1px solid #566174; border-radius:4px; background:#22252d; }
    QCheckBox::indicator:checked { background:#4f7cff; border-color:#4f7cff; }
    QLabel { color:#8f98a8; font-size:12px; }
"#;
