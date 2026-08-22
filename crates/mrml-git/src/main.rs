#![no_std]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(test, allow(dead_code))]

use mrml_error::{Context, Result, anyhow};
use mrml_git::{Change, parse_porcelain};
use mrml_runtime::{
    Command, Text, Vector, mrml_format as format, mrml_print as print, mrml_println as println,
};
use mrml_terminal_style::Colorize;

fn run(args: &[&str]) -> Result<Text> {
    let output = Command::new("git").args(args.iter().copied()).output()?;
    if !output.status.success() {
        let error = Text::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "git {} failed: {}",
            args.first().copied().unwrap_or("command"),
            error.trim()
        ));
    }
    Ok(Text::from_utf8_lossy(&output.stdout))
}

fn status() -> Result<Vector<Change>> {
    let output = Command::new("git")
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .output()?;
    if !output.status.success() {
        return Err(anyhow!("not a Git repository"));
    }
    Ok(parse_porcelain(&output.stdout))
}

fn print_changes(changes: &[Change]) {
    if changes.is_empty() {
        println!(
            "  {}",
            "clean — nothing between you and the next idea".green()
        );
        return;
    }
    for change in changes {
        let lane = match (change.staged(), change.unstaged()) {
            (true, true) => "both ".bright_yellow(),
            (true, false) => "stage".bright_green(),
            _ => "work ".red(),
        };
        let state = change.state().label();
        if let Some(old) = &change.original_path {
            println!(
                "  {}  {:9} {} → {}",
                lane,
                state,
                old.dimmed(),
                change.path.bright_white()
            );
        } else {
            println!("  {}  {:9} {}", lane, state, change.path.bright_white());
        }
    }
}

fn dashboard() -> Result<()> {
    let branch = run(&["branch", "--show-current"])?;
    let branch = if branch.trim().is_empty() {
        "detached HEAD"
    } else {
        branch.trim()
    };
    let head =
        run(&["log", "-1", "--format=%h  %s  %cr"]).unwrap_or_else(|_| "no commits yet".into());
    let changes = status()?;
    let staged = changes.iter().filter(|item| item.staged()).count();
    let unstaged = changes.iter().filter(|item| item.unstaged()).count();
    println!(
        "{}",
        "╭─ MRML GIT · WORKSPACE PULSE ─────────────────────────╮"
            .bright_cyan()
            .bold()
    );
    println!("  {} {}", "branch".dimmed(), branch.magenta().bold());
    println!("  {}   {}", "head".dimmed(), head.trim());
    println!(
        "  {}  {} staged  {} unstaged  {} total",
        "pulse".dimmed(),
        format!("{}", staged).green(),
        format!("{}", unstaged).red(),
        changes.len()
    );
    println!(
        "{}",
        "├─ CHANGES ────────────────────────────────────────────┤".bright_cyan()
    );
    print_changes(&changes);
    println!(
        "{}",
        "╰──────────────────────────────────────────────────────╯".bright_cyan()
    );
    Ok(())
}

fn help() {
    println!(
        "{}\n\n  mrml-git [status]\n  mrml-git log [count]\n  mrml-git diff [--staged] [path]\n  mrml-git branch [name]\n  mrml-git switch <name>\n  mrml-git stage <path>...\n  mrml-git unstage <path>...\n  mrml-git commit <message>\n\nWith no command, shows the workspace pulse dashboard.",
        "MRML GIT".bright_cyan().bold()
    );
}

fn passthrough(command: &str, fixed: &[&str], tail: &[Text]) -> Result<()> {
    let mut args = Vector::new();
    args.push(command);
    args.extend(fixed.iter().copied());
    args.extend(tail.iter().map(Text::as_str));
    let output = run(&args)?;
    print!("{}", output);
    Ok(())
}

fn application_main() -> Result<()> {
    let args = mrml_runtime::command_arguments();
    let tail = args.get(2..).unwrap_or(&[]);
    match args.get(1).map(Text::as_str) {
        None | Some("status") | Some("pulse") => dashboard(),
        Some("help" | "--help" | "-h") => { help(); Ok(()) }
        Some("--version" | "-V") => { println!("{}", env!("CARGO_PKG_VERSION")); Ok(()) }
        Some("log") => {
            let count = tail.first().map(Text::as_str).unwrap_or("12");
            passthrough("log", &["--graph", "--decorate", "--date=relative", "--pretty=format:%C(auto)%h%Creset %C(magenta)%d%Creset %s %C(dim white)— %an, %ar%Creset", "-n", count], &[])
        }
        Some("diff") => {
            let mut options = Vector::new();
            for arg in tail { if arg == "--staged" { options.push("--cached".into()); } else { options.push(arg.clone()); } }
            passthrough("diff", &["--color=always"], &options)
        }
        Some("branch") if tail.is_empty() => passthrough("branch", &["--all", "--verbose"], &[]),
        Some("branch") => passthrough("switch", &["-c"], tail),
        Some("switch") => passthrough("switch", &[], tail),
        Some("stage") => { if tail.is_empty() { return Err(anyhow!("stage requires at least one path")); } passthrough("add", &["--"], tail) }
        Some("unstage") => { if tail.is_empty() { return Err(anyhow!("unstage requires at least one path")); } passthrough("restore", &["--staged", "--"], tail) }
        Some("commit") => {
            if tail.is_empty() { return Err(anyhow!("commit requires a message")); }
            let mut message = Text::new();
            for (index, word) in tail.iter().enumerate() { if index > 0 { message.push(' '); } message.push_str(word); }
            passthrough("commit", &["-m", &message], &[])
        }
        Some(other) => Err(anyhow!(
            "unknown command '{}'; run mrml-git help",
            other
        )),
    }.context("mrml-git")
}

mrml_runtime::mrml_entrypoint!(application_main);
