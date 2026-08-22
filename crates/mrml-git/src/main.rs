#![no_std]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(test, allow(dead_code))]

use mrml_error::{Context, Result, anyhow};
use mrml_git::{Change, Cli, parse_porcelain, validate_positional};
use mrml_runtime::{
    Command, Output, Text, Vector, mrml_format as format, mrml_print as print,
    mrml_println as println,
};
use mrml_terminal_style::Colorize;

fn git_output(repository: Option<&str>, args: &[&str]) -> Result<Output> {
    let mut command = Command::new("git");
    command.args(args.iter().copied());
    if let Some(path) = repository {
        command.current_dir(path);
    }
    command.output().map_err(Into::into)
}

fn run(repository: Option<&str>, args: &[&str]) -> Result<Text> {
    let output = git_output(repository, args)?;
    if !output.status.success() {
        return Err(git_failure(args, &output));
    }
    Ok(Text::from_utf8_lossy(&output.stdout))
}

fn git_failure(args: &[&str], output: &Output) -> mrml_error::Error {
    let stderr = Text::from_utf8_lossy(&output.stderr);
    let stdout = Text::from_utf8_lossy(&output.stdout);
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    anyhow!(
        "git {} failed: {}",
        args.first().copied().unwrap_or("command"),
        detail
    )
}

fn run_visible(repository: Option<&str>, args: &[&str]) -> Result<()> {
    let output = git_output(repository, args)?;
    if !output.status.success() {
        return Err(git_failure(args, &output));
    }
    let stdout = Text::from_utf8_lossy(&output.stdout);
    let stderr = Text::from_utf8_lossy(&output.stderr);
    if !stdout.is_empty() {
        print!("{}", stdout);
    }
    if !stderr.is_empty() {
        print!("{}", stderr);
    }
    Ok(())
}

fn status(repository: Option<&str>) -> Result<Vector<Change>> {
    let output = git_output(
        repository,
        &["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )?;
    if !output.status.success() {
        return Err(anyhow!("not a Git repository"));
    }
    Ok(parse_porcelain(&output.stdout))
}

fn print_changes(changes: &[Change]) {
    if changes.is_empty() {
        println!("  {}", "clean — ready for the next idea".green());
        return;
    }
    for change in changes {
        let lane = match (change.staged(), change.unstaged()) {
            (true, true) => "both ".bright_yellow(),
            (true, false) => "stage".bright_green(),
            _ => "work ".red(),
        };
        if let Some(old) = &change.original_path {
            println!(
                "  {}  {:9} {} → {}",
                lane,
                change.state().label(),
                old.dimmed(),
                change.path.bright_white()
            );
        } else {
            println!(
                "  {}  {:9} {}",
                lane,
                change.state().label(),
                change.path.bright_white()
            );
        }
    }
}

fn upstream_counts(repository: Option<&str>) -> Option<(Text, Text)> {
    let upstream = run(
        repository,
        &[
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ],
    )
    .ok()?;
    let counts = run(
        repository,
        &["rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
    )
    .ok()?;
    Some((upstream.trim().into(), counts.trim().into()))
}

fn dashboard(repository: Option<&str>) -> Result<()> {
    let root = run(repository, &["rev-parse", "--show-toplevel"])?;
    let branch = run(repository, &["branch", "--show-current"])?;
    let branch = if branch.trim().is_empty() {
        "detached HEAD"
    } else {
        branch.trim()
    };
    let head = run(repository, &["log", "-1", "--format=%h  %s  %cr"])
        .unwrap_or_else(|_| "no commits yet".into());
    let changes = status(repository)?;
    let staged = changes.iter().filter(|item| item.staged()).count();
    let unstaged = changes.iter().filter(|item| item.unstaged()).count();
    println!(
        "{}",
        "╭─ MRML GIT · WORKSPACE PULSE ─────────────────────────╮"
            .bright_cyan()
            .bold()
    );
    println!("  {}   {}", "root".dimmed(), root.trim());
    println!("  {} {}", "branch".dimmed(), branch.magenta().bold());
    println!("  {}   {}", "head".dimmed(), head.trim());
    if let Some((upstream, counts)) = upstream_counts(repository) {
        let mut parts = counts.split_whitespace();
        println!(
            "  {} {}  {} ahead  {} behind",
            "track".dimmed(),
            upstream,
            parts.next().unwrap_or("0").green(),
            parts.next().unwrap_or("0").yellow()
        );
    } else {
        println!("  {} {}", "track".dimmed(), "no upstream".yellow());
    }
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
        "{}\n\n  mrml-git [-C PATH] [status]\n  mrml-git [-C PATH] doctor\n  mrml-git [-C PATH] init [path]\n  mrml-git [-C PATH] clone <url> [path]\n  mrml-git [-C PATH] log [count]\n  mrml-git [-C PATH] diff [--staged] [args...]\n  mrml-git [-C PATH] show [revision]\n  mrml-git [-C PATH] branch [new-name]\n  mrml-git [-C PATH] switch <name>\n  mrml-git [-C PATH] stage <path>...\n  mrml-git [-C PATH] unstage <path>...\n  mrml-git [-C PATH] restore <path>...\n  mrml-git [-C PATH] commit <message>\n  mrml-git [-C PATH] fetch [remote]\n  mrml-git [-C PATH] pull [remote] [branch]\n  mrml-git [-C PATH] push [remote] [branch]\n  mrml-git [-C PATH] remote\n  mrml-git [-C PATH] tag [name]\n  mrml-git [-C PATH] stash [list|push [message]|pop]\n\nPulls are fast-forward only. No command shows the workspace pulse.",
        "MRML GIT".bright_cyan().bold()
    );
}

fn require_arguments<'a>(command: &str, arguments: &'a [Text]) -> Result<&'a [Text]> {
    if arguments.is_empty() {
        Err(anyhow!("{} requires at least one argument", command))
    } else {
        Ok(arguments)
    }
}

fn collect<'a>(command: &'a str, fixed: &[&'a str], tail: &'a [Text]) -> Vector<&'a str> {
    let mut args = Vector::from([command]);
    args.extend(fixed.iter().copied());
    args.extend(tail.iter().map(Text::as_str));
    args
}

fn join_words(words: &[Text]) -> Text {
    let mut output = Text::new();
    for (index, word) in words.iter().enumerate() {
        if index > 0 {
            output.push(' ');
        }
        output.push_str(word);
    }
    output
}

fn checked_positionals(values: &[Text]) -> Result<&[Text]> {
    validate_positional(values).map_err(|error| anyhow!("{}", error))?;
    Ok(values)
}

fn dispatch(cli: &Cli) -> Result<()> {
    let repository = cli.repository.as_deref();
    let tail: &[Text] = &cli.arguments;
    match cli.command.as_str() {
        "status" | "pulse" if tail.is_empty() => dashboard(repository),
        "help" | "--help" | "-h" if tail.is_empty() => {
            help();
            Ok(())
        }
        "--version" | "-V" if tail.is_empty() => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "doctor" if tail.is_empty() => {
            let version = run(repository, &["--version"])?;
            println!("{} {}", "git".green().bold(), version.trim());
            match run(repository, &["rev-parse", "--show-toplevel"]) {
                Ok(root) => println!("{} {}", "repository".green().bold(), root.trim()),
                Err(_) => println!("{} not inside a repository", "repository".yellow().bold()),
            }
            Ok(())
        }
        "init" if tail.len() <= 1 => {
            checked_positionals(tail)?;
            run_visible(repository, &collect("init", &["--"], tail))
        }
        "clone" if matches!(tail.len(), 1 | 2) => {
            checked_positionals(tail)?;
            run_visible(repository, &collect("clone", &["--"], tail))
        }
        "log" => {
            let count = tail.first().map(Text::as_str).unwrap_or("12");
            if tail.len() > 1 || count.parse::<usize>().is_err() {
                return Err(anyhow!("log accepts one numeric count"));
            }
            run_visible(
                repository,
                &[
                    "log",
                    "--graph",
                    "--decorate",
                    "--date=relative",
                    "--pretty=format:%C(auto)%h%Creset %C(magenta)%d%Creset %s %C(dim white)— %an, %ar%Creset",
                    "-n",
                    count,
                ],
            )
        }
        "diff" => {
            let mut args = Vector::from(["diff", "--color=always"]);
            args.extend(tail.iter().map(|arg| {
                if arg == "--staged" {
                    "--cached"
                } else {
                    arg.as_str()
                }
            }));
            run_visible(repository, &args)
        }
        "show" if tail.len() <= 1 => {
            checked_positionals(tail)?;
            run_visible(
                repository,
                &[
                    "show",
                    "--color=always",
                    "--stat",
                    tail.first().map(Text::as_str).unwrap_or("HEAD"),
                    "--",
                ],
            )
        }
        "branch" if tail.is_empty() => run_visible(repository, &["branch", "--all", "--verbose"]),
        "branch" if tail.len() == 1 => {
            checked_positionals(tail)?;
            run_visible(repository, &["switch", "-c", &tail[0]])
        }
        "switch" if tail.len() == 1 => {
            checked_positionals(tail)?;
            run_visible(repository, &["switch", &tail[0]])
        }
        "stage" => run_visible(
            repository,
            &collect("add", &["--"], require_arguments("stage", tail)?),
        ),
        "unstage" => run_visible(
            repository,
            &collect(
                "restore",
                &["--staged", "--"],
                require_arguments("unstage", tail)?,
            ),
        ),
        "restore" => run_visible(
            repository,
            &collect(
                "restore",
                &["--worktree", "--"],
                require_arguments("restore", tail)?,
            ),
        ),
        "commit" => {
            require_arguments("commit", tail)?;
            let message = join_words(tail);
            run_visible(repository, &["commit", "-m", &message])
        }
        "fetch" if tail.len() <= 1 => {
            checked_positionals(tail)?;
            run_visible(repository, &collect("fetch", &["--prune"], tail))
        }
        "pull" if tail.len() <= 2 => {
            checked_positionals(tail)?;
            run_visible(repository, &collect("pull", &["--ff-only"], tail))
        }
        "push" if tail.len() <= 2 => {
            checked_positionals(tail)?;
            run_visible(repository, &collect("push", &[], tail))
        }
        "remote" if tail.is_empty() => run_visible(repository, &["remote", "--verbose"]),
        "tag" if tail.is_empty() => run_visible(repository, &["tag"]),
        "tag" if tail.len() == 1 => {
            checked_positionals(tail)?;
            run_visible(repository, &collect("tag", &["--"], tail))
        }
        "stash" if tail.is_empty() || (tail.len() == 1 && tail[0] == "list") => {
            run_visible(repository, &["stash", "list"])
        }
        "stash" if tail.len() == 1 && tail[0] == "pop" => {
            run_visible(repository, &["stash", "pop"])
        }
        "stash" if !tail.is_empty() && tail[0] == "push" => {
            let message = join_words(&tail[1..]);
            if message.is_empty() {
                run_visible(repository, &["stash", "push"])
            } else {
                run_visible(repository, &["stash", "push", "-m", &message])
            }
        }
        command => Err(anyhow!(
            "invalid arguments for '{}'; run mrml-git help",
            command
        )),
    }
}

fn application_main() -> Result<()> {
    let cli =
        Cli::parse(mrml_runtime::command_arguments()).map_err(|error| anyhow!("{}", error))?;
    dispatch(&cli).context("mrml-git")
}

mrml_runtime::mrml_entrypoint!(application_main);
