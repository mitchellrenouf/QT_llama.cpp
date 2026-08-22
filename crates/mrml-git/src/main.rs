#![no_std]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(test, allow(dead_code))]

use mrml_error::{Context, Result, anyhow};
use mrml_git::{Change, Cli, parse_porcelain, validate_positional};
use mrml_runtime::{
    Command, Output, Text, Vector, mrml_format as format, mrml_print as print,
    mrml_println as println,
};
use mrml_ssh::SshRemote;
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
    println!("{}\n", "MRML GIT".bright_cyan().bold());
    for usage in [
        "[-C PATH] [status]",
        "[-C PATH] doctor",
        "[-C PATH] init [path]",
        "[-C PATH] clone <url> [path]",
        "[-C PATH] log [count]",
        "[-C PATH] diff [--staged] [args...]",
        "[-C PATH] show [revision]",
        "[-C PATH] branch [new-name]",
        "[-C PATH] switch <name>",
        "[-C PATH] stage <path>...",
        "[-C PATH] unstage <path>...",
        "[-C PATH] restore <path>...",
        "[-C PATH] commit [--sign] <message>",
        "[-C PATH] fetch [remote]",
        "[-C PATH] pull [remote] [branch]",
        "[-C PATH] push [remote] [branch]",
        "[-C PATH] remote",
        "[-C PATH] ssh <add|set|info|check> ...",
        "[-C PATH] signing <configure|auto|status|off|verify|verify-tag> ...",
        "[-C PATH] tag [name]",
        "[-C PATH] tag-sign <name> <message>",
        "[-C PATH] stash [list|push [message]|pop]",
    ] {
        println!("  mrml-git {}", usage);
    }
    println!(
        "\nPulls are fast-forward only. SSH signing is repository-local. No command shows the workspace pulse."
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

fn ssh_remote(repository: Option<&str>, name: &str) -> Result<(Text, SshRemote)> {
    let url = run(repository, &["remote", "get-url", "--", name])?;
    let url: Text = url.trim().into();
    let parsed =
        SshRemote::parse(&url).map_err(|error| anyhow!("invalid SSH remote: {}", error))?;
    Ok((url, parsed))
}

fn print_ssh_remote(name: &str, url: &str, remote: &SshRemote) {
    println!("{} {}", "remote".green().bold(), name);
    println!("{} {}", "url".green().bold(), url);
    println!("{} {}", "destination".green().bold(), remote.destination());
    println!("{} {}", "path".green().bold(), remote.path);
    println!(
        "{} {}",
        "port".green().bold(),
        remote
            .port
            .map(|value| format!("{}", value))
            .unwrap_or_else(|| "default".into())
    );
}

fn config_value(repository: Option<&str>, key: &str) -> Option<Text> {
    run(repository, &["config", "--local", "--get", key])
        .ok()
        .map(|value| value.trim().into())
}

fn print_signing_status(repository: Option<&str>) -> Result<()> {
    run(repository, &["rev-parse", "--show-toplevel"])?;
    for (label, key) in [
        ("format", "gpg.format"),
        ("key", "user.signingkey"),
        ("commits", "commit.gpgsign"),
        ("tags", "tag.gpgsign"),
        ("allowed signers", "gpg.ssh.allowedSignersFile"),
    ] {
        println!(
            "{} {}",
            label.green().bold(),
            config_value(repository, key).unwrap_or_else(|| "not configured".into())
        );
    }
    Ok(())
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
            let (sign, words) = if tail.first().is_some_and(|value| value == "--sign") {
                (true, &tail[1..])
            } else {
                (false, tail)
            };
            require_arguments("commit", words)?;
            let message = join_words(words);
            if sign {
                run_visible(repository, &["commit", "-S", "-m", &message])
            } else {
                run_visible(repository, &["commit", "-m", &message])
            }
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
        "ssh" if tail.len() == 3 && tail[0] == "add" => {
            checked_positionals(&tail[1..2])?;
            SshRemote::parse(&tail[2]).map_err(|error| anyhow!("invalid SSH remote: {}", error))?;
            run_visible(repository, &["remote", "add", "--", &tail[1], &tail[2]])
        }
        "ssh" if tail.len() == 3 && tail[0] == "set" => {
            checked_positionals(&tail[1..2])?;
            SshRemote::parse(&tail[2]).map_err(|error| anyhow!("invalid SSH remote: {}", error))?;
            run_visible(repository, &["remote", "set-url", "--", &tail[1], &tail[2]])
        }
        "ssh" if matches!(tail.len(), 1 | 2) && tail[0] == "info" => {
            let name = tail.get(1).map(Text::as_str).unwrap_or("origin");
            checked_positionals(&tail[1..])?;
            let (url, parsed) = ssh_remote(repository, name)?;
            print_ssh_remote(name, &url, &parsed);
            Ok(())
        }
        "ssh" if matches!(tail.len(), 1 | 2) && tail[0] == "check" => {
            let name = tail.get(1).map(Text::as_str).unwrap_or("origin");
            checked_positionals(&tail[1..])?;
            let (url, parsed) = ssh_remote(repository, name)?;
            print_ssh_remote(name, &url, &parsed);
            println!("{}", "checking read-only SSH access...".dimmed());
            run_visible(
                repository,
                &["ls-remote", "--exit-code", "--", name, "HEAD"],
            )
        }
        "signing" if tail.len() == 2 && tail[0] == "configure" => {
            checked_positionals(&tail[1..])?;
            run_visible(repository, &["config", "--local", "gpg.format", "ssh"])?;
            run_visible(
                repository,
                &["config", "--local", "user.signingkey", &tail[1]],
            )?;
            print_signing_status(repository)
        }
        "signing" if tail.len() == 3 && tail[0] == "configure" => {
            checked_positionals(&tail[1..])?;
            run_visible(repository, &["config", "--local", "gpg.format", "ssh"])?;
            run_visible(
                repository,
                &["config", "--local", "user.signingkey", &tail[1]],
            )?;
            run_visible(
                repository,
                &["config", "--local", "gpg.ssh.allowedSignersFile", &tail[2]],
            )?;
            print_signing_status(repository)
        }
        "signing" if tail.len() == 1 && tail[0] == "auto" => {
            if config_value(repository, "gpg.format").as_deref() != Some("ssh")
                || config_value(repository, "user.signingkey").is_none()
            {
                return Err(anyhow!(
                    "run signing configure <key> [allowed-signers] first"
                ));
            }
            run_visible(repository, &["config", "--local", "commit.gpgsign", "true"])?;
            run_visible(repository, &["config", "--local", "tag.gpgsign", "true"])?;
            print_signing_status(repository)
        }
        "signing" if tail.len() == 1 && tail[0] == "status" => print_signing_status(repository),
        "signing" if tail.len() == 1 && tail[0] == "off" => {
            run_visible(
                repository,
                &["config", "--local", "commit.gpgsign", "false"],
            )?;
            run_visible(repository, &["config", "--local", "tag.gpgsign", "false"])?;
            print_signing_status(repository)
        }
        "signing" if tail.len() == 2 && tail[0] == "verify" => {
            checked_positionals(&tail[1..])?;
            run_visible(repository, &["verify-commit", "--", &tail[1]])
        }
        "signing" if tail.len() == 2 && tail[0] == "verify-tag" => {
            checked_positionals(&tail[1..])?;
            run_visible(repository, &["verify-tag", "--", &tail[1]])
        }
        "tag" if tail.is_empty() => run_visible(repository, &["tag"]),
        "tag" if tail.len() == 1 => {
            checked_positionals(tail)?;
            run_visible(repository, &collect("tag", &["--"], tail))
        }
        "tag-sign" if tail.len() >= 2 => {
            checked_positionals(&tail[..1])?;
            let message = join_words(&tail[1..]);
            run_visible(
                repository,
                &["tag", "--sign", "-m", &message, "--", &tail[0]],
            )
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
