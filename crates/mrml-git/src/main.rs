#![no_std]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(test, allow(dead_code))]

use mrml_error::{Context, Result, anyhow};
use mrml_git::{Change, Cli, MergeOutcome, NativeChangeKind, Repository, validate_positional};
use mrml_runtime::{Text, Vector, mrml_format as format, mrml_println as println};
use mrml_ssh::SshRemote;
use mrml_terminal_style::Colorize;

fn run(_repository: Option<&str>, args: &[&str]) -> Result<Text> {
    Err(native_missing(args))
}

fn run_visible(_repository: Option<&str>, args: &[&str]) -> Result<()> {
    Err(native_missing(args))
}

fn native_missing(args: &[&str]) -> mrml_error::Error {
    anyhow!(
        "native '{}' support is not implemented yet; mrml-git never delegates to a host Git executable",
        args.first().copied().unwrap_or("operation")
    )
}

fn status(_repository: Option<&str>) -> Result<Vector<Change>> {
    Err(native_missing(&["legacy status parser"]))
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

fn native_repository(repository: Option<&str>) -> Result<Repository> {
    Repository::discover(repository.unwrap_or(".")).map_err(|error| anyhow!("{}", error))
}

fn native_dashboard(repository: Option<&str>) -> Result<()> {
    let repository = native_repository(repository)?;
    let branch = repository
        .current_branch()
        .map_err(|error| anyhow!("{}", error))?
        .unwrap_or_else(|| "detached HEAD".into());
    let head = repository.head().map_err(|error| anyhow!("{}", error))?;
    let changes = repository.changes().map_err(|error| anyhow!("{}", error))?;
    println!("{}", "MRML GIT · NATIVE STATUS".bright_cyan().bold());
    println!("  {}   {}", "root".dimmed(), repository.worktree);
    println!("  {} {}", "branch".dimmed(), branch.magenta().bold());
    if let Some(id) = head {
        println!("  {}   {}", "head".dimmed(), id);
    } else {
        println!("  {}   no commits yet", "head".dimmed());
    }
    if changes.is_empty() {
        println!("  {}", "clean — index matches working tree".green());
    }
    for change in changes {
        let label = match change.kind {
            NativeChangeKind::Modified => "modified".yellow(),
            NativeChangeKind::Deleted => "deleted".red(),
            NativeChangeKind::Untracked => "untracked".bright_cyan(),
        };
        println!("  {:9} {}", label, change.path);
    }
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
        "[-C PATH] diff [--staged] [path]...",
        "[-C PATH] diff-ref <revision> [path]...",
        "[-C PATH] show [revision]",
        "[-C PATH] history <path>",
        "[-C PATH] blame <path>",
        "[-C PATH] conflicts",
        "[-C PATH] branch [new-name]",
        "[-C PATH] branch-delete <name>",
        "[-C PATH] switch <name>",
        "[-C PATH] upstream <remote/branch>",
        "[-C PATH] publish <remote> <branch>",
        "[-C PATH] merge <branch>",
        "[-C PATH] rebase <branch>",
        "[-C PATH] cherry-pick <revision>",
        "[-C PATH] operation-abort <merge|rebase|cherry-pick>",
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
    let url = native_repository(repository)?.remotes().map_err(|error| anyhow!("{}", error))?
        .into_iter().find(|(remote, _)| remote == name).map(|(_, url)| url).ok_or_else(|| anyhow!("remote '{}' does not exist", name))?;
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
    let (section, name) = key.rsplit_once('.')?;
    native_repository(repository).ok()?.config_value(section, name).ok().flatten()
}

fn print_signing_status(repository: Option<&str>) -> Result<()> {
    native_repository(repository)?;
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

fn conflicts(repository: Option<&str>) -> Result<()> {
    let changes = status(repository)?;
    let conflicted = changes
        .iter()
        .filter(|change| change.conflicted())
        .collect::<Vector<_>>();
    if conflicted.is_empty() {
        println!("{}", "no unresolved conflicts".green());
    } else {
        println!("{} unresolved conflict(s)", conflicted.len());
        for change in conflicted {
            println!(
                "  {}  {}{}  {}",
                "conflict".red().bold(),
                change.index,
                change.worktree,
                change.path
            );
        }
    }
    Ok(())
}

fn dispatch(cli: &Cli) -> Result<()> {
    let repository = cli.repository.as_deref();
    let tail: &[Text] = &cli.arguments;
    match cli.command.as_str() {
        "status" | "pulse" if tail.is_empty() => native_dashboard(repository),
        "help" | "--help" | "-h" if tail.is_empty() => {
            help();
            Ok(())
        }
        "--version" | "-V" if tail.is_empty() => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "doctor" if tail.is_empty() => {
            println!(
                "{} built-in object IDs, index, repository and status",
                "native".green().bold()
            );
            match native_repository(repository) {
                Ok(repo) => println!("{} {}", "repository".green().bold(), repo.worktree),
                Err(_) => println!("{} not inside a repository", "repository".yellow().bold()),
            }
            Ok(())
        }
        "init" if tail.len() <= 1 => {
            checked_positionals(tail)?;
            let path = tail.first().map(Text::as_str).or(repository).unwrap_or(".");
            let initialized = Repository::init(path).map_err(|error| anyhow!("{}", error))?;
            println!(
                "Initialized empty MRML Git repository in {}",
                initialized.git_dir
            );
            Ok(())
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
            let limit = count.parse::<usize>().map_err(|_| anyhow!("invalid log count"))?;
            let repository = native_repository(repository)?;
            let head = repository.head().map_err(|error| anyhow!("{}", error))?.ok_or_else(|| anyhow!("no commits yet"))?;
            for (id, commit) in repository.history(head, limit).map_err(|error| anyhow!("{}", error))? {
                println!("{} {}", (&id.to_hex()[..12]).bright_yellow(), commit.message.lines().next().unwrap_or(""));
                println!("  {}", commit.author.dimmed());
            }
            Ok(())
        }
        "diff" => {
            let staged = tail.first().is_some_and(|argument| argument == "--staged");
            let paths = if staged { &tail[1..] } else { tail };
            checked_positionals(paths)?;
            let diffs = native_repository(repository)?.diff(staged, paths).map_err(|error| anyhow!("{}", error))?;
            for diff in diffs { println!("{}", diff.unified()); }
            Ok(())
        }
        "diff-ref" if !tail.is_empty() => {
            checked_positionals(tail)?;
            let diffs = native_repository(repository)?.diff_revision(&tail[0], &tail[1..]).map_err(|error| anyhow!("{}", error))?;
            for diff in diffs { println!("{}", diff.unified()); }
            Ok(())
        }
        "show" if tail.len() <= 1 => {
            checked_positionals(tail)?;
            let repository = native_repository(repository)?;
            let id = repository.resolve_revision(tail.first().map(Text::as_str).unwrap_or("HEAD")).map_err(|error| anyhow!("{}", error))?;
            let commit = repository.read_commit(id).map_err(|error| anyhow!("{}", error))?;
            println!("{} {}", "commit".yellow(), id);
            println!("{} {}", "Author:".dimmed(), commit.author);
            println!("{} {}", "Tree:".dimmed(), commit.tree);
            for parent in commit.parents { println!("{} {}", "Parent:".dimmed(), parent); }
            println!("\n{}", commit.message);
            Ok(())
        }
        "history" if tail.len() == 1 => {
            checked_positionals(tail)?;
            for (id, commit) in native_repository(repository)?.file_history(&tail[0], 256).map_err(|error| anyhow!("{}", error))? {
                println!("{} {}", &id.to_hex()[..12], commit.message.lines().next().unwrap_or(""));
            }
            Ok(())
        }
        "blame" if tail.len() == 1 => {
            checked_positionals(tail)?;
            for line in native_repository(repository)?.blame(&tail[0]).map_err(|error| anyhow!("{}", error))? {
                println!("{} {:4} ({}) {}", &line.commit.to_hex()[..12], line.line_number, line.author, line.text.trim_end_matches('\n'));
            }
            Ok(())
        }
        "conflicts" if tail.is_empty() => {
            let paths = native_repository(repository)?.conflicted_paths().map_err(|error| anyhow!("{}", error))?;
            if paths.is_empty() { println!("{}", "no unresolved conflicts".green()); }
            else { for path in paths { println!("{} {}", "conflict".red().bold(), path); } }
            Ok(())
        }
        "branch" if tail.is_empty() => {
            let repository = native_repository(repository)?;
            let current = repository
                .current_branch()
                .map_err(|error| anyhow!("{}", error))?;
            for (name, id) in repository
                .branches()
                .map_err(|error| anyhow!("{}", error))?
            {
                if current.as_deref() == Some(&name) {
                    println!("{} {} {}", "*".green(), name, &id.to_hex()[..12]);
                } else {
                    println!("  {} {}", name, &id.to_hex()[..12]);
                }
            }
            Ok(())
        }
        "branch" if tail.len() == 1 => {
            checked_positionals(tail)?;
            let id = native_repository(repository)?
                .create_branch(&tail[0], true)
                .map_err(|error| anyhow!("{}", error))?;
            println!(
                "Switched to a new branch '{}' at {}",
                tail[0],
                &id.to_hex()[..12]
            );
            Ok(())
        }
        "branch-delete" if tail.len() == 1 => {
            checked_positionals(tail)?;
            native_repository(repository)?
                .delete_branch(&tail[0])
                .map_err(|error| anyhow!("{}", error))?;
            println!("Deleted branch {}", tail[0]);
            Ok(())
        }
        "switch" if tail.len() == 1 => {
            checked_positionals(tail)?;
            let id = native_repository(repository)?.switch_branch(&tail[0]).map_err(|error| anyhow!("{}", error))?;
            println!("Switched to branch '{}' at {}", tail[0], &id.to_hex()[..12]);
            Ok(())
        }
        "upstream" if tail.len() == 1 => {
            checked_positionals(tail)?;
            let setting = format!("--set-upstream-to={}", tail[0]);
            run_visible(repository, &["branch", &setting])
        }
        "publish" if tail.len() == 2 => {
            checked_positionals(tail)?;
            run_visible(repository, &["push", "--set-upstream", &tail[0], &tail[1]])
        }
        "merge" if tail.len() == 1 => {
            checked_positionals(tail)?;
            match native_repository(repository)?.merge(&tail[0]).map_err(|error| anyhow!("{}", error))? {
                MergeOutcome::UpToDate => println!("Already up to date."),
                MergeOutcome::FastForward(id) => println!("Fast-forward to {}", &id.to_hex()[..12]),
            }
            Ok(())
        }
        "rebase" if tail.len() == 1 => {
            checked_positionals(tail)?;
            run_visible(repository, &["rebase", "--", &tail[0]])
        }
        "cherry-pick" if tail.len() == 1 => {
            checked_positionals(tail)?;
            run_visible(repository, &["cherry-pick", "--", &tail[0]])
        }
        "operation-abort" if tail.len() == 1 && tail[0] == "merge" => {
            run_visible(repository, &["merge", "--abort"])
        }
        "operation-abort" if tail.len() == 1 && tail[0] == "rebase" => {
            run_visible(repository, &["rebase", "--abort"])
        }
        "operation-abort" if tail.len() == 1 && tail[0] == "cherry-pick" => {
            run_visible(repository, &["cherry-pick", "--abort"])
        }
        "stage" => {
            checked_positionals(require_arguments("stage", tail)?)?;
            native_repository(repository)?
                .stage(tail)
                .map_err(|error| anyhow!("{}", error))?;
            native_dashboard(repository)
        }
        "unstage" => {
            checked_positionals(require_arguments("unstage", tail)?)?;
            native_repository(repository)?.unstage(tail).map_err(|error| anyhow!("{}", error))?;
            native_dashboard(repository)
        }
        "restore" => {
            checked_positionals(require_arguments("restore", tail)?)?;
            native_repository(repository)?.restore(tail).map_err(|error| anyhow!("{}", error))?;
            native_dashboard(repository)
        }
        "commit" => {
            let (sign, words) = if tail.first().is_some_and(|value| value == "--sign") {
                (true, &tail[1..])
            } else {
                (false, tail)
            };
            require_arguments("commit", words)?;
            let message = join_words(words);
            if sign {
                return Err(anyhow!("native commit signing is not implemented yet"));
            }
            let name = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_NAME")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_NAME"))
                .unwrap_or_else(|| "MRML User".into());
            let email = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_EMAIL")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_EMAIL"))
                .unwrap_or_else(|| "mrml@localhost".into());
            let timestamp = mrml_runtime::unix_time_seconds()
                .ok_or_else(|| anyhow!("system time is unavailable"))?;
            let id = native_repository(repository)?
                .commit(&message, &name, &email, timestamp)
                .map_err(|error| anyhow!("{}", error))?;
            println!("[{}] {}", (&id.to_hex()[..12]).bright_green(), message);
            Ok(())
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
        "remote" if tail.is_empty() => {
            for (name, url) in native_repository(repository)?.remotes().map_err(|error| anyhow!("{}", error))? { println!("{}\t{}", name, url); }
            Ok(())
        }
        "ssh" if tail.len() == 3 && tail[0] == "add" => {
            checked_positionals(&tail[1..2])?;
            SshRemote::parse(&tail[2]).map_err(|error| anyhow!("invalid SSH remote: {}", error))?;
            native_repository(repository)?.set_remote(&tail[1], &tail[2], false).map_err(|error| anyhow!("{}", error))?;
            Ok(())
        }
        "ssh" if tail.len() == 3 && tail[0] == "set" => {
            checked_positionals(&tail[1..2])?;
            SshRemote::parse(&tail[2]).map_err(|error| anyhow!("invalid SSH remote: {}", error))?;
            native_repository(repository)?.set_remote(&tail[1], &tail[2], true).map_err(|error| anyhow!("{}", error))?;
            Ok(())
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
        "tag" if tail.is_empty() => {
            for (name, _) in native_repository(repository)?
                .tags()
                .map_err(|error| anyhow!("{}", error))?
            {
                println!("{}", name);
            }
            Ok(())
        }
        "tag" if tail.len() == 1 => {
            checked_positionals(tail)?;
            let id = native_repository(repository)?
                .create_tag(&tail[0])
                .map_err(|error| anyhow!("{}", error))?;
            println!("Tagged {} at {}", tail[0], &id.to_hex()[..12]);
            Ok(())
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
            for (index, (id, commit)) in native_repository(repository)?.stash_list(256).map_err(|error| anyhow!("{}", error))?.into_iter().enumerate() {
                println!("stash@{{{}}}: {} {}", index, &id.to_hex()[..12], commit.message.trim());
            }
            Ok(())
        }
        "stash" if tail.len() == 1 && tail[0] == "pop" => {
            let id = native_repository(repository)?.stash_pop().map_err(|error| anyhow!("{}", error))?;
            println!("Applied stash {}", &id.to_hex()[..12]);
            Ok(())
        }
        "stash" if !tail.is_empty() && tail[0] == "push" => {
            let message = join_words(&tail[1..]);
            let name = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_NAME").or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_NAME")).unwrap_or_else(|| "MRML User".into());
            let email = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_EMAIL").or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_EMAIL")).unwrap_or_else(|| "mrml@localhost".into());
            let timestamp = mrml_runtime::unix_time_seconds().ok_or_else(|| anyhow!("system time is unavailable"))?;
            let id = native_repository(repository)?.stash_push(&message, &name, &email, timestamp).map_err(|error| anyhow!("{}", error))?;
            println!("Saved stash {}", &id.to_hex()[..12]);
            Ok(())
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
