#![no_std]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(test, allow(dead_code))]

use mrml_error::{Context, Result, anyhow};
use mrml_git::{
    Cli, MergeOutcome, NativeChangeKind, RebaseOutcome, Repository, check_ssh, fetch_ssh, push_ssh,
    validate_positional,
};
use mrml_runtime::{
    Text, Vector, mrml_format as format, mrml_println as println, read_file_text_bounded,
};
use mrml_ssh::{
    RsaPrivateKey, SshRemote, encode_rsa_public_key, parse_rsa_private_pem, parse_rsa_public_line,
};
use mrml_terminal_style::Colorize;

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
    if let Some((upstream, ahead, behind)) = repository
        .upstream_status()
        .map_err(|error| anyhow!("{}", error))?
    {
        println!(
            "  {} {}  {} ahead  {} behind",
            "track".dimmed(),
            upstream,
            ahead,
            behind
        );
    } else {
        println!("  {} {}", "track".dimmed(), "no upstream".yellow());
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
        "[-C PATH] ssh <add|set|auth|info|check> ...",
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
    let url = native_repository(repository)?
        .remotes()
        .map_err(|error| anyhow!("{}", error))?
        .into_iter()
        .find(|(remote, _)| remote == name)
        .map(|(_, url)| url)
        .ok_or_else(|| anyhow!("remote '{}' does not exist", name))?;
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

fn ssh_credentials_paths(
    private_path: &str,
    host_path: &str,
) -> Result<(RsaPrivateKey, Vector<u8>)> {
    let private = read_file_text_bounded(private_path, 64 * 1024)
        .map_err(|_| anyhow!("cannot read SSH private key"))?;
    let host = read_file_text_bounded(host_path, 64 * 1024)
        .map_err(|_| anyhow!("cannot read pinned SSH host key"))?;
    let private = parse_rsa_private_pem(&private)
        .map_err(|error| anyhow!("invalid SSH private key: {}", error))?;
    let host = parse_rsa_public_line(&host)
        .and_then(|key| encode_rsa_public_key(&key))
        .map_err(|error| anyhow!("invalid SSH host key: {}", error))?;
    Ok((private, host))
}
fn ssh_credentials(repository: &Repository) -> Result<(RsaPrivateKey, Vector<u8>)> {
    let private = repository
        .config_value("ssh", "privateKey")
        .map_err(|error| anyhow!("{}", error))?
        .ok_or_else(|| anyhow!("run ssh auth <private-key.pem> <host-public-key> first"))?;
    let host = repository
        .config_value("ssh", "hostKey")
        .map_err(|error| anyhow!("{}", error))?
        .ok_or_else(|| anyhow!("run ssh auth <private-key.pem> <host-public-key> first"))?;
    ssh_credentials_paths(&private, &host)
}
fn repository_signing_key(repository: &Repository) -> Result<RsaPrivateKey> {
    let path = repository
        .config_value("user", "signingkey")
        .map_err(|error| anyhow!("{}", error))?
        .or_else(|| repository.config_value("ssh", "privateKey").ok().flatten())
        .ok_or_else(|| anyhow!("run signing configure <private-key.pem> [allowed-signer] first"))?;
    let text =
        read_file_text_bounded(&path, 64 * 1024).map_err(|_| anyhow!("cannot read signing key"))?;
    parse_rsa_private_pem(&text).map_err(|error| anyhow!("invalid signing key: {}", error))
}
fn repository_verification_key(repository: &Repository) -> Result<mrml_ssh::RsaPublicKey> {
    let path = repository
        .config_value("gpg.ssh", "allowedSignersFile")
        .map_err(|error| anyhow!("{}", error))?
        .ok_or_else(|| anyhow!("signing verification requires an allowed signer file"))?;
    let text = read_file_text_bounded(&path, 64 * 1024)
        .map_err(|_| anyhow!("cannot read allowed signer"))?;
    parse_rsa_public_line(&text).map_err(|error| anyhow!("invalid allowed signer: {}", error))
}

fn native_fetch(repository: Option<&str>, name: &str) -> Result<()> {
    let repo = native_repository(repository)?;
    let (_, remote) = ssh_remote(repository, name)?;
    let (key, host) = ssh_credentials(&repo)?;
    let result =
        fetch_ssh(&repo, name, &remote, &key, &host).map_err(|error| anyhow!("{}", error))?;
    println!(
        "Fetched {} object(s) and {} branch ref(s) from {}",
        result.objects.len(),
        result.branches.len(),
        name
    );
    Ok(())
}
fn native_push(repository: Option<&str>, name: &str, branch: Option<&str>) -> Result<()> {
    let repo = native_repository(repository)?;
    let branch = branch
        .map(Into::into)
        .or_else(|| repo.current_branch().ok().flatten())
        .ok_or_else(|| anyhow!("push requires a branch for detached HEAD"))?;
    let (_, remote) = ssh_remote(repository, name)?;
    let (key, host) = ssh_credentials(&repo)?;
    let result = push_ssh(&repo, name, &branch, &remote, &key, &host)
        .map_err(|error| anyhow!("{}", error))?;
    if result.old == result.new {
        println!("Everything up to date.");
    } else {
        println!(
            "Pushed {} to {}/{}",
            &result.new.to_hex()[..12],
            name,
            branch
        );
    }
    Ok(())
}

fn config_value(repository: Option<&str>, key: &str) -> Option<Text> {
    let (section, name) = key.rsplit_once('.')?;
    native_repository(repository)
        .ok()?
        .config_value(section, name)
        .ok()
        .flatten()
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
            let remote = SshRemote::parse(&tail[0])
                .map_err(|error| anyhow!("invalid SSH remote: {}", error))?;
            let target = tail.get(1).cloned().unwrap_or_else(|| {
                remote
                    .path
                    .rsplit('/')
                    .next()
                    .unwrap_or("repository")
                    .trim_end_matches(".git")
                    .into()
            });
            let private_path = mrml_runtime::environment_variable("MRML_GIT_SSH_KEY")
                .ok_or_else(|| anyhow!("clone requires MRML_GIT_SSH_KEY"))?;
            let host_path = mrml_runtime::environment_variable("MRML_GIT_SSH_HOST_KEY")
                .ok_or_else(|| anyhow!("clone requires MRML_GIT_SSH_HOST_KEY"))?;
            let (key, host) = ssh_credentials_paths(&private_path, &host_path)?;
            let repo = Repository::init(&target).map_err(|error| anyhow!("{}", error))?;
            repo.set_remote("origin", &tail[0], false)
                .map_err(|error| anyhow!("{}", error))?;
            repo.set_config_value("ssh", "privateKey", &private_path)
                .map_err(|error| anyhow!("{}", error))?;
            repo.set_config_value("ssh", "hostKey", &host_path)
                .map_err(|error| anyhow!("{}", error))?;
            let result = fetch_ssh(&repo, "origin", &remote, &key, &host)
                .map_err(|error| anyhow!("{}", error))?;
            let branch = result
                .default_branch
                .as_ref()
                .and_then(|name| result.branches.iter().find(|(branch, _)| branch == name))
                .or_else(|| result.branches.iter().find(|(branch, _)| branch == "main"))
                .or_else(|| {
                    result
                        .branches
                        .iter()
                        .find(|(branch, _)| branch == "master")
                })
                .or_else(|| result.branches.first())
                .ok_or_else(|| anyhow!("remote has no branch to check out"))?;
            repo.checkout_branch_at(&branch.0, branch.1)
                .map_err(|error| anyhow!("{}", error))?;
            println!("Cloned {} into {} on branch {}", tail[0], target, branch.0);
            Ok(())
        }
        "log" => {
            let count = tail.first().map(Text::as_str).unwrap_or("12");
            if tail.len() > 1 || count.parse::<usize>().is_err() {
                return Err(anyhow!("log accepts one numeric count"));
            }
            let limit = count
                .parse::<usize>()
                .map_err(|_| anyhow!("invalid log count"))?;
            let repository = native_repository(repository)?;
            let head = repository
                .head()
                .map_err(|error| anyhow!("{}", error))?
                .ok_or_else(|| anyhow!("no commits yet"))?;
            for (id, commit) in repository
                .history(head, limit)
                .map_err(|error| anyhow!("{}", error))?
            {
                println!(
                    "{} {}",
                    (&id.to_hex()[..12]).bright_yellow(),
                    commit.message.lines().next().unwrap_or("")
                );
                println!("  {}", commit.author.dimmed());
            }
            Ok(())
        }
        "diff" => {
            let staged = tail.first().is_some_and(|argument| argument == "--staged");
            let paths = if staged { &tail[1..] } else { tail };
            checked_positionals(paths)?;
            let diffs = native_repository(repository)?
                .diff(staged, paths)
                .map_err(|error| anyhow!("{}", error))?;
            for diff in diffs {
                println!("{}", diff.unified());
            }
            Ok(())
        }
        "diff-ref" if !tail.is_empty() => {
            checked_positionals(tail)?;
            let diffs = native_repository(repository)?
                .diff_revision(&tail[0], &tail[1..])
                .map_err(|error| anyhow!("{}", error))?;
            for diff in diffs {
                println!("{}", diff.unified());
            }
            Ok(())
        }
        "show" if tail.len() <= 1 => {
            checked_positionals(tail)?;
            let repository = native_repository(repository)?;
            let id = repository
                .resolve_revision(tail.first().map(Text::as_str).unwrap_or("HEAD"))
                .map_err(|error| anyhow!("{}", error))?;
            let commit = repository
                .read_commit(id)
                .map_err(|error| anyhow!("{}", error))?;
            println!("{} {}", "commit".yellow(), id);
            println!("{} {}", "Author:".dimmed(), commit.author);
            println!("{} {}", "Tree:".dimmed(), commit.tree);
            for parent in commit.parents {
                println!("{} {}", "Parent:".dimmed(), parent);
            }
            println!("\n{}", commit.message);
            Ok(())
        }
        "history" if tail.len() == 1 => {
            checked_positionals(tail)?;
            for (id, commit) in native_repository(repository)?
                .file_history(&tail[0], 256)
                .map_err(|error| anyhow!("{}", error))?
            {
                println!(
                    "{} {}",
                    &id.to_hex()[..12],
                    commit.message.lines().next().unwrap_or("")
                );
            }
            Ok(())
        }
        "blame" if tail.len() == 1 => {
            checked_positionals(tail)?;
            for line in native_repository(repository)?
                .blame(&tail[0])
                .map_err(|error| anyhow!("{}", error))?
            {
                println!(
                    "{} {:4} ({}) {}",
                    &line.commit.to_hex()[..12],
                    line.line_number,
                    line.author,
                    line.text.trim_end_matches('\n')
                );
            }
            Ok(())
        }
        "conflicts" if tail.is_empty() => {
            let paths = native_repository(repository)?
                .conflicted_paths()
                .map_err(|error| anyhow!("{}", error))?;
            if paths.is_empty() {
                println!("{}", "no unresolved conflicts".green());
            } else {
                for path in paths {
                    println!("{} {}", "conflict".red().bold(), path);
                }
            }
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
            let id = native_repository(repository)?
                .switch_branch(&tail[0])
                .map_err(|error| anyhow!("{}", error))?;
            println!("Switched to branch '{}' at {}", tail[0], &id.to_hex()[..12]);
            Ok(())
        }
        "upstream" if tail.len() == 1 => {
            checked_positionals(tail)?;
            native_repository(repository)?
                .set_upstream(&tail[0])
                .map_err(|error| anyhow!("{}", error))?;
            println!("Branch now tracks {}", tail[0]);
            Ok(())
        }
        "publish" if tail.len() == 2 => {
            checked_positionals(tail)?;
            native_push(repository, &tail[0], Some(&tail[1]))?;
            native_repository(repository)?
                .set_upstream(&format!("{}/{}", tail[0], tail[1]))
                .map_err(|error| anyhow!("{}", error))?;
            Ok(())
        }
        "merge" if tail.len() == 1 => {
            checked_positionals(tail)?;
            let name = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_NAME")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_NAME"))
                .unwrap_or_else(|| "MRML User".into());
            let email = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_EMAIL")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_EMAIL"))
                .unwrap_or_else(|| "mrml@localhost".into());
            let timestamp = mrml_runtime::unix_time_seconds()
                .ok_or_else(|| anyhow!("system time is unavailable"))?;
            match native_repository(repository)?
                .merge(&tail[0], &name, &email, timestamp)
                .map_err(|error| anyhow!("{}", error))?
            {
                MergeOutcome::UpToDate => println!("Already up to date."),
                MergeOutcome::FastForward(id) => println!("Fast-forward to {}", &id.to_hex()[..12]),
                MergeOutcome::Merged(id) => println!("Merge made commit {}", &id.to_hex()[..12]),
                MergeOutcome::Conflicts(count) => {
                    println!("Merge has {} conflict(s); resolve and commit", count)
                }
            }
            Ok(())
        }
        "rebase" if tail.len() == 1 => {
            checked_positionals(tail)?;
            let name = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_NAME")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_NAME"))
                .unwrap_or_else(|| "MRML User".into());
            let email = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_EMAIL")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_EMAIL"))
                .unwrap_or_else(|| "mrml@localhost".into());
            let timestamp = mrml_runtime::unix_time_seconds()
                .ok_or_else(|| anyhow!("system time is unavailable"))?;
            match native_repository(repository)?
                .rebase(&tail[0], &name, &email, timestamp)
                .map_err(|error| anyhow!("{}", error))?
            {
                RebaseOutcome::UpToDate => println!("Current branch is up to date."),
                RebaseOutcome::Rebased { count, head } => println!(
                    "Rebased {} commit(s); HEAD is {}",
                    count,
                    &head.to_hex()[..12]
                ),
                RebaseOutcome::Conflicts(count) => {
                    println!("Rebase stopped with {} conflict(s)", count)
                }
            }
            Ok(())
        }
        "cherry-pick" if tail.len() == 1 => {
            checked_positionals(tail)?;
            let name = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_NAME")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_NAME"))
                .unwrap_or_else(|| "MRML User".into());
            let email = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_EMAIL")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_EMAIL"))
                .unwrap_or_else(|| "mrml@localhost".into());
            let timestamp = mrml_runtime::unix_time_seconds()
                .ok_or_else(|| anyhow!("system time is unavailable"))?;
            match native_repository(repository)?
                .cherry_pick(&tail[0], &name, &email, timestamp)
                .map_err(|error| anyhow!("{}", error))?
            {
                MergeOutcome::Merged(id) => {
                    println!("Cherry-picked {} as {}", tail[0], &id.to_hex()[..12])
                }
                MergeOutcome::Conflicts(count) => {
                    println!("Cherry-pick has {} conflict(s); resolve and commit", count)
                }
                _ => return Err(anyhow!("unexpected cherry-pick outcome")),
            }
            Ok(())
        }
        "operation-abort" if tail.len() == 1 && tail[0] == "merge" => {
            let id = native_repository(repository)?
                .abort_merge()
                .map_err(|error| anyhow!("{}", error))?;
            println!("Aborted merge; restored {}", &id.to_hex()[..12]);
            Ok(())
        }
        "operation-abort" if tail.len() == 1 && tail[0] == "rebase" => {
            let id = native_repository(repository)?
                .abort_rebase()
                .map_err(|error| anyhow!("{}", error))?;
            println!("Aborted rebase; restored {}", &id.to_hex()[..12]);
            Ok(())
        }
        "operation-abort" if tail.len() == 1 && tail[0] == "cherry-pick" => {
            let id = native_repository(repository)?
                .abort_cherry_pick()
                .map_err(|error| anyhow!("{}", error))?;
            println!("Aborted cherry-pick; restored {}", &id.to_hex()[..12]);
            Ok(())
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
            native_repository(repository)?
                .unstage(tail)
                .map_err(|error| anyhow!("{}", error))?;
            native_dashboard(repository)
        }
        "restore" => {
            checked_positionals(require_arguments("restore", tail)?)?;
            native_repository(repository)?
                .restore(tail)
                .map_err(|error| anyhow!("{}", error))?;
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
            let name = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_NAME")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_NAME"))
                .unwrap_or_else(|| "MRML User".into());
            let email = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_EMAIL")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_EMAIL"))
                .unwrap_or_else(|| "mrml@localhost".into());
            let timestamp = mrml_runtime::unix_time_seconds()
                .ok_or_else(|| anyhow!("system time is unavailable"))?;
            let repo = native_repository(repository)?;
            let sign = sign
                || repo
                    .config_value("commit", "gpgsign")
                    .ok()
                    .flatten()
                    .is_some_and(|value| value.eq_ignore_ascii_case("true"));
            let id = if sign {
                let key = repository_signing_key(&repo)?;
                repo.commit_signed(&message, &name, &email, timestamp, &key)
            } else {
                repo.commit(&message, &name, &email, timestamp)
            }
            .map_err(|error| anyhow!("{}", error))?;
            println!("[{}] {}", (&id.to_hex()[..12]).bright_green(), message);
            Ok(())
        }
        "fetch" if tail.len() <= 1 => {
            checked_positionals(tail)?;
            native_fetch(
                repository,
                tail.first().map(Text::as_str).unwrap_or("origin"),
            )
        }
        "pull" if tail.len() <= 2 => {
            checked_positionals(tail)?;
            let remote = tail.first().map(Text::as_str).unwrap_or("origin");
            native_fetch(repository, remote)?;
            let repo = native_repository(repository)?;
            let branch = tail
                .get(1)
                .cloned()
                .or_else(|| repo.current_branch().ok().flatten())
                .ok_or_else(|| anyhow!("pull requires a branch for detached HEAD"))?;
            let revision = format!("refs/remotes/{remote}/{branch}");
            match repo
                .fast_forward(&revision)
                .map_err(|error| anyhow!("fast-forward pull failed: {}", error))?
            {
                MergeOutcome::UpToDate => println!("Already up to date."),
                MergeOutcome::FastForward(id) => println!("Fast-forward to {}", &id.to_hex()[..12]),
                _ => return Err(anyhow!("unexpected pull outcome")),
            }
            Ok(())
        }
        "push" if tail.len() <= 2 => {
            checked_positionals(tail)?;
            native_push(
                repository,
                tail.first().map(Text::as_str).unwrap_or("origin"),
                tail.get(1).map(Text::as_str),
            )
        }
        "remote" if tail.is_empty() => {
            for (name, url) in native_repository(repository)?
                .remotes()
                .map_err(|error| anyhow!("{}", error))?
            {
                println!("{}\t{}", name, url);
            }
            Ok(())
        }
        "ssh" if tail.len() == 3 && tail[0] == "add" => {
            checked_positionals(&tail[1..2])?;
            SshRemote::parse(&tail[2]).map_err(|error| anyhow!("invalid SSH remote: {}", error))?;
            native_repository(repository)?
                .set_remote(&tail[1], &tail[2], false)
                .map_err(|error| anyhow!("{}", error))?;
            Ok(())
        }
        "ssh" if tail.len() == 3 && tail[0] == "set" => {
            checked_positionals(&tail[1..2])?;
            SshRemote::parse(&tail[2]).map_err(|error| anyhow!("invalid SSH remote: {}", error))?;
            native_repository(repository)?
                .set_remote(&tail[1], &tail[2], true)
                .map_err(|error| anyhow!("{}", error))?;
            Ok(())
        }
        "ssh" if tail.len() == 3 && tail[0] == "auth" => {
            checked_positionals(&tail[1..])?;
            let repo = native_repository(repository)?;
            let private = read_file_text_bounded(&tail[1], 64 * 1024)
                .map_err(|_| anyhow!("cannot read SSH private key"))?;
            parse_rsa_private_pem(&private)
                .map_err(|error| anyhow!("invalid SSH private key: {}", error))?;
            let host = read_file_text_bounded(&tail[2], 64 * 1024)
                .map_err(|_| anyhow!("cannot read SSH host key"))?;
            parse_rsa_public_line(&host)
                .map_err(|error| anyhow!("invalid SSH host key: {}", error))?;
            repo.set_config_value("ssh", "privateKey", &tail[1])
                .map_err(|error| anyhow!("{}", error))?;
            repo.set_config_value("ssh", "hostKey", &tail[2])
                .map_err(|error| anyhow!("{}", error))?;
            println!(
                "{} repository-local SSH credentials configured",
                "native".green().bold()
            );
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
            let repo = native_repository(repository)?;
            let (key, host) = ssh_credentials(&repo)?;
            let refs = check_ssh(&parsed, &key, &host).map_err(|error| anyhow!("{}", error))?;
            println!(
                "{} authenticated; {} reference(s) advertised",
                "SSH access valid".green().bold(),
                refs
            );
            Ok(())
        }
        "signing" if tail.len() == 2 && tail[0] == "configure" => {
            checked_positionals(&tail[1..])?;
            let repo = native_repository(repository)?;
            let text = read_file_text_bounded(&tail[1], 64 * 1024)
                .map_err(|_| anyhow!("cannot read signing key"))?;
            parse_rsa_private_pem(&text)
                .map_err(|error| anyhow!("invalid signing key: {}", error))?;
            repo.set_config_value("gpg", "format", "ssh")
                .map_err(|error| anyhow!("{}", error))?;
            repo.set_config_value("user", "signingkey", &tail[1])
                .map_err(|error| anyhow!("{}", error))?;
            print_signing_status(repository)
        }
        "signing" if tail.len() == 3 && tail[0] == "configure" => {
            checked_positionals(&tail[1..])?;
            let repo = native_repository(repository)?;
            let private = read_file_text_bounded(&tail[1], 64 * 1024)
                .map_err(|_| anyhow!("cannot read signing key"))?;
            parse_rsa_private_pem(&private)
                .map_err(|error| anyhow!("invalid signing key: {}", error))?;
            let allowed = read_file_text_bounded(&tail[2], 64 * 1024)
                .map_err(|_| anyhow!("cannot read allowed signer"))?;
            parse_rsa_public_line(&allowed)
                .map_err(|error| anyhow!("invalid allowed signer: {}", error))?;
            repo.set_config_value("gpg", "format", "ssh")
                .map_err(|error| anyhow!("{}", error))?;
            repo.set_config_value("user", "signingkey", &tail[1])
                .map_err(|error| anyhow!("{}", error))?;
            repo.set_config_value("gpg.ssh", "allowedSignersFile", &tail[2])
                .map_err(|error| anyhow!("{}", error))?;
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
            let repo = native_repository(repository)?;
            repo.set_config_value("commit", "gpgsign", "true")
                .map_err(|error| anyhow!("{}", error))?;
            repo.set_config_value("tag", "gpgsign", "true")
                .map_err(|error| anyhow!("{}", error))?;
            print_signing_status(repository)
        }
        "signing" if tail.len() == 1 && tail[0] == "status" => print_signing_status(repository),
        "signing" if tail.len() == 1 && tail[0] == "off" => {
            let repo = native_repository(repository)?;
            repo.set_config_value("commit", "gpgsign", "false")
                .map_err(|error| anyhow!("{}", error))?;
            repo.set_config_value("tag", "gpgsign", "false")
                .map_err(|error| anyhow!("{}", error))?;
            print_signing_status(repository)
        }
        "signing" if tail.len() == 2 && tail[0] == "verify" => {
            checked_positionals(&tail[1..])?;
            let repo = native_repository(repository)?;
            let id = repo
                .resolve_revision(&tail[1])
                .map_err(|error| anyhow!("{}", error))?;
            let key = repository_verification_key(&repo)?;
            repo.verify_commit_signature(id, &key)
                .map_err(|error| anyhow!("{}", error))?;
            println!(
                "{} commit {}",
                "valid SSH signature".green().bold(),
                &id.to_hex()[..12]
            );
            Ok(())
        }
        "signing" if tail.len() == 2 && tail[0] == "verify-tag" => {
            checked_positionals(&tail[1..])?;
            let repo = native_repository(repository)?;
            let id = repo
                .resolve_revision(&tail[1])
                .map_err(|error| anyhow!("{}", error))?;
            let key = repository_verification_key(&repo)?;
            repo.verify_tag_signature(id, &key)
                .map_err(|error| anyhow!("{}", error))?;
            println!("{} tag {}", "valid SSH signature".green().bold(), tail[1]);
            Ok(())
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
            let repo = native_repository(repository)?;
            let key = repository_signing_key(&repo)?;
            let name = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_NAME")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_NAME"))
                .unwrap_or_else(|| "MRML User".into());
            let email = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_EMAIL")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_EMAIL"))
                .unwrap_or_else(|| "mrml@localhost".into());
            let timestamp = mrml_runtime::unix_time_seconds()
                .ok_or_else(|| anyhow!("system time is unavailable"))?;
            let id = repo
                .create_signed_tag(&tail[0], &message, &name, &email, timestamp, &key)
                .map_err(|error| anyhow!("{}", error))?;
            println!("Signed tag {} at {}", tail[0], &id.to_hex()[..12]);
            Ok(())
        }
        "stash" if tail.is_empty() || (tail.len() == 1 && tail[0] == "list") => {
            for (index, (id, commit)) in native_repository(repository)?
                .stash_list(256)
                .map_err(|error| anyhow!("{}", error))?
                .into_iter()
                .enumerate()
            {
                println!(
                    "stash@{{{}}}: {} {}",
                    index,
                    &id.to_hex()[..12],
                    commit.message.trim()
                );
            }
            Ok(())
        }
        "stash" if tail.len() == 1 && tail[0] == "pop" => {
            let id = native_repository(repository)?
                .stash_pop()
                .map_err(|error| anyhow!("{}", error))?;
            println!("Applied stash {}", &id.to_hex()[..12]);
            Ok(())
        }
        "stash" if !tail.is_empty() && tail[0] == "push" => {
            let message = join_words(&tail[1..]);
            let name = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_NAME")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_NAME"))
                .unwrap_or_else(|| "MRML User".into());
            let email = mrml_runtime::environment_variable("MRML_GIT_AUTHOR_EMAIL")
                .or_else(|| mrml_runtime::environment_variable("GIT_AUTHOR_EMAIL"))
                .unwrap_or_else(|| "mrml@localhost".into());
            let timestamp = mrml_runtime::unix_time_seconds()
                .ok_or_else(|| anyhow!("system time is unavailable"))?;
            let id = native_repository(repository)?
                .stash_push(&message, &name, &email, timestamp)
                .map_err(|error| anyhow!("{}", error))?;
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
