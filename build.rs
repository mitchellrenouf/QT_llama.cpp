use std::env;
use std::path::Path;
use std::process::Command;

fn command_works(command: &str, args: &[&str]) -> bool {
    Command::new(command)
        .args(args)
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn qt6_available(command: &str) -> bool {
    Command::new(command)
        .args(["-query", "QT_VERSION"])
        .output()
        .map(|out| {
            out.status.success()
                && String::from_utf8_lossy(&out.stdout)
                    .trim()
                    .starts_with("6.")
        })
        .unwrap_or(false)
}

#[cfg(windows)]
fn command_exists(command: &str) -> bool {
    Command::new("where.exe")
        .arg(command)
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

#[cfg(windows)]
fn visual_studio_cxx_installed() -> bool {
    let program_files_x86 = env::var_os("ProgramFiles(x86)");
    let Some(program_files_x86) = program_files_x86 else {
        return false;
    };

    let vswhere = Path::new(&program_files_x86)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    vswhere.is_file()
        && Command::new(vswhere)
            .args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ])
            .output()
            .map(|out| out.status.success() && !out.stdout.is_empty())
            .unwrap_or(false)
}

fn main() {
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=QTDIR");
    println!("cargo:rerun-if-env-changed=QT_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let qmake = ["qmake6", "qmake", "qmake-qt5"]
        .iter()
        .find(|command| qt6_available(command));
    if qmake.is_none() {
        panic!(
            "Qt 6 development tools were not found. Install Qt 6 Widgets and QtUiTools, then make qmake or qmake6 available on PATH. See README.md."
        );
    }

    #[cfg(windows)]
    {
        if env::var_os("VCINSTALLDIR").is_none()
            && !command_exists("cl.exe")
            && !visual_studio_cxx_installed()
        {
            panic!(
                "MSVC C++ Build Tools were not found. Install the Visual Studio Desktop development with C++ workload."
            );
        }
        println!("cargo:warning=Using native Qt 6 Widgets with MSVC.");
    }

    #[cfg(not(windows))]
    if !command_works("c++", &["--version"]) {
        panic!("A C++ compiler is required for the native Qt binding. Install your platform's C++ build tools.");
    }

    let cuda = env::var("CUDA_PATH")
        .ok()
        .map(|path| {
            Path::new(&path)
                .join(if cfg!(windows) {
                    "bin/nvcc.exe"
                } else {
                    "bin/nvcc"
                })
                .is_file()
        })
        .unwrap_or_else(|| command_works("nvcc", &["--version"]));
    if cuda {
        println!("cargo:warning=CUDA toolkit detected; CUDA acceleration is available.");
    } else {
        println!("cargo:warning=CUDA toolkit not detected; building without CUDA acceleration.");
    }
}
