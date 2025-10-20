#!/usr/bin/env python


# Imports
# ------------------------------------------------------------------------------

# Python standard library.
import logging
from pathlib import Path
import shlex
import subprocess
import sys


# Structure of Triton source files.
# ------------------------------------------------------------------------------


def check_dir(p: Path) -> Path:
    if not p.exists():
        logging.critical("Required directory [%s] doesn't exist.", p)
        sys.exit(1)
    if not p.is_dir():
        logging.critical("Required directory [%s] isn't a directory.", p)
        sys.exit(1)
    return p


ROOT_DIR: Path = check_dir(Path(__file__).parent.parent.parent)
TRITON_OP_DIR: Path = check_dir(ROOT_DIR / "aiter" / "ops" / "triton")
TRITON_CONFIG_DIR: Path = check_dir(TRITON_OP_DIR / "configs")
TEST_DIR: Path = check_dir(ROOT_DIR / "op_tests")
TRITON_TEST_DIR: Path = check_dir(TEST_DIR / "triton_tests")
TRITON_BENCH_DIR: Path = check_dir(TEST_DIR / "op_benchmarks" / "triton")


def list_files(dir: Path, suffix: str = "") -> set[Path]:
    return {p.relative_to(ROOT_DIR) for p in dir.glob(f"**/*{suffix}") if p.is_file()}


def list_triton_op_files() -> set[Path]:
    files = list_files(TRITON_OP_DIR, suffix=".py")
    logging.debug("Found %d Triton operator source files.", len(files))
    return files


def list_triton_config_files() -> set[Path]:
    files = list_files(TRITON_CONFIG_DIR, suffix=".json")
    logging.debug("Found %d Triton operator config files.", len(files))
    return files


def list_triton_test_files() -> set[Path]:
    files = list_files(TRITON_TEST_DIR, suffix=".py")
    logging.debug("Found %d Triton test source files.", len(files))
    return files


def list_triton_bench_files() -> set[Path]:
    # TODO: How to deal with these files?
    #       op_tests/op_benchmarks/triton/utils/model_configs.json
    #       op_tests/op_benchmarks/triton/bench_schema.yaml
    files = list_files(TRITON_BENCH_DIR, suffix=".py")
    logging.debug("Found %d Triton benchmark source files.", len(files))
    return files


def list_triton_source_files() -> tuple[set[Path], ...]:
    op_files = list_triton_op_files()
    config_files = list_triton_config_files()
    test_files = list_triton_test_files()
    bench_files = list_triton_bench_files()
    all_files = op_files | config_files | test_files | bench_files
    return all_files, config_files, test_files, bench_files


# Git commands.
# ------------------------------------------------------------------------------


def git(args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git"] + shlex.split(args),
            capture_output=True,
            text=True,
            check=check,
        )
    except FileNotFoundError:
        logging.critical("Git not found.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logging.critical("Malformed Git command: [git %s].", args)
        sys.exit(1)


def git_current_branch() -> str:
    return git("rev-parse --abbrev-ref HEAD").stdout.rstrip()


def git_check_branch(branch: str) -> None:
    if git(f"rev-parse --verify --quiet {branch}", check=False).returncode != 0:
        logging.critical("Branch [%s] doesn't exist.", branch)
        sys.exit(1)


def git_filename_diff(source_branch: str, target_branch: str) -> set[Path]:
    files = {
        Path(p)
        for p in git(f"diff --name-only {target_branch} {source_branch}")
        .stdout.rstrip()
        .splitlines()
    }
    logging.debug(
        "There %s %d file%s in the diff from [%s] to [%s].",
        "is" if len(files) == 1 else "are",
        len(files),
        "" if len(files) == 1 else "s",
        source_branch,
        target_branch,
    )
    files = {p for p in files if p.exists() and p.is_file()}
    logging.debug(
        "There %s %d file%s in the diff from [%s] to [%s] after filtering existing files.",
        "is" if len(files) == 1 else "are",
        len(files),
        "" if len(files) == 1 else "s",
        source_branch,
        target_branch,
    )
    return files


# Script entry point.
# ------------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)

    # TODO: Add command line parser.
    source_branch: str | None = None
    target_branch: str = "main"

    if source_branch is None:
        source_branch = git_current_branch()
        logging.info(
            "Source branch wasn't provided, using current branch [%s] as source branch.",
            source_branch,
        )
    else:
        git_check_branch(source_branch)

    git_check_branch(target_branch)

    if target_branch == source_branch:
        logging.error("Source and target branches must be different.")
        sys.exit(1)

    diff_files = git_filename_diff(source_branch, target_branch)
    all_files, config_files, test_files, bench_files = list_triton_source_files()

    diff_inter_triton = diff_files & all_files
    if not diff_inter_triton:
        logging.info(
            "There are no Triton source files in diff, there's no need to run Triton tests."
        )
        sys.exit(0)

    logging.info(
        "There %s %d Triton file%s in the diff:",
        "is" if len(diff_inter_triton) == 1 else "are",
        len(diff_inter_triton),
        "" if len(diff_inter_triton) == 1 else "s",
    )
    for p in sorted(diff_inter_triton):
        logging.info("* %s", p)


if __name__ == "__main__":
    main()
