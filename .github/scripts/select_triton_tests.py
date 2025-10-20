#!/usr/bin/env python


# Imports
# ------------------------------------------------------------------------------

from pathlib import Path


# Structure of Triton source files.
# ------------------------------------------------------------------------------


def check_dir(p: Path) -> Path:
    assert p.exists(), f"[{p}] doesn't exist."
    assert p.is_dir(), f"[{p}] isn't a directory."
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
    return list_files(TRITON_OP_DIR, suffix=".py")


def list_triton_config_files() -> set[Path]:
    return list_files(TRITON_CONFIG_DIR, suffix=".json")


def list_triton_test_files() -> set[Path]:
    return list_files(TRITON_TEST_DIR, suffix=".py")


def list_triton_bench_files() -> set[Path]:
    # TODO: How to deal with these files?
    #       op_tests/op_benchmarks/triton/utils/model_configs.json
    #       op_tests/op_benchmarks/triton/bench_schema.yaml
    return list_files(TRITON_BENCH_DIR, suffix=".py")


def list_triton_source_files() -> tuple[set[Path], ...]:
    op_files = list_triton_op_files()
    config_files = list_triton_config_files()
    test_files = list_triton_test_files()
    bench_files = list_triton_bench_files()
    all_files = op_files | config_files | test_files | bench_files
    return all_files, config_files, test_files, bench_files


# Script entry point.
# ------------------------------------------------------------------------------


def main() -> None:
    # TODO: Add command line parser.
    source_branch: str | None = None
    target_branch: str = "main"

    all_files, config_files, test_files, bench_files = list_triton_source_files()


if __name__ == "__main__":
    main()
