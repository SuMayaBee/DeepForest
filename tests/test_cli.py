import os
import subprocess
import sys
from importlib.resources import files

import pandas as pd
from omegaconf import OmegaConf

from deepforest import get_data
from deepforest.scripts.sweep_scores import normalize_thresholds, plot_pr_curve

SCRIPT = files("deepforest.scripts").joinpath("cli.py")


def test_train_cli(tmpdir):
    """Check a basic training run, including overrides for unit testing
    see test_main.py fixtures for setup reference."""

    test_labels = get_data("OSBS_029.csv")

    args = [
        sys.executable,
        str(SCRIPT),
        "train",
        "train.fast_dev_run=True",
        f"train.csv_file={test_labels}",
        f"train.root_dir={os.path.dirname(test_labels)}"
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_train_cli_fail(tmpdir):
    """Check that training fails if no dataset paths are provided"""

    args = [
        sys.executable,
        str(SCRIPT),
        "train",
        "train.fast_dev_run=True",
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode != 0


def test_train_cli_user_config(tmpdir):
    """Check whether we can provide a custom YAML file for configuration"""

    # Create a modified config
    test_labels = get_data("OSBS_029.csv")
    config = OmegaConf.load(get_data("config.yaml"))
    config.train.csv_file = test_labels
    config.train.root_dir = os.path.dirname(test_labels)
    OmegaConf.save(config, tmpdir.join("user_config.yaml").open('w'))

    # This will fail if the config is not correctly created
    # as the csv/root parameters are not set by default.
    args = [
        sys.executable,
        str(SCRIPT),
        f"--config-dir", tmpdir,
        f"--config-name", "user_config",
        "train",
        "train.fast_dev_run=True"
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_predict_cli(tmp_path):
    """Check we can predict an image and save results"""
    input_path = get_data("OSBS_029.png")
    output_path = tmp_path / "result.csv"
    args = [input_path, "-o", str(output_path)]

    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_path.exists(), f"Expected output file not found: {output_path}"


def test_predict_cli_with_opt(tmp_path):
    """Check we can predict an image and save results"""
    input_path = get_data("OSBS_029.png")
    output_path = tmp_path / "result.csv"
    args = [input_path, "-o", str(output_path), "patch_size=250"]

    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_path.exists(), f"Expected output file not found: {output_path}"


def test_predict_cli_missing_input(tmp_path):
    # Running the script without any inputs should yield an error
    result = subprocess.run(
        [sys.executable, SCRIPT, "predict"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0


def test_predict_cli_config_help(tmp_path):
    # Script should show config without requiring input
    result = subprocess.run(
        [sys.executable, SCRIPT, "config"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert len(result.stdout) > 0


# --- sweep-scores tests ---


def test_normalize_thresholds_defaults():
    """No input returns 10 thresholds between 0.0 and 0.9."""
    result = normalize_thresholds(None)
    assert len(result) == 10
    assert result == sorted(result)
    assert result[0] == 0.0
    assert result[-1] == 0.9


def test_normalize_thresholds_custom():
    """Custom thresholds are deduplicated and sorted."""
    result = normalize_thresholds([0.5, 0.3, 0.5, 0.1])
    assert result == [0.1, 0.3, 0.5]


def test_plot_pr_curve_creates_file(tmp_path):
    """plot_pr_curve writes a PNG to the given path."""
    df = pd.DataFrame(
        {
            "score_thresh": [0.1, 0.3, 0.5],
            "box_precision": [0.9, 0.8, 0.95],
            "box_recall": [0.7, 0.6, 0.4],
        }
    )
    output_path = str(tmp_path / "pr_curve.png")
    plot_pr_curve(df, output_path, label_thresholds=True)
    assert os.path.exists(output_path)


def test_plot_pr_curve_empty_df(tmp_path):
    """plot_pr_curve handles an empty DataFrame without error."""
    df = pd.DataFrame(
        {"score_thresh": [], "box_precision": [], "box_recall": []}
    )
    output_path = str(tmp_path / "pr_curve.png")
    plot_pr_curve(df, output_path)
    assert not os.path.exists(output_path)


def test_sweep_scores_cli_missing_validation(tmp_path):
    """sweep-scores subcommand fails when validation config is not set."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "sweep-scores",
            "--output-dir",
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0


def test_sweep_scores_cli(tmp_path):
    """sweep-scores subcommand writes CSV and PNG to output-dir."""
    test_labels = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(test_labels)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "sweep-scores",
            "--output-dir",
            str(tmp_path),
            "--thresholds", "0.3", "0.5",
            f"validation.csv_file={test_labels}",
            f"validation.root_dir={root_dir}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert (tmp_path / "precision_recall_thresholds.csv").exists()
    assert (tmp_path / "precision_recall_curve.png").exists()
