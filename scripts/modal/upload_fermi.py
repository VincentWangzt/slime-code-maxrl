from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = REPO_ROOT / "fermi" / "outputs"
REMOTE_OUTPUT_DIRECTORY = "/fermi/outputs"
VOLUME_NAME = "code-maxrl-slime"

ARTIFACT_NAMES = (
    "fermi_train.parquet",
    "fermi_val.parquet",
    "fermi_test.parquet",
    "fermi_analysis.txt",
    "fermi_answer_log_distribution.png",
    "fermi_token_lengths.png",
    "fermi_eval_unit_audit.parquet",
    "fermi_eval_decontaminated.parquet",
)

app = modal.App("slime-upload-fermi")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.local_entrypoint()
def main(output_dir: str = str(DEFAULT_OUTPUT_DIRECTORY)) -> None:
    local_directory = Path(output_dir).resolve()
    artifacts = [local_directory / name for name in ARTIFACT_NAMES]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing Fermi artifacts:\n{formatted}")

    with volume.batch_upload(force=True) as upload:
        for path in artifacts:
            upload.put_file(str(path), f"{REMOTE_OUTPUT_DIRECTORY}/{path.name}")

    print(f"Uploaded {len(artifacts)} Fermi artifacts to Modal Volume {VOLUME_NAME}.")
