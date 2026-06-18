import os
import subprocess
import shutil
import modal

app = modal.App("spectrum-matching-experiments")

data_vol = modal.Volume.from_name("celeba-data", create_if_missing=True)
results_vol = modal.Volume.from_name("experiment-results", create_if_missing=True)
cache_vol = modal.Volume.from_name("cleanfid-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "accelerate>=1.13.0", "clean-fid>=0.1.35", "diffusers>=0.37.1",
        "matplotlib>=3.10.9", "numpy>=2.4.4", "opencv-python>=4.13.0.92",
        "pandas>=3.0.3", "scikit-learn>=1.8.0", "seaborn>=0.13.2",
        "torch>=2.11.0", "torch-fidelity>=0.4.0", "torchmetrics>=1.9.0",
        "torchvision>=0.26.0", "transformers>=5.8.1", "tqdm"
    )
    .run_commands(
        "python -c \"from cleanfid.features import build_feature_extractor; build_feature_extractor('clean', 'cpu')\""
    )
    .add_local_dir("./repa", remote_path="/project/repa")
    .add_local_dir("./diffuser", remote_path="/project/diffuser")
    .add_local_file("./data/celeba.zip", remote_path="/project/celeba.zip")
)


@app.function(
    image=image,
    gpu="A10",
    timeout=86400,
    volumes={
        "/data": data_vol,
        "/results": results_vol,
        "/root/.cache/cleanfid": cache_vol
    }
)
def run_all_experiments():
    os.chdir("/project")

    # 1. Dataset Extraction Logic
    # ---------------------------------------------------------
    # Target path inside the persistent volume
    local_data_dir = "/tmp/data"
    dataset_path = os.path.join(local_data_dir, "celeba")

    if not os.path.exists(dataset_path):
        print("Unzipping dataset to local fast storage...")
        os.makedirs(local_data_dir, exist_ok=True)
        shutil.unpack_archive("/project/celeba.zip", local_data_dir)

        # remove macosx metadata
        macosx_junk_path = os.path.join(local_data_dir, "__MACOSX")
        if os.path.exists(macosx_junk_path):
            shutil.rmtree(macosx_junk_path)
            print("Purged macOS metadata directory.")

        # Also purge any loose hidden files in the actual dataset folder
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.startswith("._"):
                    os.remove(os.path.join(root, file))
                    
        print("Extraction complete.")
    else:
        print("Dataset already extracted locally.")
    # ---------------------------------------------------------

    experiments = [
        ("sit", "vanilla", 0.0),
        ("sit", "repa", 0.4),
        ("sit", "irepa", 1.0),
        ("sit", "dog", 1.0),
        ("unet", "vanilla", 0.0),
        ("unet", "repa", 0.4),
        ("unet", "irepa", 1.0),
        ("unet", "dog", 1.0)
    ]

    for arch, mode, lambda_repa in experiments:
        cmd = [
            "python", "-m", "repa.main",
            "--model_type", arch,
            "--mode", mode,
            "--lambda_repa", str(lambda_repa),
            "--max_steps", "10000",
            "--num_evals", "15",
            "--batch_size", "32",
            "--data_dir", local_data_dir,
            "--output_dir", f"/results/{arch}_{mode}"
        ]

        print(f"\n==== Executing: Model={arch} | Mode={mode} ====")

        env = os.environ.copy()
        env["PYTHONPATH"] = "/project"

        subprocess.run(cmd, check=True, env=env)


@app.local_entrypoint()
def main():
    run_all_experiments.remote()