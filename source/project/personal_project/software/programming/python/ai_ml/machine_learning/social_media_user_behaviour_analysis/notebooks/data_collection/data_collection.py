import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import shutil
    from pathlib import Path

    import kagglehub

    return Path, kagglehub, shutil


@app.cell
def _(kagglehub):
    path = kagglehub.dataset_download("rockyt07/social-media-user-analysis")
    return (path,)


@app.cell
def _(path):
    print("Path to dataset files:", path)
    return


@app.cell
def _(Path):
    source_dir = Path(
        "/home/azmi/.cache/kagglehub/datasets/rockyt07/social-media-user-analysis/versions/2"
    )
    target_dir = Path("./01_Data/01_Raw_Data")
    return source_dir, target_dir


@app.cell
def _(target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    return


@app.cell
def _(shutil, source_dir, target_dir):
    for file_path in source_dir.iterdir():
        if file_path.is_file():
            new_name = (
                "_".join(
                    word.capitalize()
                    for word in file_path.stem.replace("-", "_").split("_")
                )
                + file_path.suffix
            )
            dest_path = target_dir / new_name
            shutil.copy2(file_path, dest_path)
            print(f"Copied: {file_path.name} -> {new_name}")
    return


if __name__ == "__main__":
    app.run()
