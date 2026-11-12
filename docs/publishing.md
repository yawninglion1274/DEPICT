# Publishing

## GitHub

The source repository uses `.gitignore` to exclude `manifest/`,
`complex_field/`, `results/`, `log/`, model weights, binary figure assets,
generated caches and local credentials. Small numerical CSVs needed to
reproduce panels are retained alongside the code.

From the `DEPICT` root, initialize Git and inspect the proposed source release:

```bash
git init -b main
git add --dry-run .
git status --short --ignored
```

Configure your Git author identity and authenticate with GitHub before
committing or pushing. With the GitHub CLI installed:

```bash
gh auth login
gh auth status
```

Create or select a GitHub repository, then use its remote URL to push the
reviewed files. Keep code and data licenses explicit and separate; no project
license has been selected yet.

## Hugging Face

Use a dataset repository for observations, volumes and numerical figure
outputs, and a model repository for checkpoints. Install the Hub client in a
publishing environment and authenticate with a token that can write to the
selected repositories:

```bash
python -m pip install -U huggingface_hub hf_xet
hf auth login
hf auth whoami
export HF_DATASET_REPO="your-account/your-dataset"
export HF_MODEL_REPO="your-account/your-models"
```

Once those repositories exist and their contents are ready for publication,
upload the selected data from the `DEPICT` root:

```python
import os
from huggingface_hub import HfApi

api = HfApi()
api.upload_large_folder(
    repo_id=os.environ["HF_DATASET_REPO"],
    repo_type="dataset",
    folder_path=".",
    allow_patterns=[
        "complex_field/**/*.npy",
        "Figures/**/*.tif", "Figures/**/*.npz", "Figures/**/*.npy",
        "Figures/**/*.csv", "Figures/**/*.json",
        "Figures/**/*.svg", "Figures/**/*.pdf", "Figures/**/*.png",
    ],
    ignore_patterns=[
        "**/cache/**", "**/*_manifest.json", "**/*_validation.json",
        "**/*validation_report.json", "**/threshold_0p54_backup/**",
        "**/threshold_comparison.*",
    ],
)
api.upload_large_folder(
    repo_id=os.environ["HF_MODEL_REPO"],
    repo_type="model",
    folder_path="models",
    allow_patterns=["*.pt"],
)
```

The dataset selection does not include `manifest/`, `results/`, `log/`,
credentials or fonts. Review its CSV and JSON input paths before publishing.
Native postprocessing recordings outside `DEPICT` require a separate data
release; see [Data](data.md).

Add a dataset card describing acquisitions, units, shapes and data licensing,
and a model card describing training data, intended use and checkpoint
compatibility. Replace the example identifiers in the download instructions
with the published repositories.

## Reference Guidelines

- [GitHub: About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
- [GitHub: Best practices for repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories)
- [GitHub: About large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Hugging Face: Upload files](https://huggingface.co/docs/huggingface_hub/guides/upload)

GitHub warns about files over 50 MiB and blocks ordinary Git files over
100 MiB. Keeping scientific arrays and model weights outside the Git history
also keeps source clones small.
