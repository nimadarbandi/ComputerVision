# 0_Project: Background Removal for Multiplex Tissue Imaging

## What this project is
This folder is focused on **background and artifact removal in multiplex pathology images** (WSI and multi-channel microscopy data).

The core goal is to improve image quality before downstream analysis (cell segmentation, biomarker quantification, spatial analysis).


## Why this matters
Multiplex tissue images often contain:
- Autofluorescence and residual staining noise
- Non-tissue background
- Pen/ink artifacts and bright splash-like outliers

If these artifacts are not removed, they can distort biomarker signal estimates and reduce reliability of scientific conclusions.

This is especially important in cancer workflows where researchers need accurate, reproducible quantification. The proposal explicitly targets use in biomedical research contexts (including breast cancer research support).

## What is in this folder

### 1. Project documents
- `4__Proposal_Computer_Vision.pdf`: project motivation, literature context, and research plan.
- `Presentation-WSI.pptx`: presentation of selected methods and final processing results.

### 2. Included third-party method repositories (for comparison/reference)
- `PyHIST-master`: tissue/background segmentation and tiling for whole-slide histology.
- `PENGUIN-main`: multiplex spatial proteomics preprocessing and denoising.
- `HistoQC-master`: slide quality-control toolkit for digital pathology.
- `EntropyMasker-master`: entropy-based automatic foreground masking for WSI.
- `BaSiC-master`: illumination/background correction utilities.
- `he_otsu_thresholding`: H&E-specific Otsu tissue thresholding implementation.

### 3. Local pipeline scripts (custom glue/cleanup code)
- `imageMasker.py`
  - Applies a grayscale mask to an image using binary thresholding and bitwise masking.
  - Used to remove non-foreground regions after mask generation.
- `artifactssmoother.py`
  - Detects local outlier pixels by comparing each pixel to a median-filtered neighborhood.
  - Replaces only detected outliers (conservative cleanup).
- `artifacts-remover.py`
  - Detects small, sharp-intensity segments and replaces them with neighborhood medians.
  - Useful for more aggressive small-artifact cleanup.
- `overlayer.py`
  - Loads cleaned channels and composes an RGB overlay.
  - Maps CH1→Red, CH2→Green, CH3→Blue, and blends CH4 into red/blue to appear purple.
- `splashremover.py`
  - Detects bright connected components using local contrast and removes them (sets to black).
- `splashsoftner.py`
  - Detects bright splash regions and softens them with Gaussian blur instead of hard removal.
- `WSImages/other/WSIViewer.py`
  - Uses OpenSlide to preview channel-specific NDPI thumbnails.

## Practical processing flow in this folder
1. Load/inspect raw channel slides (`WSIViewer.py`).
2. Generate a foreground mask (typically via `EntropyMasker-master` or other included methods).
3. Apply mask to each channel (`imageMasker.py`) to remove background.
4. Remove or smooth local artifacts (`artifactssmoother.py`, `artifacts-remover.py`, `splashremover.py`, `splashsoftner.py`).
5. Merge cleaned channels for visualization (`overlayer.py`).

## Notes from code review
- Several local scripts use **hard-coded absolute paths** (e.g., `/Users/nima/...`).
- To reuse this project on another machine/dataset, first parameterize these paths (CLI args or config file).
- `artifactssmoother.py` and `artifacts-remover.py` already accept an input image path argument:
  - `python artifactssmoother.py <image_path>`
  - `python artifacts-remover.py <image_path>`

## Minimal environment requirements (from used scripts)
- Python 3.x
- `numpy`
- `opencv-python` (`cv2`)
- `scipy`
- `scikit-image`
- `matplotlib`
- `openslide-python` (+ OpenSlide system library) for NDPI/WSI viewing

## Project outcome
This folder is a comparative and applied preprocessing workspace:
- It gathers multiple open-source methods relevant to multiplex/WSI cleanup.
- It adds custom scripts to build an end-to-end background/artifact removal pipeline.
- It supports producing cleaner channel images and clearer overlays for downstream biomedical image analysis.
