<p align="center">
  <a href="https://www.uit.edu.vn/"><img src="https://www.uit.edu.vn/sites/vi/files/banner.png"></a>
<h1 align="center"><b>CS336.P11.KHTN - Multimedia Information Retrieval</b></h1>

# Project Title: VIDEO RETRIEVAL SYSTEM

This repository implements a video retrieval system that uses frame extraction, embeddings, and an interactive interface for retrieving video frames based on text or multimodal inputs.

---

<p align="center">
  <a href="https://www.uit.edu.vn/"><img src="https://www.uit.edu.vn/sites/vi/files/banner.png"></a>
<h1 align="center"><b>CS519.O21.KHTN - Scientific Research Methodology</b></h1>

# CS336.P11.KHTN - Video Retrieval System

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [File Descriptions](#file-descriptions)
7. [Acknowledgements](#acknowledgements)
8. [Team Information](#team-information)

---

## Overview

This project implements a **Video Retrieval System** that supports extracting frames from videos, processing them with Optical Character Recognition (OCR), generating embeddings, and performing efficient retrieval using text or multimodal inputs. The system is equipped with a user-friendly interface built with Streamlit.

---

## Features

- **Frame Extraction:** Extracts key frames from videos based on scene detection.
- **Optical Character Recognition (OCR):** Extracts text from video frames, supporting Vietnamese language processing.
- **Embeddings:** Generates embeddings for text and video frames for efficient similarity-based search.
- **Multimodal Retrieval:** Combines text and frame embeddings to perform more comprehensive searches.
- **Interactive Interface:** Streamlit-based interface for easy interaction.
- **Temporal Search:** Retrieves sequential frames aligned with temporal input prompts.

---

## Directory Structure

```
.
├── SourceCode/           # Core scripts and notebooks
│   ├── extractframe.py       # Frame extraction script
│   ├── Interface.py          # Streamlit-based interface
│   ├── vietocr-embed.ipynb   # Vietnamese text embedding generation
│   ├── vietocr.ipynb         # OCR processing notebook
│   └── vectordb-blip2-coco.ipynb # Image embedding generation
├── Data/                # Sample data and processed resources
│   ├── L01_V001.mp4         # Sample video file
│   ├── frame/               # Extracted video frames
│   ├── vietocr-embedding2/  # Vietnamese text embedding database
│   │   ├── vector_database_text.usearch
│   │   └── image_metadata_text.csv
│   ├── vectordb-blip2-12/   # Image embedding database
│       ├── vector_database.usearch
│       ├── image_metadata.csv
│       ├── removed_images.pkl
│       └── full/            # Additional embedding data
├── requirements.txt      # Dependencies
├── README.md             # Documentation
```

---

## Installation

### Prerequisites

- Python 3.8+
- `ffmpeg` (for video processing)
- CUDA (optional, for GPU acceleration)

### Steps

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd video-retrieval
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the following directories and files:
   - `SourceCode/`: Contains main scripts.
   - `Data/`: Includes sample videos and preprocessed resources.

---

## Usage

### Step 1: Extract Frames

1. Place your video files in the `Data/` folder.
2. Run the frame extraction script:
   ```bash
   python SourceCode/extractframe.py
   ```
   Frames and a CSV file with timestamps will be saved in a new directory named after the video.

### Step 2: Generate Embeddings

1. Open `vietocr-embed.ipynb` for text embeddings and run all cells.
2. Open `vectordb-blip2-coco.ipynb` for image embeddings and execute the notebook.

### Step 3: Launch the Interface

1. Start the Streamlit interface:
   ```bash
   streamlit run SourceCode/Interface.py
   ```
2. Access the interface through the local server URL provided by Streamlit.

### Step 4: Query and Retrieve Results

- Input text, reference frames, or both.
- Use temporal search to retrieve sequences of frames matching temporal prompts.

---

## File Descriptions

### `extractframe.py`

- **Purpose:** Extracts key frames from videos based on scene changes.
- **Key Functions:**
  - `process_video(video_path)`: Detects scenes and extracts frames.
  - `process_all_videos_in_folder(folder_path)`: Processes all videos in a specified folder.

### `Interface.py`

- **Purpose:** Implements a Streamlit-based user interface for video retrieval.
- **Features:**
  - Text-to-frame and multimodal search.
  - Temporal sequence retrieval.
  - Interactive image display and download options.

### `vietocr-embed.ipynb`

- **Purpose:** Generates embeddings for Vietnamese text extracted from video frames.
- **Usage:**
  - Extract text using OCR.
  - Generate embeddings for similarity-based retrieval.

### `vietocr.ipynb`

- **Purpose:** Performs OCR on video frames.
- **Usage:**
  - Detects and processes Vietnamese text.

### `vectordb-blip2-coco.ipynb`

- **Purpose:** Generates BLIP2-based embeddings for video frames.
- **Usage:**
  - Extracts visual features and stores them in a vector database for retrieval.

---

## Acknowledgements

This project utilizes the following tools and libraries:

- [SceneDetect](https://github.com/Breakthrough/PySceneDetect): Video scene detection.
- [BLIP2](https://github.com/salesforce/LAVIS): Visual feature extraction.
- [VietOCR](https://github.com/pbcquoc/vietocr): Vietnamese OCR.
- [Sentence Transformers](https://www.sbert.net/): Text embeddings.

---

## Team Information

### Instructor

- **PhD Ngo Duc Thanh**
  - Email: [thanhnd@uit.edu.vn](mailto:thanhnd@uit.edu.vn)

### Team Members

| STT | Name               | MSSV     | Email                         |
| --- | ------------------ | -------- | ----------------------------- |
| 1   | Tran Kim Ngoc Ngan | 22520002 | 22520002@gm.uit.edu.vn       |
| 2   | Tran Minh Quan     | 22521191 | 22521191@gm.uit.edu.vn       |
| 3   | Chau The Vi        | 22521653 | 22521653@gm.uit.edu.vn       |

