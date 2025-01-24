<p align="center">
  <a href="https://www.uit.edu.vn/"><img src="https://www.uit.edu.vn/sites/vi/files/banner.png"></a>
<h2 align="center"><b>CS336.P11.KHTN - Multimedia Information Retrieval</b></h2>

---

# EVENT RETRIEVAL FROM VIDEO

---

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

This project implements an **Event Retrieval System** that supports extracting frames from videos, processing them with Optical Character Recognition (OCR), generating embeddings, and performing efficient retrieval using text or multimodal inputs. The system is equipped with a user-friendly interface built with Streamlit.

---

## Features

- **Frame Extraction:** Extracts key frames from videos based on scene detection using `scenedetect`.
- **Optical Character Recognition (OCR):** Extracts text from video frames, supporting Vietnamese language processing with `VietOCR`.
- **Embeddings:** Generates embeddings for text and video frames using `Sentence Transformers` and `BLIP2` for efficient similarity-based search.
- **Multimodal Retrieval:** Combines text and frame embeddings to perform more comprehensive searches.
- **Interactive Interface:** Streamlit-based interface for seamless interaction.
- **Temporal Search:** Retrieves sequential frames aligned with temporal input prompts to facilitate structured video analysis.

---

## Directory Structure

```
.
├── Data/                # Contains the dataset and processed resources
│   ├── L01_V001.mp4         # Sample video file
│   ├── frame/               # Directory containing extracted frames
│   ├── vietocr-embedding2/  # Resources for text embeddings
│   │   ├── vector_database_text.usearch
│   │   └── image_metadata_text.csv
│   ├── vectordb-blip2-12/   # Resources for image embeddings
│       ├── vector_database.usearch
│       ├── image_metadata.csv
│       ├── removed_images.pkl
│       └── full/            # Directory for raw embedding data
├── SourceCode/           # Contains the source code for the project
│   ├── extractframe.py       # Script for extracting frames from videos
│   ├── Interface.py          # Streamlit-based retrieval interface
│   ├── vietocr-embed.ipynb   # Notebook for Vietnamese text embeddings
│   ├── vietocr.ipynb         # Notebook for OCR processing
│   └── vectordb-blip2-coco.ipynb # Notebook for image embedding generation
├── model/                # Contains pretrained models
│   ├── sentence_transformer_model/  # Sentence Transformer model
│   ├── translation_model/           # Translation model
│   └── blip2_full_model.pth         # BLIP2 model checkpoint
├── requirements.txt      # Dependencies
├── README.md             # Documentation
```

---

## Installation

### Prerequisites

- Python 3.11.20
- `ffmpeg` (for video processing)
- CUDA (optional, for GPU acceleration)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/chauthevi2004/CS336.P11.KHTN.git
   ```

2. Create and activate the Conda environment:
   ```bash
   conda create -n ER-env python=3.11 -y
   conda activate ER-env
   ```

3. Upgrade `pip` and install dependencies:
   ```bash
   pip install --upgrade pip
   pip install salesforce-lavis
   pip install opencv-python==4.10.0.84
   pip install sentence-transformers==2.2.2
   pip install transformers==4.26.1
   pip install huggingface-hub==0.12.1
   pip install -r requirements.txt
   ```

4. Verify the following directories and files:
   - `SourceCode/`: Contains main scripts.
   - `Data/`: Includes sample videos and preprocessed resources.
   - `model/`: Stores pretrained models required for embeddings and translation.

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

- **Text Retrieval:** Input descriptive text and retrieve matching frames.
- **Multimodal Retrieval:** Combine text and image input for enhanced search results.
- **Temporal Search:** Define sequences of temporal prompts to retrieve aligned video segments.

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
- **Usage:** Detects and processes Vietnamese text.

### `vectordb-blip2-coco.ipynb`

- **Purpose:** Generates BLIP2-based embeddings for video frames.
- **Usage:** Extracts visual features and stores them in a vector database for retrieval.

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

  Email: [thanhnd@uit.edu.vn](mailto:thanhnd@uit.edu.vn)

### Team Members

| STT | Name               | MSSV     | Email                         |
| --- | ------------------ | -------- | ----------------------------- |
| 1   | Tran Kim Ngoc Ngan | 22520002 | 22520002@gm.uit.edu.vn       |
| 2   | Tran Minh Quan     | 22521191 | 22521191@gm.uit.edu.vn       |
| 3   | Chau The Vi        | 22521653 | 22521653@gm.uit.edu.vn       |

### Data and Model Resources

Due to size limitations on GitHub, the `Data` and `model` directories are hosted on Google Drive. You can download them from the following link:

- [Google Drive: Data and Models](https://drive.google.com/drive/folders/1Z0wAtb6-KvkreyvOZNH1O_sOB_P6iQNA?usp=sharing)

After downloading, ensure the directories are placed in the root project folder as described in the **Directory Structure** section.
