### README
<p align="center">
  <a href="https://www.uit.edu.vn/"><img src="https://www.uit.edu.vn/sites/vi/files/banner.png"></a>
<h1 align="center"><b>CS519.O21.KHTN - Scientific Research Methodology</b></h1>

# Project Title: VIDEO RETRIEVAL SYSTEM

This repository implements a video retrieval system that uses frame extraction, embeddings, and an interactive interface for retrieving video frames based on text or multimodal inputs.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [How to Run](#how-to-run)
5. [File Descriptions](#file-descriptions)
6. [Example Use Cases](#example-use-cases)
7. [Acknowledgements](#acknowledgements)
8. [Data Folder Description](#data-folder-description)

---

## **Overview**

This system enables users to:

- Extract frames from videos based on scene changes.
- Generate embeddings for text and video frames.
- Perform multimodal retrieval (text-to-frame or frame-to-frame search).
- Use an interactive interface for exploring and ranking retrieved results.

---

## **Directory Structure**

```
├── extractframe.py           # Frame extraction from videos
├── Interface.py              # Streamlit-based video retrieval interface
├── vietocr-embed.ipynb       # Generate Vietnamese text embeddings
├── vietocr.ipynb             # OCR for text recognition
├── vectordb-blip2-coco.ipynb # BLIP2-based image embedding creation
├── model/                    # Folder for models and checkpoints
├── frames/                   # Folder containing extracted video frames
├── vectordb/                 # Vector databases for embeddings
└── videos/                   # Folder containing input video files
```

---

## **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/video-retrieval.git
   cd video-retrieval
   ```

2. Install the required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the following additional tools installed:

   - `ffmpeg` for video processing.
   - `CUDA` for GPU acceleration (if available).

---

## **How to Run**

### **Step 1: Extract Frames from Videos**

1. Place your video files in the `videos/` directory.
2. Run `extractframe.py` to extract frames based on scene changes:
   ```bash
   python extractframe.py
   ```
   Frames will be saved in a new directory named after the video file, along with a CSV file containing frame timestamps.

### **Step 2: Generate Embeddings**

1. **Text Embeddings (Vietnamese):**

   - Open `vietocr-embed.ipynb` and run all cells to generate text embeddings.

2. **Image Embeddings:**

   - Open `vectordb-blip2-coco.ipynb` and execute the notebook to create BLIP2-based embeddings for extracted frames.

### **Step 3: Start the Interface**

1. Launch the Streamlit interface:

   ```bash
   streamlit run Interface.py
   ```

2. Use the web interface to perform searches. Options include:

   - Text-based retrieval.
   - Multimodal (text and frame) retrieval.
   - Temporal search for sequential frames.

---

## **File Descriptions**

### `extractframe.py`

- **Purpose:** Extracts key frames from videos using scene detection.
- **Key Functions:**
  - `process_video(video_path)`: Detects scenes and saves the first frame of each scene.
  - `process_all_videos_in_folder(folder_path)`: Processes all videos in a directory.

### `Interface.py`

- **Purpose:** Provides an interactive interface for video retrieval.
- **Features:**
  - BLIP2-based image retrieval.
  - Sentence Transformer-based text embedding search.
  - Multimodal retrieval combining text and frame similarity.
  - Temporal search for related video frames.

### `vietocr-embed.ipynb`

- **Purpose:** Generates embeddings for Vietnamese text using OCR and a Sentence Transformer model.
- **Usage:**
  - Load text from the video frames.
  - Generate embeddings for each text snippet.

### `vietocr.ipynb`

- **Purpose:** Performs OCR on video frames to extract text.
- **Usage:**
  - Detects and processes text in Vietnamese.

### `vectordb-blip2-coco.ipynb`

- **Purpose:** Creates BLIP2-based embeddings for video frames.
- **Usage:**
  - Extracts visual features from frames and saves them to a vector database.

---

## **Example Use Cases**

### **Text-to-Video Retrieval**

1. Enter a Vietnamese or English description in the interface.
2. View the retrieved frames that best match your input.

### **Multimodal Retrieval**

1. Input both a description and a reference frame.
2. The system retrieves frames matching both inputs.

### **Temporal Search**

1. Specify a sequence of subprompts (e.g., multiple descriptions for different scenes).
2. Retrieve sequences of frames that align with the temporal order of the subprompts.

---

## **Acknowledgements**

This project uses:

- [SceneDetect](https://github.com/Breakthrough/PySceneDetect) for scene detection.
- [BLIP2](https://github.com/salesforce/LAVIS) for visual feature extraction.
- [VietOCR](https://github.com/pbcquoc/vietocr) for Vietnamese OCR.
- [Sentence Transformers](https://www.sbert.net/) for text embeddings.

---

## **Group Information**

### **Course Information**

- Course Name: MULTIMEDIA INFORMATION RETRIEVAL
- Class Code: CS336.P11.KHTN

### **Instructor**

- **PhD Ngo Duc Thanh**  
  - Email: thanhnd@uit.edu.vn

### **Team Members**

| STT | Name               | MSSV     | Email                                                   | GitHub |
| --- | ------------------ | -------- | ------------------------------------------------------- | ------ |
| 1   | Tran Kim Ngoc Ngan | 22520002 | [22520002@gm.uit.edu.vn](mailto:22520002@gm.uit.edu.vn) |        |
| 2   | Tran Minh Quan     | 22521191 | [22521191@gm.uit.edu.vn](mailto:22521191@gm.uit.edu.vn) |        |
| 3   | Chau The Vi        | 22521653 | [22521653@gm.uit.edu.vn](mailto:22521653@gm.uit.edu.vn) |        |

---

## **Data Folder Description**

### Overview

The `Data` folder contains all resources required for video retrieval, including videos, extracted frames, embeddings, and vector databases.

### Folder Structure

```
Data/
├── L01_V001.mp4          # Sample video
├── frame/                # Extracted frames from videos
│   ├── L01_V001.47.jpg   # Example frame
│   ├── L01_V001.531.jpg
│   └── ...
├── vietocr-embedding2/   # Text embedding resources
│   ├── vector_database_text.usearch  # Text embedding vector database
│   └── image_metadata_text.csv       # Metadata for text embeddings
├── vectordb-blip2-12/    # Image embedding resources
│   ├── vector_database.usearch      # Image embedding vector database
│   ├── image_metadata.csv           # Metadata for image embeddings
│   ├── removed_images.pkl           # Log of removed images
│   └── full/                        # Raw embedding data
```

### Details

1. ``

   - A sample video used for testing frame extraction and retrieval.

2. ``

   - Contains extracted frames from videos. Each frame is saved as a `.jpg` file named in the format `<video_name>.<frame_number>.jpg`.

3. ``

   - Stores the vector database for text embeddings (`vector_database_text.usearch`).
   - Includes metadata (`image_metadata_text.csv`) linking frames to their corresponding text embeddings.

4. ``

   - Contains the vector database for image embeddings (`vector_database.usearch`).
   - Metadata (`image_metadata.csv`) provides mappings between frames and embeddings.
   - `removed_images.pkl` logs any frames excluded during embedding creation.
   - The `full/` directory includes raw data files used to generate embeddings.
