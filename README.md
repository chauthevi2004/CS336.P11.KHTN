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
8. [Group Information](#group-information)

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
CS336.P11.KHTN/
├── SourceCode/           # Contains the source code for the project
│   ├── extractframe.py       # Script for extracting frames from videos
│   ├── Interface.py          # Streamlit-based retrieval interface
│   ├── vietocr-embed.ipynb   # Notebook for Vietnamese text embeddings
│   ├── vietocr.ipynb         # Notebook for OCR processing
│   └── vectordb-blip2-coco.ipynb # Notebook for image embedding generation
├── Data/                # Contains the dataset and processed resources
│   ├── L01_V001.mp4         # Sample video file
│   ├── frame/               # Directory containing extracted frames
│   ├── vietocr-embedding2/  # Resources for text embeddings
│   │   ├── vector_database_text.usearch  # Text embedding vector database
│   │   └── image_metadata_text.csv       # Text embedding metadata
│   ├── vectordb-blip2-12/   # Resources for image embeddings
│   │   ├── vector_database.usearch       # Image embedding vector database
│   │   ├── image_metadata.csv            # Metadata for image embeddings
│   │   ├── removed_images.pkl            # Log of removed images
│   │   └── full/                         # Directory for raw embedding data
```

### **SourceCode Folder**

- **extractframe.py**: Script to extract frames from videos using scene detection.
- **Interface.py**: Provides an interactive Streamlit interface for video retrieval.
- **vietocr-embed.ipynb**: Generates text embeddings for Vietnamese text extracted from video frames.
- **vietocr.ipynb**: Processes frames to extract Vietnamese text using OCR.
- **vectordb-blip2-coco.ipynb**: Creates image embeddings using BLIP2 and stores them in a vector database.

### **Data Folder**

- **L01_V001.mp4**: A sample video file for testing the retrieval system.
- **frame/**: Contains frames extracted from the sample video, each named in the format `<video_name>.<frame_number>.jpg`.
- **vietocr-embedding2/**:
  - `vector_database_text.usearch`: Vector database of text embeddings for the OCR-processed text.
  - `image_metadata_text.csv`: Metadata mapping text embeddings to their corresponding frames.
- **vectordb-blip2-12/**:
  - `vector_database.usearch`: Vector database for image embeddings created from video frames.
  - `image_metadata.csv`: Metadata mapping image embeddings to their respective frames.
  - `removed_images.pkl`: Contains information about frames removed during preprocessing.
  - `full/`: Stores raw data and additional embeddings for images.

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

