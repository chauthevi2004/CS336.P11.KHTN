# Multimedia Information Retrieval - Final Project  


## Project Team  
**Instructor:** **PhD. Ngo Duc Thanh** 

**Members:**

- **Tran Kim Ngoc Ngan** - 22520002 - 22520002@gm.uit.edu.vn
- **Tran Minh Quan** - 22521191 - 22521191@gm.uit.edu.vn
- **Chau The Vi** - 22521653 - 22521653@gm.uit.edu.vn 

## Overview
This project is designed to facilitate multimedia information retrieval by integrating multiple components for video frame extraction, OCR processing, embedding generation, and a user-friendly interface. Below is an overview of the core functionalities and their corresponding files.  

## Project Structure  
The project consists of the following files:  

1. **`extractframe].py`**  
   - Responsible for extracting frames from video files.  
   - This component ensures that the frames are correctly processed and ready for downstream tasks such as OCR and embedding.  

2. **`vectordb-blip2-coco.ipynb`**  
   - This module handles the embedding generation process for video frame features using a pre-trained BLIP2 model with COCO embeddings.  
   - It stores the vector representations in the `usearch` database for efficient retrieval.  

3. **`vietocr.ipynb`**  
   - Focused on Optical Character Recognition (OCR).  
   - Extracts text data from the frames to support text-based information retrieval.  

4. **`vietocr-embed.ipynb`**  
   - Processes the OCR-extracted text to generate embeddings.  
   - Stores the embeddings in the `usearch` database for efficient similarity queries.  

5. **`Interface.py`**  
   - Implements the user interface using **Streamlit**.  
   - Provides a simple and interactive platform for users to query the system and retrieve results.  

## Usage  
To execute the project and interact with the system, follow these steps:  
1. Ensure all dependencies are installed. Refer to the [Dependencies](#dependencies) section for more details.  
2. Run the following command to launch the user interface:  
   ```bash
   streamlit run Interface.py
