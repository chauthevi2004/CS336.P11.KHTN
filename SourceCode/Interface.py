MODEL_DIR = './model'
# Change this path if the data is moved to another location
base_path = r"C:\\Users\\chaut\\Downloads\\CS336.P11.KHTN\\Data"

# Paths to the vector databases for image and text embeddings
input_index_path = f"{base_path}\\vectordb-blip2-12\\vector_database.usearch"
input_metadata_path = f"{base_path}\\vectordb-blip2-12\\image_metadata.csv"

# Paths to the vector databases for Vietnamese text embeddings
input_index_path_sen = f"{base_path}\\vietocr-embedding2\\vector_database_text.usearch"
input_metadata_path_sen = f"{base_path}\\vietocr-embedding2\\image_metadata_text.csv"

# Path to the directory containing image frames
dir_img = f"{base_path}\\frames"
local=True

import time
import json
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from usearch.index import Index
from tqdm import tqdm
import matplotlib.pyplot as plt
from lavis.models import load_model_and_preprocess
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from pyvi.ViTokenizer import tokenize
import sys
from contextlib import redirect_stdout, redirect_stderr
from contextlib import contextmanager
import logging
from sklearn.metrics.pairwise import cosine_distances
import streamlit as st
import csv
from io import BytesIO
from io import StringIO
from PIL import Image, UnidentifiedImageError
from sklearn.preprocessing import MinMaxScaler
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F

st.set_page_config(page_title="Video Retrieval Interface", page_icon="🖼️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    /* Set background color for the main section */
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }

    /* Customize button styling */
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 12px 28px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }

    /* Customize input text box styling */
    .stTextInput>div>div>input {
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }

    /* Customize selectbox styling */
    .stSelectbox>div>div>div {
        padding: 10px;
        font-size: 16px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }

    /* Customize text area styling */
    .stTextArea textarea {
        border: 2px solid #ccc; /* Add border for text area */
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }

    /* Customize header (h1) */
    h1 {
        text-align: center;
        color: #FF8C00;
        font-family: 'Georgia', serif;
        font-size: 32px;
    }

    </style>

    <h1> Multimedia Information Retrieval - CS336.P11.KHTN </h1>
""", unsafe_allow_html=True)

# Page title and description
st.title("Final Project - Video Retrieval Interface")
st.markdown("""
    Welcome to the Image Retrieval Interface. Use the form below to enter your prompt and sentence, select the retrieval method, and visualize the results.
    """)

# Suppress logging for specific libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("some_other_library").setLevel(logging.ERROR)
os.environ['TRANSFORMERS_NO_TQDM'] = '1'  # Disable TQDM progress bars for transformers

# Define the device to use 'cuda' or fallback to CPU if CUDA is not available
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device: ",device1)
@st.cache_resource
def load_blip2_model_and_preprocessors(device):
    
    # Load the BLIP2 model checkpoint and preprocessors from the saved file
    checkpoint = torch.load(os.path.join(MODEL_DIR, "blip2_full_model.pth"), map_location=device)
    
    # Initialize the BLIP2 model structure and load the saved state
    model, vis_processors, txt_processors = load_model_and_preprocess(
        "blip2_feature_extractor", "coco", device=device, is_eval=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Use the saved visual and text processors from the checkpoint
    vis_processors = checkpoint['vis_processors']
    txt_processors = checkpoint['txt_processors']
    print("Loading BLIP2 model...: Done")
    
    return model, vis_processors, txt_processors

@st.cache_resource
def load_sentence_transformer_model(device):
    # Load the sentence transformer model and move it to the specified device
    model_emb = SentenceTransformer(os.path.join(MODEL_DIR, "sentence_transformer_model"))
    model_emb = model_emb.to(device)
    print("Loading Sentence Transformer model...: Done")
    return model_emb

@st.cache_resource
def load_translate_model(device):
    # Load the translation model (VietAI/envit5-translation) and tokenizer
    model_name = os.path.join(MODEL_DIR, "translation_model")
    tokenizer1 = AutoTokenizer.from_pretrained(model_name)
    model1 = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print("Loading translation model...: Done")
    return tokenizer1, model1

# Load the models with caching to avoid reloading multiple times
model, vis_processors, txt_processors = load_blip2_model_and_preprocessors(device=device1)
model_emb = load_sentence_transformer_model(device=device1)
tokenizer_trans, model_trans = load_translate_model(device=device1)

def extract_image_features(image):
    image = Image.open(image[0])
    image_tensor = vis_processors["eval"](image.convert("RGB")).to(device1)
    image_tensor = image_tensor.unsqueeze(0)
    sample = {"image": image_tensor}
    with torch.no_grad():
        features_image = model.extract_features(sample, mode="image")
        features_image = torch.mean(features_image.image_embeds, dim=1)
    return features_image

# Function to extract text features using BLIP2
def extract_text_features(text):
    text = txt_processors["eval"](text)
    sample = {"text_input": [text]}
    with torch.no_grad():        
        features_text = model.extract_features(sample, mode="text")
        features_text = torch.mean(features_text.text_embeds, dim=1).squeeze(1)
        
    return features_text

@st.cache_resource
def load_usearch_and_metadata(input_index_path, input_metadata_path):
    dimension = 768  # Ensure this matches the dimension used during creation
    index = Index(ndim=dimension, dtype=np.float16)
    index.load(input_index_path)
    
    # Load metadata
    metadata_df = pd.read_csv(input_metadata_path)
    
    # Print the shape of the DataFrame and the size of the index
    print(f'metadata_df shape: {metadata_df.shape}')
    print(f'index shape: {index.size}')  # Assuming `index` has a `size()` method to get the number of vectors

    
    print('load_usearch_and_metadata: Done')
    return index, metadata_df

@st.cache_resource
def load_sentence_usearch_and_metadata(input_index_path, input_metadata_path):
    dimension_sen = 384  # Example dimension; adjust according to your embedding model
    index_sen = Index(ndim=dimension_sen, dtype=np.float16)
    index_sen.load(input_index_path)
    
    # Load metadata
    metadata_df_sen = pd.read_csv(input_metadata_path)
    
    # Print the shape of the DataFrame and the size of the index
    print(f'metadata_df_sen shape: {metadata_df_sen.shape}')
    print(f'index_sen shape: {index_sen.size}')  # Assuming `index` has a `size()` method to get the number of vectors
    
    print('load_sentence_usearch_and_metadata: Done')
    return index_sen, metadata_df_sen



# DOWNNNNNNNNNNNNNNNNNNNNNNNNN
from langdetect import detect
import pickle
from itertools import combinations

def translate_vietnamese_to_english(prompt_vietnamese):
    if detect(prompt_vietnamese) == 'en':
        return "",prompt_vietnamese,""
    
    else:
        # Nếu có dấu "/", tách chuỗi thành các phần nhỏ và strip() từng phần
        if "/" in prompt_vietnamese:
            parts = [part.strip() for part in prompt_vietnamese.split("/")]
        else:
            parts = [prompt_vietnamese.strip()]

        # Thêm tiền tố "vi: " cho mỗi phần
        inputs = ["vi: " + part for part in parts]

        # Mã hóa các câu đầu vào và tạo batch đầu vào
        input_ids = tokenizer_trans(inputs, return_tensors="pt", padding=True).input_ids.to(device1)

        # Dùng mô hình để tạo bản dịch cho tất cả các câu trong batch
        output_ids = model_trans.generate(input_ids, max_length=512)

        # Giải mã bản dịch từ các token ID
        translations = tokenizer_trans.batch_decode(output_ids, skip_special_tokens=True)

        # Loại bỏ tiền tố "en: " nếu có trong mỗi bản dịch
        translations = [translation[4:] if translation.startswith("en: ") else translation for translation in translations]

        # Kết hợp các bản dịch thành một chuỗi, phân cách bằng "/"
        final_translation = "/".join(translations)

        return "",final_translation,""



# Ensure this function is defined to calculate the combined score with three components
def calculate_combined_score_with_sentence(results, weight_score=0.3, weight_count=0.2, weight_sen=0.5):
    # Normalize the scores
    results['normalized_score'] = (1 - normalize(results['score']))  # Lower scores are better
    results['normalized_count'] = normalize(results['count'])
    results['normalized_sen'] = (1 - normalize(results['sentence_dis']))  # Lower sentence_dis is better-------

    # Calculate the combined score
    results['combined_score'] = (
        weight_score * results['normalized_score'] +
        weight_count * results['normalized_count'] +
        weight_sen * results['normalized_sen']
    )

    return results.sort_values(by='combined_score', ascending=False)


def normalize(series):
    """Normalize a pandas series to a range between 0 and 1."""
    return (series - series.min()) / (series.max() - series.min())

def merge_and_rank_datasets(dataset_1, dataset_2):
    # Merge datasets using outer join
    merged_df = pd.merge(dataset_1, dataset_2, on='image_path', how='outer')

    # Filter to keep rows where 'score_text' and 'score' are not null
    filtered_df = merged_df.dropna(subset=['score_text', 'score']).copy()

    # Rename columns
    filtered_df.rename(columns={'score_text': 'text_similarity_score', 'score': 'image_similarity_score'}, inplace=True)

    # Normalize both text and image similarity scores in one step
    scaler = MinMaxScaler()
    filtered_df[['norm_text_similarity_score', 'norm_image_similarity_score']] = scaler.fit_transform(
        filtered_df[['text_similarity_score', 'image_similarity_score']]
    )

    # Calculate the new score
    filtered_df['score'] = 0.55 * filtered_df['norm_text_similarity_score'] + 0.45 * filtered_df['norm_image_similarity_score']

    # Sort by the calculated score
    ranked_df = filtered_df.sort_values(by='score', ascending=True).copy()

    return ranked_df




def calculate_combined_score(top_k_results, weight_score=0.5, weight_count=0.5):
    # Normalize the 'score' (distance) and invert it so that higher is better
    normalized_score = normalize(top_k_results['score'])
    inverted_score = 1 - normalized_score  # Inverting because lower distance is better

    # Normalize the 'count'
    normalized_count = normalize(top_k_results['count'])

    # Calculate the combined score as a weighted sum
    top_k_results['combined_score'] = (weight_score * inverted_score) + (weight_count * normalized_count)

    # Sort the DataFrame based on the combined score
    ranked_results = top_k_results.sort_values(by='combined_score', ascending=False)

    return ranked_results

def plot_ranked_images(ranked_results, images_per_row=5, max_images=100):
    # Determine the number of images to plot
    num_images = min(ranked_results.shape[0], max_images)
    
    # Calculate the number of rows needed
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    # Create a plot with the appropriate size
    plt.figure(figsize=(20, num_rows * 4))  # Adjust the size based on the number of rows

    for i, (_, row) in enumerate(ranked_results.iloc[:num_images].iterrows()):
        image_path = row["image_path"]
        combined_score = row["combined_score"]
        image = Image.open(image_path)
        
        # Plot the image in the grid
        plt.subplot(num_rows, images_per_row, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Score: {combined_score:.2f}")

    # Display the plot
    plt.tight_layout()
    plt.show()
    
def plot_images(image_paths, images_per_row=5):
    num_images = len(image_paths)
    
    # Calculate the number of rows needed
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    # Create a plot with the appropriate size
    plt.figure(figsize=(20, num_rows * 4))  # Adjust the size based on the number of rows

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        
        # Plot the image in the grid
        plt.subplot(num_rows, images_per_row, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Image {i + 1}")

    # Display the plot
    plt.tight_layout()
    plt.show()
    
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr



# Load index và metadata với caching
index, metadata_df = load_usearch_and_metadata(input_index_path, input_metadata_path)
index_sen, metadata_df_sen = load_sentence_usearch_and_metadata(input_index_path_sen, input_metadata_path_sen)

def query_with_text_emb(sentence, top_k=100, index=index_sen, metadata_df=metadata_df_sen, model=model_emb):
    # Generate the sentence embedding
    sentence_embedding = model.encode(sentence, convert_to_tensor=True).detach().cpu().numpy().astype(np.float16)

    matches = index.search(sentence_embedding, top_k)
    # Retrieve the top indices and their corresponding scores
    top_indices = matches.keys  # Assuming matches.keys is a 1D array for a single query
    top_scores = matches.distances  # Similarly, assuming matches.distances is a 1D array
    
    # Retrieve the top results from metadata_df using the indices
    top_results = metadata_df.iloc[top_indices].copy()
    
    # Add the scores to the results dataframe
    top_results['score_text'] = top_scores

    return top_results

def query_with_text(prompt, top_k=5, index=index, metadata_df=metadata_df):
    # Extract text features
    text_features = extract_text_features(prompt)
    text_features = text_features.cpu().numpy().astype(np.float16)
    
    # Perform the search with usearch
    matches = index.search(text_features, top_k)
    
    # Retrieve the top indices and their corresponding scores
    top_indices = matches.keys  # Assuming matches.keys is a 1D array for a single query
    top_scores = matches.distances  # Similarly, assuming matches.distances is a 1D array
    
    # Retrieve the top results from metadata_df using the indices
    top_results = metadata_df.iloc[top_indices].copy()
    
    # Add the scores to the results dataframe
    top_results['score'] = top_scores

    return top_results



# 100 images
def main3(prompt_vietnamese, top_k=100, index=index, metadata_df=metadata_df, plot=True):
    # Start timing
    start_time = time.time()

    # Step 1: Translate Vietnamese prompt to English
    prompt_output, translated_text, response_time = translate_vietnamese_to_english(prompt_vietnamese)
#     print(f"Translated English text: {translated_text}")

    # Step 2: Query the top 100 results based on the translated text
    top_k_results = query_with_text(translated_text, top_k=top_k, index=index, metadata_df=metadata_df)
    
    # Extract the image paths for plotting
    image_paths = top_k_results['image_path'].tolist()

    return top_k_results, translated_text


# 5000 ảnh + 5000 text_emb --> 100 img
def main4(prompt_vietnamese, sentence, top_k=5000, flag=True, flag2=False):
    dataset_1 = query_with_text_emb(sentence, top_k=top_k)
    dataset_2, translated_text = main3(prompt_vietnamese, top_k=top_k, plot=False)
    # Merge, filter, normalize, and rank the datasets
    final_dataset = merge_and_rank_datasets(dataset_1, dataset_2)
    # Return only the top 100 rows if there are more than 100 rows
    final_dataset = final_dataset.drop_duplicates(subset='image_path', keep='first')
    return final_dataset, translated_text


def main1(prompt_vietnamese, objects, top_k=5000):
    dataset_1, translated_text = main3(prompt_vietnamese, top_k=top_k, plot=False)
    dataset_2, text2print = object(objects, model=model_emb, label_dict=label_dict)

    merged_df = pd.merge(
        dataset_1[['image_path', 'score']].astype({'score': 'float16'}),  # Use float16 for lower memory usage
        dataset_2[['image_path', 'total_confidence']].astype({'total_confidence': 'float16'}),
        on='image_path',
        how='inner'
    )

    # Scale both 'score' and 'total_confidence' columns simultaneously using NumPy for better performance
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(merged_df[['score', 'total_confidence']].values)

    # Assign scaled values directly to the DataFrame
    merged_df['score_scaled'], merged_df['total_conf_scaled'] = scaled_values[:, 0], 1 - scaled_values[:, 1]  # Invert 'total_conf_scaled'

    # Vectorized final score computation using NumPy
    merged_df['final_score'] = np.add(0.65 * merged_df['score_scaled'], 0.35 * merged_df['total_conf_scaled'])

    # In-place sorting by 'final_score' in ascending order (lower is better)
    final_df = merged_df.sort_values(by='final_score', ascending=True, inplace=False)
    
    final_df = final_df.drop_duplicates(subset='image_path', keep='first')

    # Giữ lại chỉ hai cột 'image_path' và 'final_score', sau đó đổi tên cột
    final_df = final_df[['image_path', 'final_score']].rename(columns={'final_score': 'score'})

    return final_df, translated_text + "\n\n\n" + text2print

        
def filter_df_by_frame_gap(df, frame_column, max_frame_gap=200):
    """
    Lọc các frame trong khoảng từ frame nhỏ nhất là idx đến idx + max_frame_gap
    và chỉ giữ lại frame ở vị trí trên cùng trong DataFrame ban đầu cho mỗi nhóm video.
    """
    # Sắp xếp theo frame_column để đảm bảo frame được xử lý tuần tự
    df_sorted = df.sort_values(by=[frame_column])
    df_filtered = []

    # Nhóm theo video_name
    for video_name, group in df_sorted.groupby('video_name'):
        group = group.copy()
        start_idx = 0
        while start_idx < len(group):
            # Lấy phần còn lại của nhóm bắt đầu từ start_idx
            subset_group = group.iloc[start_idx:]

            # Tìm các frame trong khoảng từ [idx : idx + max_frame_gap]
            subset = subset_group[subset_group[frame_column] <= subset_group[frame_column].iloc[0] + max_frame_gap]

            # Chọn frame ở vị trí cao nhất trong DataFrame ban đầu
            highest_row = df.loc[subset.index].iloc[0]  # Lấy hàng ở vị trí cao nhất trong df ban đầu
            df_filtered.append(highest_row)

            # Cập nhật start_idx để bỏ qua khoảng frame đã xét
            start_idx += len(subset)

    # Tạo lại DataFrame từ danh sách các frame đã lọc
    df_filtered = pd.DataFrame(df_filtered)
    df_filtered = df_filtered.drop_duplicates(subset=['image_path_0'])
    return df_filtered



def extract_video_and_frame(image_path):
    parts = image_path.split('/')[-1].split('.')
    video_name = parts[0]
    frame = int(parts[1])
    return video_name, frame



def temporal_search(dataset_subprompts, fps=25, base_gap=10):
    """
    Tìm tất cả các chuỗi frame liên tiếp trong cùng một video thỏa mãn điều kiện thời gian.
    """
    n_prompts = len(dataset_subprompts)
    
    # Tạo max_gap_seconds tự động dựa trên số lượng prompts
    max_gap_seconds = [base_gap * i for i in range(1, n_prompts)]  # max_gap_seconds cho từng frame sau frame đầu
    max_gap_frames = [gap * fps for gap in max_gap_seconds]  # max_gap_frames giữa frame_0 và các frame còn lại
    result = []
    
    # Thêm video_name và frame vào tất cả DataFrame
    for i, df in enumerate(dataset_subprompts):
        df['video_name'], df[f'frame_{i}'] = zip(*df['image_path'].apply(extract_video_and_frame))
        df.rename(columns={'image_path': f'image_path_{i}', 'index': f'index_{i}', 'score': f'score_{i}'}, inplace=True)
        dataset_subprompts[i] = df
    
    max_frame_gap = int(base_gap * fps * 0.2)
    # Bắt đầu với DataFrame đầu tiên đã được lọc
    df_filtered = filter_df_by_frame_gap(dataset_subprompts[0], f'frame_0', max_frame_gap=max_frame_gap)
    
    # Duyệt qua từng hàng trong df_filtered
    for idx, row in df_filtered.iterrows():
        current_video = row['video_name']
        frames = [row[f'frame_0']]
        image_paths = [row[f'image_path_0']]
        total_score = row[f'score_0']
    
        # Tạo danh sách các chuỗi frame tiềm năng
        sequences = [{
            'frames': frames,
            'image_paths': image_paths,
            'total_score': total_score,
            'current_frame': frames[-1]
        }]
    
        # Duyệt qua từng frame tiếp theo
        for i in range(1, n_prompts):
            new_sequences = []
            for seq in sequences:
                current_frame = seq['current_frame']
                # Lấy DataFrame tương ứng
                next_df = dataset_subprompts[i]
                # Tìm tất cả các frame thỏa mãn điều kiện
                next_frames = next_df[
                    (next_df['video_name'] == current_video) & 
                    (next_df[f'frame_{i}'] - seq['frames'][0] <= max_gap_frames[i-1]) &
                    (next_df[f'frame_{i}'] - current_frame > 0)
                ]
                if not next_frames.empty:
                    # Sắp xếp next_frames theo số frame
                    next_frames = next_frames.sort_values(by=f'frame_{i}')
                    for _, next_row in next_frames.iterrows():
                        next_frame_value = next_row[f'frame_{i}']
                        # Tạo chuỗi mới
                        new_seq = {
                            'frames': seq['frames'] + [next_frame_value],
                            'image_paths': seq['image_paths'] + [next_row[f'image_path_{i}']],
                            'total_score': seq['total_score'] + next_row[f'score_{i}'],
                            'current_frame': next_frame_value
                        }
                        new_sequences.append(new_seq)
                # Nếu không tìm thấy frame tiếp theo, không thêm chuỗi mới
            sequences = new_sequences
            # Nếu không còn chuỗi nào, thoát khỏi vòng lặp
            if not sequences:
                break
    
        # Thêm các chuỗi hoàn chỉnh vào kết quả
        for seq in sequences:
            if len(seq['frames']) == n_prompts:
                result.append({
                    "video": current_video,
                    "frames": seq['frames'],
                    "image_paths": seq['image_paths'],
                    "total_score": seq['total_score']
                })
    
    # Sắp xếp kết quả theo total_score (từ nhỏ đến lớn)
    result = sorted(result, key=lambda x: x['total_score'])

    # Loại bỏ các sequences giống nhau (nếu 1 sequences mà có (n_prompts - 1)/n_prompts giống với sequences khác thì chỉ giữ lại 1)
    unique_results = []
    for seq in result:
        is_duplicate = False
        for unique_seq in unique_results:
            shared_frames = set(seq['frames']).intersection(set(unique_seq['frames']))
            if len(shared_frames) >= n_prompts - 1:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_results.append(seq)
    result = unique_results

    return result


def temporal_search_plus(dataset_subprompts, fps=25, base_gap=10):
    """
    Tìm các frame liên tiếp trong cùng một video sử dụng tính năng merge của Pandas và xếp hạng theo tổng score.

    - dataset_subprompts: Danh sách các DataFrame chứa các subprompt.
    - fps: Số frame per second của video (mặc định là 25).
    - base_gap: Khoảng cách cơ bản giữa các frame tính bằng giây (mặc định là 10s).

    Trả về danh sách các frame liên tiếp thỏa mãn điều kiện, được xếp hạng theo tổng score.
    """
    n_prompts = len(dataset_subprompts)
    
    # Tạo max_gap_seconds tự động dựa trên số lượng prompts
    max_gap_seconds = [base_gap * i for i in range(1, n_prompts)]  # max_gap_seconds cho từng frame sau frame đầu
    max_gap_frames = [gap * fps for gap in max_gap_seconds]  # max_gap_frames giữa frame_0 và các frame còn lại
    result = []

    # Thêm video_name và frame vào tất cả DataFrame
    for i, df in enumerate(dataset_subprompts):
        df['video_name'], df[f'frame_{i}'] = zip(*df['image_path'].apply(extract_video_and_frame))
        df = df.rename(columns={'image_path': f'image_path_{i}', 'index': f'index_{i}', 'score': f'score_{i}'})
        dataset_subprompts[i] = df
    
    max_frame_gap= int(base_gap*fps*0.2)
    # Bắt đầu với DataFrame đầu tiên
    df_filtered = filter_df_by_frame_gap(dataset_subprompts[0], f'frame_0', max_frame_gap=max_frame_gap)

    # Chạy qua từng frame tiếp theo và áp dụng điều kiện max_gap_frames
    for _, row in df_filtered.iterrows():
        valid_frames = True
        current_video = row['video_name']
        current_frame = row['frame_0']
        frames = [current_frame]
        image_paths = [row[f'image_path_0']]
        total_score = row[f'score_0']  # Tổng score ban đầu

        # Duyệt qua các DataFrame tiếp theo (frame_1, frame_2, ...)
        for i in range(1, n_prompts):
            next_df = dataset_subprompts[i]
            
            # Tìm frame trong cùng video_name, cách current_frame không quá max_gap_frames[i-1]
            next_frames = next_df[(next_df['video_name'] == current_video) & 
                                  (next_df[f'frame_{i}'] - current_frame <= max_gap_frames[i-1]) &
                                  (next_df[f'frame_{i}'] - current_frame > 0)]
            
            # Sắp xếp next_frames theo số frame trước khi chọn frame đầu tiên
            if not next_frames.empty:
                next_frames = next_frames.sort_values(by=f'frame_{i}')
                next_frame_value = next_frames.iloc[0][f'frame_{i}']

                # Đảm bảo frame tiếp theo lớn hơn frame hiện tại
                if next_frame_value > frames[-1]:
                    frames.append(next_frame_value)
                    image_paths.append(next_frames.iloc[0][f'image_path_{i}'])
                    total_score += next_frames.iloc[0][f'score_{i}']
                else:
                    valid_frames = False
                    break
            else:
                valid_frames = False
                break

        if valid_frames:
            result.append({
                "video": current_video,
                "frames": frames,
                "image_paths": image_paths,
                "total_score": total_score  # Lưu lại tổng score cho mỗi tập hợp kết quả
            })

    # Sắp xếp kết quả theo total_score (từ nhỏ đến lớn)
    result = sorted(result, key=lambda x: x['total_score'])

    return result


def main10(prompt_vietnamese, sentence, top_k=500, flag=True, base_gap=10):

    if "/" in prompt_vietnamese:
        sub_prompts = [sub.strip() for sub in prompt_vietnamese.split('/')]
        prompt_vietnamese = prompt_vietnamese.replace("/", ", ")
    else:
        sub_prompts = [prompt_vietnamese.strip()]
    
    dataset_subprompts = []
    translated_text_all = ""
    
    # Nếu không có sentence hoặc sentence rỗng, áp dụng main3 cho tất cả các prompt
    if sentence is None or sentence == '':
        for prompt in sub_prompts:
            dataset_subprompt, translated_text = main3(prompt, top_k=top_k, plot=False)
            translated_text_all += str(translated_text) + "/ "
            dataset_subprompts.append(dataset_subprompt)
    else:
        # Nếu có sentence, tách thành các câu
        sub_sens = [sub.strip() for sub in sentence.split('/')]
        
        # Nếu số lượng prompts và sentences không bằng nhau, in ra thông báo
        if len(sub_prompts) != len(sub_sens):
            print(f"Warning: Number of prompts ({len(sub_prompts)}) and sentences ({len(sub_sens)}) are not equal.")
        
        # Xử lý các cặp prompt và sen
        for prompt, sen in zip(sub_prompts, sub_sens):
            if sen == '':
                # Nếu sen rỗng, sử dụng main3
                dataset_subprompt, translated_text = main3(prompt, top_k=top_k, plot=False)
            else:
                # Nếu sen không rỗng, sử dụng main4
                dataset_subprompt, translated_text = main4(prompt, sen, top_k=top_k, flag2=True)
                
            translated_text_all += str(translated_text) + "/ "
            dataset_subprompts.append(dataset_subprompt)

    # Thực hiện temporal search
    result = temporal_search_plus(dataset_subprompts, base_gap=base_gap)
    
    # Chỉ giữ lại những item có số lượng image_paths bằng số lượng sub_prompts
    all_image_paths = []
    for item in result:
        if len(item['image_paths']) == len(sub_prompts):
            all_image_paths.extend(item['image_paths'])  # Thêm tất cả image_paths từ mỗi item vào danh sách
    
    df_image_paths = pd.DataFrame({'image_path': all_image_paths})
    
    # Trả về DataFrame chứa image_paths và chuỗi translated_text_all
    return df_image_paths, translated_text_all, len(sub_prompts)
            
def suppress_output():
    return redirect_stdout(sys.stdout), redirect_stderr(sys.stderr)

def run_main_function(main_option, prompt_vietnamese, sentence):    
    if main_option == "Only img":
        a, b = main3(prompt_vietnamese, top_k=500, plot=False)
        return a, b, None  # Return None for img_per_row
    elif main_option == "Img & Text":
        a, b = main4(prompt_vietnamese, sentence, top_k=5000)
        return a, b, None  # Return None for img_per_row
    elif main_option == "Only text":
        a = query_with_text_emb(sentence, top_k=100, index=index_sen, metadata_df=metadata_df_sen, model=model_emb)
        return a, "", None
    elif main_option == "Temporal Search(+)":
        df_image_paths, translated_text_all, img_per_row = main10(prompt_vietnamese, sentence, top_k=top_k, base_gap=base_gap)
        if img_per_row == 1:
            img_per_row = 5
        return df_image_paths, translated_text_all, img_per_row


# Hàm thay đổi đường dẫn của ảnh
def change_path_img(image_paths, dir_img='/kaggle/input/aic-frames/output', local=True):
    if local:
        # Thay thế phần đầu của đường dẫn bằng dir_img, giữ lại tên file
        image_paths = [os.path.join(dir_img, os.path.basename(path)) for path in image_paths]
    return image_paths
    
    
def plot_images_from_csv(df, images_per_row=5, dir_img='', local=True):
    image_paths = df['image_path'].tolist()
    image_paths = change_path_img(image_paths, dir_img, local)
    num_images = len(image_paths)
    
    # Display images in a grid using Streamlit's st.image
    for i in range(0, num_images, images_per_row):
        cols = st.columns(images_per_row)
        for j, col in enumerate(cols):
            if i + j < num_images:
                image_path = image_paths[i + j]
                try:
                    # Open the image
                    image = Image.open(image_path)
                    
                    # Extract the filename and format the caption
                    base_filename = os.path.basename(image_path)
                    # Remove the extension and split by the dot
                    name_parts = base_filename.rsplit('.', 1)[0].split('.')
                    if len(name_parts) == 2:
                        formatted_caption = f"{name_parts[0]}, {name_parts[1]}"
                    else:
                        formatted_caption = base_filename  # Fallback to the original if unexpected format
                    
                    # Display the image with the formatted caption
                    col.image(image, caption=formatted_caption, use_column_width=True)
                except Exception as e:
                    col.error(f"Error loading image: {e}")

def convert_df_to_csv(df: pd.DataFrame, text: str = "") -> str:
    # Giữ lại cột 'image_path'
    df = df[['image_path']].copy()

    # Thay thế '.' bằng ', ' và loại bỏ phần mở rộng '.jpg'
    df['image_path'] = df['image_path'].apply(lambda x: re.sub(r'\.', ', ', x.split('/')[-1].replace('.jpg', '')))

    # Append the provided text to each line in the CSV, only if text is not empty
    if text:
        df['image_path'] = df['image_path'].apply(lambda x: f"{x}, {str(text)}")
    else:
        df['image_path'] = df['image_path'].apply(lambda x: f"{x}")

    # Chuyển đổi DataFrame thành chuỗi CSV mà không có bất kỳ dấu " hoặc \ nào
    csv_data = '\n'.join(df['image_path'].tolist())
    

    return csv_data

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    import base64
    import pickle
    import uuid
    import re

    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None
    else:
        if isinstance(object_to_download, bytes):
            pass
        elif isinstance(object_to_download, pd.DataFrame):
            # Get the text input from Streamlit
            text_to_append = st.session_state.get("text_to_append", "")
            object_to_download = convert_df_to_csv(object_to_download, text_to_append)
        else:
            object_to_download = json.dumps(object_to_download)

    # Encode the object to base64
    try:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    # Generate a unique button ID
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    # Define custom CSS for the download button
    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    # Create the download link using base64 encoding
    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/csv;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

# Layout setup

col1, col2 = st.columns(2)
with col1:
    prompt_vietnamese = st.text_area(
        "Enter Vietnamese prompt:",
        key="prompt_vietnamese",
        height=50  # Adjust the height as needed
    )
    
with col2:
    sentence = st.text_area(
        "Enter sentence (OCR):",
        key="sentence",
        height=50  # Adjust the height as needed
    )

# text_to_append = st.text_input("Enter text to append to CSV:", key="text_to_append")

main_option = st.selectbox(
    "Choose a retrieval method:",
    ("Only img", "Only text", "Img & Text","Temporal Search(+)"),
    key="main_option"
)

# Display sliders only when "Temporal Search" is selected
if main_option == "Temporal Search(+)":
    top_k = st.slider("Select top_k:", min_value=0, max_value=3500, value=1200, step=100, key="top_k_slider")
    base_gap = st.slider("Select base_gap:", min_value=0, max_value=350, value=35, step=1, key="base_gap_slider")
else:
    top_k = None
    base_gap = None
    
st.markdown("---")  # Horizontal rule for separation

# Process the inputs and run the selected function when the "Run Retrieval" button is clicked
if st.button("🔍 Run Retrieval"):
    start_time = time.time()

    with st.spinner("Processing..."):
        # result_df, translated_text, img_per_row = run_main_function(main_option, prompt_vietnamese, sentence, objects=None, audio=None, images=None)
        result_df, translated_text, img_per_row = run_main_function(main_option, prompt_vietnamese, sentence)
        st.session_state["result_df"] = result_df
        st.session_state["images_plotted"] = False  # Reset image plot state
        st.session_state["more_images_plotted"] = False  # Track if more images are plotted

    processing_time = time.time() - start_time

    # Ensure the result_df has at most 60 rows initially for plotting
    result_df_plot = result_df.head(60)
    st.session_state["result_df_plot"] = result_df_plot

    # Store remaining images only if result_df has more than 100 rows
    if len(result_df) > 60:
        if len(result_df) > 500:
            st.session_state["remaining_images"] = result_df.iloc[60:500]
        else:
            st.session_state["remaining_images"] = result_df.iloc[60:]
    else:
        st.session_state["remaining_images"] = None  # No remaining images

    # Display the translated text
    st.markdown("### Translated Text")
    st.write(translated_text)

    # Display the download button (limit to 100 rows)
    download_result_df = result_df.head(1)  # Only keep the first 100 rows for the CSV download
    download_button_str = download_button(download_result_df, "retrieval_results.csv", "Download Results as CSV")
    st.markdown(download_button_str, unsafe_allow_html=True)

    # Plot the first 100 retrieved images
    plot_start_time = time.time()
    st.markdown("### 🎨 Retrieved Images")
    plot_images_from_csv(result_df_plot, images_per_row=img_per_row if img_per_row else 5, dir_img=dir_img, local=local)
    st.session_state["images_plotted"] = True

    plotting_time = time.time() - plot_start_time
    total_time_with_plotting = processing_time + plotting_time

    st.write(f"*Total processing time (without plotting):* {processing_time:.2f} seconds")
    st.write(f"*Total plotting time:* {plotting_time:.2f} seconds")
    st.write(f"*Total execution time (with plotting):* {total_time_with_plotting:.2f} seconds")
    st.markdown("---")


# Button to plot more images, only show if there are remaining images
if st.session_state.get("remaining_images") is not None and st.button("🖼️ More img"):
    remaining_images = st.session_state["remaining_images"]

    if not st.session_state.get("more_images_plotted", False):
        st.markdown("### 🎨 Additional Retrieved Images")
        plot_images_from_csv(remaining_images, images_per_row= 5, dir_img=dir_img, local=local)
        st.session_state["more_images_plotted"] = True


# Ensure the images remain plotted when interacting with the download button
if "result_df" in st.session_state:
    result_df_plot = st.session_state["result_df_plot"]

    if not st.session_state.get("images_plotted", False):
        st.markdown("### 🎨 Retrieved Images")
        plot_images_from_csv(result_df_plot, images_per_row=5, dir_img=dir_img, local=local)
        st.session_state["images_plotted"] = True

    # Ensure only the first 100 rows are available for download
    download_result_df = st.session_state["result_df"].head(1)  
    download_button_str = download_button(download_result_df, "retrieval_results.csv", "Download Results as CSV")
    st.markdown(download_button_str, unsafe_allow_html=True)