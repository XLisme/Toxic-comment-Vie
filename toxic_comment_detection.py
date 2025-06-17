import pandas as pd
import numpy as np
import re
import emoji
from underthesea import word_tokenize, pos_tag, sent_tokenize
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 1. Tiền xử lý dữ liệu
def preprocess_text(text):
    # Chuyển về chữ thường
    text = text.lower()
    
    # Loại bỏ emoji
    text = emoji.replace_emoji(text, replace='')
    
    # Loại bỏ các ký tự đặc biệt và dấu câu
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Chuẩn hóa ký tự lặp lại (ví dụ: nguuuuu -> nguu). Cần đặt sau bước loại bỏ ký tự đặc biệt.
    # Sử dụng regex linh hoạt hơn để bắt các ký tự tiếng Việt có dấu
    text = re.sub(r'([a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎọỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ])\1+', r'\1\1', text)
    
    # Loại bỏ các ký tự thừa (nhiều dấu cách, xuống dòng)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Chuẩn hóa unicode
    text = unicodedata.normalize('NFC', text)
    
    return text

# 2. Trích xuất đặc trưng
def extract_features(df):
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(df['processed_comment'])
    
    # Đặc trưng bổ sung
    def extract_additional_features(text):
        features = {}
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0
        return features
    
    additional_features = df['processed_comment'].apply(extract_additional_features)
    additional_features_df = pd.DataFrame(additional_features.tolist())
    
    return tfidf_features, additional_features_df

# 3. PhoBERT Embedding
def get_phobert_embeddings(texts, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModel.from_pretrained("vinai/phobert-base")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def main():
    # Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df = pd.read_csv('data/ViCTSD_data.csv')
    print(f"Kích thước dữ liệu: {df.shape}")
    
    # Tiền xử lý
    print("\nĐang tiền xử lý dữ liệu...")
    df['processed_comment'] = df['Comment'].apply(preprocess_text)
    
    # Trích xuất đặc trưng
    print("\nĐang trích xuất đặc trưng...")
    tfidf_features, additional_features = extract_features(df)
    
    # Lưu kết quả
    print("\nĐang lưu kết quả...")
    df.to_csv('data/processed_data.csv', index=False)
    
    # Hiển thị thông tin
    print("\nThông tin về dữ liệu:")
    print(f"Số lượng mẫu: {len(df)}")
    print(f"Số lượng toxic: {len(df[df['label'] == 'toxic'])}")
    print(f"Số lượng non-toxic: {len(df[df['label'] == 'non_toxic'])}")
    
    # Vẽ biểu đồ phân phối
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='label')
    plt.title('Phân phối của nhãn toxic/non_toxic')
    plt.savefig('data/label_distribution.png')
    plt.close()

if __name__ == "__main__":
    main() 