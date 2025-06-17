import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Lấy text và label theo index
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ToxicCommentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ToxicCommentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0, :]
        output = self.drop(pooled_output)
        return self.fc(output)

def balance_data(X, y):
    """Cân bằng dữ liệu sử dụng RandomUnderSampler"""
    # Chỉ sử dụng RandomUnderSampler vì SMOTE không hoạt động tốt với dữ liệu văn bản
    rus = RandomUnderSampler(random_state=42)
    
    # Chuyển đổi X thành ma trận 2D
    X_2d = np.array(X).reshape(-1, 1)
    
    # Áp dụng RandomUnderSampler
    X_balanced, y_balanced = rus.fit_resample(X_2d, y)
    
    # In thông tin về số lượng mẫu sau khi cân bằng
    print("Số lượng mẫu sau khi cân bằng:")
    print(Counter(y_balanced))
    
    return X_balanced.ravel(), y_balanced

def prepare_data():
    """Chuẩn bị dữ liệu cho training"""
    # Đọc dữ liệu đã xử lý
    df = pd.read_csv('data/processed_data.csv')
    
    # Chuyển đổi nhãn thành số
    label_map = {'non_toxic': 0, 'toxic': 1}
    df['label'] = df['label'].map(label_map)
    
    # In thông tin về phân phối nhãn
    print("Phân phối nhãn ban đầu:")
    print(df['label'].value_counts())
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_comment'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Chia tập train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    # Cân bằng dữ liệu training
    X_train_balanced, y_train_balanced = balance_data(
        X_train.values,
        y_train.values
    )
    
    # Chuyển đổi thành Series để giữ index
    X_train_balanced = pd.Series(X_train_balanced, index=X_train.index[:len(X_train_balanced)])
    y_train_balanced = pd.Series(y_train_balanced, index=X_train.index[:len(y_train_balanced)])
    
    return (X_train_balanced, y_train_balanced,
            X_val, y_val,
            X_test, y_test)

def train_model(X_train, y_train, X_val, y_val, batch_size=16, epochs=5):
    """Huấn luyện mô hình"""
    # Khởi tạo tokenizer và dataset
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    train_dataset = ToxicCommentDataset(X_train, y_train, tokenizer)
    val_dataset = ToxicCommentDataset(X_val, y_val, tokenizer)
    
    # Tạo dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Khởi tạo model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ToxicCommentClassifier(n_classes=2)
    model = model.to(device)
    
    # Định nghĩa optimizer và loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    class_weights = torch.tensor([1.0, 8.22], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(classification_report(val_labels, val_preds))
        
        # Lưu model tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def evaluate_model(model, X_test, y_test, batch_size=16):
    """Đánh giá mô hình trên tập test"""
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    test_dataset = ToxicCommentDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # In báo cáo phân loại
    print("\nTest Results:")
    print(classification_report(test_labels, test_preds))
    
    # Vẽ confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Chuẩn bị dữ liệu
    print("Chuẩn bị dữ liệu...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    
    # Huấn luyện mô hình
    print("\nBắt đầu huấn luyện mô hình...")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Đánh giá mô hình
    print("\nĐánh giá mô hình trên tập test...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()