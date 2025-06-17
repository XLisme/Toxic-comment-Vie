import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from toxic_comment_detection import preprocess_text

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

def load_model(model_path='best_model.pth'):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ToxicCommentClassifier(n_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_comment(model, tokenizer, comment, max_length=128):
    """Predict if a comment is toxic or not"""
    # Tiền xử lý comment
    processed_comment = preprocess_text(comment)
    
    # Tokenize
    encoding = tokenizer(
        processed_comment,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Chuyển dữ liệu vào device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][prediction].item()
    
    # Chuyển đổi kết quả
    label_map = {0: 'non_toxic', 1: 'toxic'}
    result = {
        'comment': comment,
        'processed_comment': processed_comment,
        'prediction': label_map[prediction.item()],
        'confidence': confidence
    }
    
    return result

def predict_batch(model, tokenizer, comments):
    """Predict for a batch of comments"""
    results = []
    for comment in comments:
        result = predict_comment(model, tokenizer, comment)
        results.append(result)
    return results

def main():
    # Load model và tokenizer
    print("Đang tải model...")
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    while True:
        print("\nNhập comment cần kiểm tra (hoặc 'q' để thoát):")
        comment = input().strip()
        
        if comment.lower() == 'q':
            break
            
        if not comment:
            print("Comment không được để trống!")
            continue
            
        # Dự đoán
        result = predict_comment(model, tokenizer, comment)
        
        # In kết quả
        print("\nKết quả phân tích:")
        print(f"Comment gốc: {result['comment']}")
        print(f"Comment sau xử lý: {result['processed_comment']}")
        print(f"Kết quả: {result['prediction']}")
        print(f"Độ tin cậy: {result['confidence']:.2%}")

if __name__ == "__main__":
    main() 