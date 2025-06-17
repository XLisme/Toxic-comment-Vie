import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from toxic_comment_detection import preprocess_text
import re
from underthesea import word_tokenize

# Define common Vietnamese profanity words
# YOU SHOULD EXPAND THIS LIST WITH MORE SPECIFIC TERMS RELEVANT TO YOUR DATASET
PROFANITY_WORDS = [
    "đmm", "dm", "đm", "cc", "cặc", "lồn", "cl", "địt", "đéo", "mày", "tao",
    "óc chó", "ngu", "chết tiệt", "khốn nạn", "súc vật", "thằng chó",
    "đĩ", "phò", "động vật", "vô học", "rác rưởi", "bẩn thỉu", "mất dạy",
    # Add more words here
]

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

def is_profane(comment, profanity_words=PROFANITY_WORDS):
    """Checks if a comment contains any forbidden words using underthesea.word_tokenize."""
    # Simple preprocessing for profanity check: lowercase and remove some common punctuation
    temp_comment = comment.lower()
    temp_comment = re.sub(r'[^\w\s]', ' ', temp_comment) # Keep words and spaces
    temp_comment = re.sub(r'\s+', ' ', temp_comment).strip()

    # Tokenize the comment to get individual words
    # Using format="text" and then splitting to ensure compatibility with `in` operator
    words_in_comment = word_tokenize(temp_comment, format="text").split()

    for word in profanity_words:
        if word in words_in_comment:
            return True
    return False

def is_spam_comment(comment_for_spam_check):
    """
    Checks if a comment is likely spam based on character repetition and randomness.
    This is a heuristic and may require refinement.
    """
    words = comment_for_spam_check.split()
    if not words:
        return False

    spam_word_count = 0
    vowels = "aeiouáàảạăằẳặẵâầấẩẫậéèẻẽêềếểễệíìỉĩịóòỏõôồốổỗộơờớởỡợúùủũụưừứửữựýỳỷỹỵ"
    
    for word in words:
        # Rule 1: Excessive consecutive identical character repetition (e.g., "aaaaa")
        if re.search(r'(.)\\1{4,}', word): # 5 or more identical characters
            spam_word_count += 1
            continue

        # Rule 2: Long words with very low character diversity (suggests random key presses)
        if len(word) > 4: # Lower threshold for word length to be more sensitive
            unique_chars_ratio = len(set(word)) / len(word)
            
            # If unique characters are very few (increased sensitivity)
            if unique_chars_ratio < 0.4:
                spam_word_count += 1
                continue

        # Rule 3: Check for words with very few vowels (common in gibberish)
        if len(word) > 3:
            vowel_count = sum(1 for char in word.lower() if char in vowels)
            if vowel_count / len(word) < 0.15: # Very low vowel ratio for longer words
                spam_word_count += 1
                continue

    # If even one word is flagged as spam, mark the entire comment as spam
    if spam_word_count >= 1:
        return True
    
    return False

def predict_comment(model, tokenizer, comment, max_length=128):
    """Predict if a comment is toxic or not, with pre-checks for profanity and spam."""
    
    # Rule 1: Check for profanity in the original, lowercased, lightly processed comment
    if is_profane(comment):
        return {
            'comment': comment,
            'processed_comment': preprocess_text(comment), # Still preprocess for display
            'prediction': 'toxic',
            'confidence': 1.0 # 100% confidence
        }

    # Prepare comment for spam detection (only lowercasing and remove punctuation, NO repetition normalization)
    comment_for_spam_check = comment.lower()
    comment_for_spam_check = re.sub(r'[^\w\s]', ' ', comment_for_spam_check) # Keep words and spaces
    comment_for_spam_check = re.sub(r'\s+', ' ', comment_for_spam_check).strip()

    # Rule 2: Check for spam in the comment before repetition normalization
    if is_spam_comment(comment_for_spam_check):
        return {
            'comment': comment,
            'processed_comment': preprocess_text(comment), # Still preprocess for display
            'prediction': 'toxic',
            'confidence': 1.0 # 100% confidence
        }
    
    # If not caught by rules, proceed with model prediction
    # Preprocess comment for model input (this will include repetition normalization)
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