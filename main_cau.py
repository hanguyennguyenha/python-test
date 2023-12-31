import nltk
nltk.download('punkt')

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Đầu vào là một danh sách các câu tiếng Ê Đê
corpus = ["Cây cầu này dài một cây số.", "Người Ê Đê thường sống ở các làng dọc theo sông.",
          "Tiếng Ê Đê có những âm thanh độc đáo và phức tạp cây cây cây."]

# Tokenization
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Tạo mô hình Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Lưu mô hình vào một tệp để sử dụng sau này
model.save("word2vec_edê.model")

# Load mô hình đã lưu
loaded_model = Word2Vec.load("word2vec_edê.model")

# Lấy embedding của câu
def get_sentence_embedding(sentence, model):
    # Tokenization
    tokenized_sentence = word_tokenize(sentence.lower())
    
    # Lấy embedding của từng từ trong câu
    word_embeddings = [model.wv[word] for word in tokenized_sentence if word in model.wv]
    
    # Tính trung bình của các embedding để có embedding của câu
    sentence_embedding = sum(word_embeddings) / len(word_embeddings) if word_embeddings else None
    
    return sentence_embedding

# Lấy embedding của câu "Cây cầu này dài một cây số."
embedding_cau = get_sentence_embedding("Cây cầu này dài một cây số.", loaded_model)
print(embedding_cau)
