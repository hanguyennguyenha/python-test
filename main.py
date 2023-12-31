import nltk
nltk.download('punkt')  # Add this line to download the 'punkt' resource

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Đầu vào là một danh sách các câu tiếng Ê Đê
corpus = ["Cây cầu này dài một cây số.", "Người Ê Đê thường sống ở các làng dọc theo sông.",
          "Tiếng Ê Đê có những âm thanh độc đáo và phức tạp cây cây cây."]

# Tokenization
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
print(tokenized_corpus)
# Tạo mô hình Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Lưu mô hình vào một tệp để sử dụng sau này
model.save("word2vec_edê.model")

# Load mô hình đã lưu
loaded_model = Word2Vec.load("word2vec_edê.model")

# Lấy embedding của từ "cây"
embedding_cay = loaded_model.wv['cây cầu']
print(embedding_cay)