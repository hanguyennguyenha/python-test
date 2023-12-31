import tensorflow as tf
from tensorflow.keras.layers import Embedding

# Chuẩn bị dữ liệu và từ điển
text_data = ["I love natural language processing.", "Embeddings are powerful."]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text_data)

# Số lượng từ trong từ điển
vocab_size = len(tokenizer.word_index) + 1

# Đặc tính của mỗi từ: ví dụ, 50 chiều
embedding_dim = 50

# Tạo lớp embedding
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

# Sử dụng mô hình embedding cho một đoạn văn bản
input_sequence = tokenizer.texts_to_sequences(["I like embeddings."])

# Convert the input sequence to a tensor
input_sequence = tf.constant(input_sequence)

# Apply the embedding layer to the input tensor
embedded_sequence = embedding_layer(input_sequence)

print(embedded_sequence)
