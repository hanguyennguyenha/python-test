import torch
import torch.nn as nn
import torch.optim as optim

# Dữ liệu mẫu (cần thay thế bằng dữ liệu thực)
english_sentences = ["Hello, how are you?", "What is your name?"]
vietnamese_sentences = ["Xin chào, bạn khỏe không?", "Bạn tên là gì?"]

# Tokenization và xây dựng từ điển
all_sentences = english_sentences + vietnamese_sentences
word_to_index = {word: idx for idx, word in enumerate(set(" ".join(all_sentences).lower().split()))}
vocab_size = len(word_to_index)

# Chuyển văn bản thành các mã định danh
english_indices = [[word_to_index[word] for word in sentence.lower().split()] for sentence in english_sentences]
vietnamese_indices = [[word_to_index[word] for word in sentence.lower().split()] for sentence in vietnamese_sentences]

# Chuyển văn bản thành tensor PyTorch
english_tensors = torch.LongTensor(english_indices)
vietnamese_tensors = torch.LongTensor(vietnamese_indices)

# Tạo lớp embedding
embedding_dim = 50
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Mô hình dịch máy đơn giản
class SimpleTranslationModel(nn.Module):
    def __init__(self, embedding_layer):
        super(SimpleTranslationModel, self).__init__()
        self.embedding_layer = embedding_layer
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_tensor):
        embedded = self.embedding_layer(input_tensor)
        output = self.linear(embedded)
        return output

# Khởi tạo mô hình
model = SimpleTranslationModel(embedding_layer)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 100
for epoch in range(num_epochs):
    # Tiền xử lý đầu vào
    input_data = english_tensors  # Đối với mô hình đơn giản, chúng ta chỉ sử dụng tiếng Anh làm đầu vào

    # Lan truyền và tối ưu hóa
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, input_data.view(-1))
    loss.backward()
    optimizer.step()

    # In ra loss
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Lấy vectơ embedding của một từ cụ thể
word_to_lookup = "hello"
word_index = word_to_index[word_to_lookup]
embedding_vector = embedding_layer(torch.LongTensor([word_index]))

print(f"Embedding vector for '{word_to_lookup}': {embedding_vector}")
