import torch 
import torch.nn as nn
import torch.optim as optim 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

x1 = [50, 100]
x2 = [5, 2]
y = [2.0, 4.0]

xx = np.column_stack([x1, x2])
yy = np.column_stack([y])

x_tensor = torch.tensor(xx, dtype = torch.float32)
y_tensor = torch.tensor(yy, dtype = torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(2, 5),
    nn.Sigmoid(),
    nn.Linear(5, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr= 0.1)

epochs = 100  # Tăng epochs để học tốt hơn
train_losses = []  # Lưu loss để vẽ đồ thị

for epoch in range(epochs):
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Dự đoán sau khi training
model.eval()
with torch.no_grad():
    y_pred_final = model(x_tensor).numpy()

# Vẽ đồ thị
plt.figure(figsize=(14, 5))

# 1. Đồ thị Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, color='blue', linewidth=2)
plt.title("Training Loss (MSE)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# 2. So sánh giá trị thực tế vs dự đoán
plt.subplot(1, 2, 2)
samples = ['Mẫu 1\n(50m², 5km)', 'Mẫu 2\n(100m², 2km)']
x_pos = np.arange(len(samples))

width = 0.35
plt.bar(x_pos - width/2, y, width, label='Giá thực tế', color='green', alpha=0.7, edgecolor='black')
plt.bar(x_pos + width/2, y_pred_final.flatten(), width, label='Giá dự đoán', color='orange', alpha=0.7, edgecolor='black')

# Thêm giá trị lên đầu cột
for i in range(len(samples)):
    plt.text(i - width/2, y[i] + 0.1, f'{y[i]:.2f}', ha='center', fontweight='bold')
    plt.text(i + width/2, y_pred_final[i,0] + 0.1, f'{y_pred_final[i,0]:.2f}', ha='center', fontweight='bold')

plt.xlabel('Mẫu dữ liệu')
plt.ylabel('Giá nhà (tỷ đồng)')
plt.title('So sánh: Giá thực tế vs Dự đoán', fontsize=14, fontweight='bold')
plt.xticks(x_pos, samples)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("KẾT QUẢ DỰ ĐOÁN")
print("="*50)
for i in range(len(x1)):
    print(f"Mẫu {i+1}: Diện tích={x1[i]}m², Khoảng cách={x2[i]}km")
    print(f"  Giá thực tế: {y[i]:.2f} tỷ")
    print(f"  Giá dự đoán: {y_pred_final[i,0]:.2f} tỷ")
    print(f"  Sai số: {abs(y[i] - y_pred_final[i,0]):.2f} tỷ")
    print()
