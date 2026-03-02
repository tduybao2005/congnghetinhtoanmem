import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 

data = pd.read_csv("housing.csv")
data = data.drop('ocean_proximity', axis=1)

target_col = data.columns[-1]
feature_cols = [c for c in data.columns if c != target_col]

print("=" * 40)
print("mô tả dữ liệu")
print("=" * 40)
print(f"số lượng mẫu: {data.shape[0]}")
print(f"số đặc trưng đầu vào: {len(feature_cols)}")
print(f"số đầu ra: {len(target_col)}")
print(f"đặc trưng đầu vào: {feature_cols}")
print(f"Biến đầu ra: {target_col}")
print()
print("5 dòng đầu tiên")
print(data.head())
print()
print("thống kê mô tả")
print(data.describe())

# tiền xử lý dữ liệu
print("\n" + "=" * 40)
print("Kiểm tra dữ liệu thiếu")
print("=" * 60)
print(data.isnull().sum())
print(f"\nTổng số giá trị thiếu: {data.isnull().sum().sum()}")
data = data.dropna()
print(f"số mẫu sau khi loại bỏ dòng thiếu: {data.shape[0]}") 

X = data[[i for i in feature_cols]].values
y = data[target_col].values

n = len(X) #số lượng mẫu
train_size = int(n * 2/3)

x_train, x_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# chuẩn hóa z-score cho X (đặc trưng đầu vào) z = (x - micro) / sigma
micro_x = x_train.mean(axis = 0)
sigma_x = x_train.std(axis = 0)
x_train = (x_train - micro_x) / sigma_x
x_test = (x_test - micro_x) / sigma_x
# z = (y - micro) / sigma
micro_y = y_train.mean()
sigma_y = y_train.std()
y_train = (y_train - micro_y) / sigma_y
y_test = (y_test - micro_y) / sigma_y

x_train_tensor = torch.tensor(x_train, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.float32).view(-1, 1)
x_test_tensor = torch.tensor(x_test, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test, dtype = torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(8, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 1000 
train_losses = []
for epoch in range(epochs):
    y_pred = model(x_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.5f}")

model.eval()
with torch.no_grad():
    y_train_pred_tensor = model(x_train_tensor)
    y_test_pred_tensor = model(x_test_tensor)

    train_mse = criterion(y_train_pred_tensor, y_train_tensor).item()
    test_mse = criterion(y_test_pred_tensor, y_test_tensor).item()
    
    y_train_pred = y_train_pred_tensor.numpy().flatten()
    y_test_pred = y_test_pred_tensor.numpy().flatten()
print(f"\n[Neural Network]")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# ============================================
# Linear Regression (Hồi quy tuyến tính)
# ============================================
print("\n" + "=" * 40)
print("Linear Regression (Hồi quy tuyến tính)")
print("=" * 40)

# Train mô hình hồi quy tuyến tính
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Dự đoán
y_train_pred_lr = lr_model.predict(x_train)
y_test_pred_lr = lr_model.predict(x_test)

# Tính MSE
train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)

print(f"\n[Linear Regression]")
print(f"Train MSE: {train_mse_lr:.4f}")
print(f"Test MSE: {test_mse_lr:.4f}")

# So sánh kết quả
print("\n" + "=" * 40)
print("SO SÁNH KẾT QUẢ")
print("=" * 40)
print(f"Neural Network Test MSE: {test_mse:.4f}")
print(f"Linear Regression Test MSE: {test_mse_lr:.4f}")
print(f"Chênh lệch: {abs(test_mse - test_mse_lr):.4f}")
if test_mse < test_mse_lr:
    improvement = ((test_mse_lr - test_mse) / test_mse_lr) * 100
    print(f"→ Neural Network tốt hơn {improvement:.2f}%")
else:
    improvement = ((test_mse - test_mse_lr) / test_mse) * 100
    print(f"→ Linear Regression tốt hơn {improvement:.2f}%")

plt.figure(figsize = (20, 10))

# 1. Đồ thị Loss
plt.subplot(3, 3, 1)
plt.plot(train_losses)
plt.title("Training Loss (Neural Network)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# 2. Actual vs Predicted - Neural Network
plt.subplot(3, 3, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, s=10, label='Data points', color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'y--', lw=2, label='Ideal (y=x)')

z = np.polyfit(y_test, y_test_pred, 1)
p = np.poly1d(z)
x_line = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(x_line, p(x_line), 'r-', lw=2, label=f'Regression: y={z[0]:.2f}x+{z[1]:.2f}')

plt.title("Actual vs Predicted (Neural Network)")
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Actual vs Predicted - Linear Regression
plt.subplot(3, 3, 3)
plt.scatter(y_test, y_test_pred_lr, alpha=0.5, s=10, label='Data points', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=2, label='Ideal (y=x)')

z_lr = np.polyfit(y_test, y_test_pred_lr, 1)
p_lr = np.poly1d(z_lr)
x_line_lr = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(x_line_lr, p_lr(x_line_lr), 'r-', lw=2, label=f'Regression: y={z_lr[0]:.2f}x+{z_lr[1]:.2f}')

plt.title("Actual vs Predicted (Linear Regression)")
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. So sánh MSE - Bar Chart
plt.subplot(3, 3, 4)
models = ['Neural\nNetwork', 'Linear\nRegression']
mse_values = [test_mse, test_mse_lr]
colors_bar = ['#3498db', '#e74c3c']
bars = plt.bar(models, mse_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.ylabel('MSE')
plt.title('So sánh MSE giữa 2 mô hình')
plt.grid(True, alpha=0.3, axis='y')
# Thêm giá trị lên đầu cột
for bar, val in zip(bars, mse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 5. Histogram phần dư (Residuals) - Neural Network
plt.subplot(3, 3, 5)
residuals = y_test - y_test_pred
plt.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Phần dư (Residual = y - ŷ)')
plt.ylabel('Tần suất')
plt.title('Phân phối phần dư (Neural Network)')
plt.grid(True, alpha=0.3)

# 6. Histogram phần dư (Residuals) - Linear Regression
plt.subplot(3, 3, 6)
residuals_lr = y_test - y_test_pred_lr
plt.hist(residuals_lr, bins=50, color='orange', edgecolor='white', alpha=0.8)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Phần dư (Residual = y - ŷ)')
plt.ylabel('Tần suất')
plt.title('Phân phối phần dư (Linear Regression)')
plt.grid(True, alpha=0.3)

# 7. Biểu đồ trọng số (Feature Importance) - Neural Network
plt.subplot(3, 3, 7)
# Lấy trọng số từ layer đầu tiên
weights = model[0].weight.data.numpy()  # Shape: (100, 8)
feature_importance = np.abs(weights).mean(axis=0)  # Trung bình theo 100 neurons
sorted_idx = np.argsort(feature_importance)[::-1]

colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(feature_cols)))
plt.barh(range(len(feature_cols)), feature_importance[sorted_idx], color=colors)
plt.yticks(range(len(feature_cols)), [feature_cols[i] for i in sorted_idx])
plt.xlabel('Trọng số trung bình')
plt.title('Trọng số đặc trưng (Neural Network)')
plt.grid(True, alpha=0.3, axis='x')

# 8. Hệ số Linear Regression
plt.subplot(3, 3, 8)
lr_coef = lr_model.coef_
sorted_idx_lr = np.argsort(np.abs(lr_coef))[::-1]
colors_lr = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(feature_cols)))
plt.barh(range(len(feature_cols)), np.abs(lr_coef)[sorted_idx_lr], color=colors_lr)
plt.yticks(range(len(feature_cols)), [feature_cols[i] for i in sorted_idx_lr])
plt.xlabel('Hệ số (|Coefficient|)')
plt.title('Hệ số đặc trưng (Linear Regression)')
plt.grid(True, alpha=0.3, axis='x')

# 9. Scatter plot: NN predictions vs LR predictions
plt.subplot(3, 3, 9)
plt.scatter(y_test_pred, y_test_pred_lr, s=20, color='purple', alpha=0.5, edgecolors='black', linewidth=0.5)

# Đường y=x (khi 2 mô hình dự đoán giống hệt nhau)
min_val = min(y_test_pred.min(), y_test_pred_lr.min())
max_val = max(y_test_pred.max(), y_test_pred_lr.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x (Perfect Agreement)')

# Tính correlation
correlation = np.corrcoef(y_test_pred, y_test_pred_lr)[0, 1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('Neural Network Predictions')
plt.ylabel('Linear Regression Predictions')
plt.title('So sánh dự đoán: NN vs LR')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout(h_pad=5.0, w_pad = 4.0)
plt.show()
