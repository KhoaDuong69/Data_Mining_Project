import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Đọc dữ liệu
# =========================
url = "https://raw.githubusercontent.com/KhoaDuong69/Data_Mining_Project/main/Data_Mining_Project/THKTDL/Data/food_coded.csv"

df_raw = pd.read_csv(url)

print("Shape ban đầu:", df_raw.shape)

selected_cols = [
    "breakfast",
    "eating_out",
    "fruit_day",
    "veggies_day",
    "calories_day",
    "cook",
    "healthy_feeling",
    "nutritional_check",
    "italian_food",
    "greek_food",
    "indian_food",
    "thai_food",
    "ethnic_food"
]

# ==============================================================================
# PHẦN A. KHẢO SÁT DỮ LIỆU GỐC
# ==============================================================================

# ===== A1. Tổng quan =====
print("\n" + "=" * 60); print("A1. TỔNG QUAN"); print("=" * 60)
print("Kích thước:", df_raw.shape)
print("Kiểu dữ liệu:\n", df_raw.dtypes.value_counts().to_string())
print("Dòng trùng lặp:", df_raw.duplicated().sum())
n_miss = int(df_raw.isnull().sum().sum())
print(f"Tổng ô thiếu: {n_miss} ({n_miss / df_raw.size:.1%})")

# ===== A2. Missing values toàn bộ cột =====
print("\n" + "=" * 60); print("A2. MISSING VALUES (toàn bộ cột)"); print("=" * 60)
miss_all = pd.DataFrame({
    "so_thieu": df_raw.isnull().sum(),
    "ty_le_%": (df_raw.isnull().mean() * 100).round(1)
})
print(miss_all[miss_all.so_thieu > 0].sort_values("so_thieu", ascending=False).to_string())

# ===== A3. Thống kê mô tả 13 cột được chọn =====
sub = df_raw[selected_cols].apply(pd.to_numeric, errors="coerce")

print("\n" + "=" * 60); print("A3. THỐNG KÊ MÔ TẢ 13 CỘT"); print("=" * 60)
print(sub.describe().T.round(2).to_string())

# ===== A4. Tóm tắt thang đo =====
summary = pd.DataFrame({
    "kieu_goc": df_raw[selected_cols].dtypes.astype(str),
    "so_thieu": sub.isnull().sum(),
    "ty_le_thieu_%": (sub.isnull().mean() * 100).round(1),
    "min": sub.min(), "max": sub.max(),
    "so_gia_tri_khac_nhau": sub.nunique(),
    "cac_gia_tri": sub.apply(lambda s: sorted(s.dropna().unique().tolist())),
})
print("\n" + "=" * 60); print("A4. TÓM TẮT THANG ĐO"); print("=" * 60)
print(summary.to_string())

# ===== A5. Phân bố giá trị + biểu đồ =====
print("\n" + "=" * 60); print("A5. PHÂN BỐ GIÁ TRỊ TỪNG CỘT"); print("=" * 60)
for c in selected_cols:
    print(f"{c}:", sub[c].value_counts(dropna=False).sort_index().to_dict())

fig, axes = plt.subplots(3, 5, figsize=(20, 11)); axes = axes.ravel()
for i, c in enumerate(selected_cols):
    vc = sub[c].value_counts(dropna=False).sort_index()
    axes[i].bar([str(k) for k in vc.index], vc.values)
    axes[i].set_title(c)
for j in range(len(selected_cols), len(axes)):
    axes[j].axis("off")
plt.suptitle("Phân bố giá trị 13 thuộc tính (dữ liệu gốc)", fontsize=14)
plt.tight_layout(); plt.show()

# ===== A6. Ma trận tương quan =====
corr = sub.corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
plt.colorbar(im); plt.title("Tương quan giữa 13 thuộc tính"); plt.tight_layout(); plt.show()

# ==============================================================================
# PHẦN B. TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================

# =========================
# 2. Chọn các thuộc tính cần thiết
# =========================
df = df_raw[selected_cols].copy()

# =========================
# 3. Kiểm tra Missing Values
# =========================
print("\n" + "=" * 60); print("B3. MISSING VALUES TRƯỚC KHI XỬ LÝ"); print("=" * 60)
print(df.isnull().sum())

# =========================
# 4. Xử lý Missing Values
# Dùng median cho cột số
# =========================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in df.columns:
    df[col] = df[col].fillna(df[col].median())   # gán lại (không dùng inplace=True, vì không có tác dụng)

print("\nMissing Values sau khi điền median:", int(df.isnull().sum().sum()))

# =========================
# 5. Kiểm tra dữ liệu trùng lặp
# =========================
print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

# =========================
# 6. Rời rạc hóa dữ liệu
# =========================

def categorize_1_5(x):
    if x <= 2:
        return "Low"
    elif x <= 4:
        return "Medium"
    else:
        return "High"

cols_to_bin = [
    "eating_out",
    "fruit_day",
    "veggies_day",
    "cook",
    "healthy_feeling",
    "nutritional_check"
]

for col in cols_to_bin:
    df[col] = df[col].apply(categorize_1_5)

# =========================
# 7. Rời rạc hóa calories_day
# =========================
df["calories_day"] = pd.cut(
    df["calories_day"],
    bins=3,
    labels=["Low", "Medium", "High"]
)

# =========================
# 8. Chuyển các cột sở thích thực phẩm
# =========================

food_cols = [
    "italian_food",
    "greek_food",
    "indian_food",
    "thai_food",
    "ethnic_food"
]

for col in food_cols:
    df[col] = df[col].apply(categorize_1_5)

# =========================
# 9. Chuyển breakfast
# =========================
df["breakfast"] = df["breakfast"].map({
    1: "Breakfast_No",
    2: "Breakfast_Yes"
})

# =========================
# 10. One-Hot Encoding
# =========================
transaction_df = pd.get_dummies(df)

transaction_df = transaction_df.astype(int)

print("\nShape sau encoding:")
print(transaction_df.shape)

print(transaction_df.head())

# =========================
# 11. Lưu dữ liệu
# =========================
transaction_df.to_csv(
    "food_association_ready.csv",
    index=False
)

print("\nĐã lưu file: food_association_ready.csv")
