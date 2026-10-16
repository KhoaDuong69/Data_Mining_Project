import pandas as pd
import numpy as np

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv("food_coded.csv")

print("Shape ban đầu:", df.shape)

# =========================
# 2. Chọn các thuộc tính cần thiết
# =========================
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

df = df[selected_cols]

# =========================
# 3. Kiểm tra Missing Values
# =========================
print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# 4. Xử lý Missing Values
# Dùng median cho cột số
# =========================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in df.columns:
    df[col].fillna(df[col].median(), inplace=True)

# =========================
# 5. Kiểm tra dữ liệu trùng lặp
# =========================
print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

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