import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ==============================================================================
# 0. CẤU HÌNH TRANG WEB STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Data Mining - Phân Tích Thói Quen Ăn Uống Sinh Viên",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Từ điển ánh xạ tên cột nhị phân sang tiếng Việt dễ đọc
ITEM_NAME_MAP = {
    "breakfast_Breakfast_No": "Không ăn sáng",
    "breakfast_Breakfast_Yes": "Có ăn sáng",
    "eating_out_Low": "Ăn ngoài: Ít",
    "eating_out_Medium": "Ăn ngoài: Vừa",
    "eating_out_High": "Ăn ngoài: Nhiều",
    "fruit_day_Low": "Ăn trái cây: Ít",
    "fruit_day_Medium": "Ăn trái cây: Vừa",
    "fruit_day_High": "Ăn trái cây: Nhiều",
    "veggies_day_Low": "Ăn rau: Ít",
    "veggies_day_Medium": "Ăn rau: Vừa",
    "veggies_day_High": "Ăn rau: Nhiều",
    "calories_day_Low": "Ước lượng Calo: Thấp",
    "calories_day_Medium": "Ước lượng Calo: Vừa",
    "calories_day_High": "Ước lượng Calo: Cao",
    "cook_Low": "Tự nấu ăn: Ít",
    "cook_Medium": "Tự nấu ăn: Vừa",
    "cook_High": "Tự nấu ăn: Nhiều",
    "healthy_feeling_Low": "Cảm nhận sức khỏe: Thấp",
    "healthy_feeling_Medium": "Cảm nhận sức khỏe: Vừa",
    "healthy_feeling_High": "Cảm nhận sức khỏe: Cao",
    "nutritional_check_Low": "Kiểm tra dinh dưỡng: Ít",
    "nutritional_check_Medium": "Kiểm tra dinh dưỡng: Vừa",
    "nutritional_check_High": "Kiểm tra dinh dưỡng: Nhiều",
    "italian_food_Low": "Thích món Ý: Thấp",
    "italian_food_Medium": "Thích món Ý: Vừa",
    "italian_food_High": "Thích món Ý: Cao",
    "greek_food_Low": "Thích món Hy Lạp: Thấp",
    "greek_food_Medium": "Thích món Hy Lạp: Vừa",
    "greek_food_High": "Thích món Hy Lạp: Cao",
    "indian_food_Low": "Thích món Ấn: Thấp",
    "indian_food_Medium": "Thích món Ấn: Vừa",
    "indian_food_High": "Thích món Ấn: Cao",
    "thai_food_Low": "Thích món Thái: Thấp",
    "thai_food_Medium": "Thích món Thái: Vừa",
    "thai_food_High": "Thích món Thái: Cao",
    "ethnic_food_Low": "Thích món dân tộc: Thấp",
    "ethnic_food_Medium": "Thích món dân tộc: Vừa",
    "ethnic_food_High": "Thích món dân tộc: Cao",
}

def format_items(itemset, human_readable=True):
    items = sorted(list(itemset))
    if human_readable:
        return " + ".join([ITEM_NAME_MAP.get(item, item) for item in items])
    return " + ".join(items)

def find_file(filenames):
    """Tìm đường dẫn file tồn tại trong workspace"""
    for fname in filenames:
        if os.path.exists(fname):
            return fname
    return None

# ==============================================================================
# 1. HÀM CỦA THÀNH VIÊN 1: TIỀN XỬ LÝ DỮ LIỆU (TienXuLyDL.py)
# ==============================================================================
def categorize_1_5(x):
    if x <= 2:
        return "Low"
    elif x <= 4:
        return "Medium"
    else:
        return "High"

@st.cache_data
def load_raw_data():
    candidates = [
        "THKTDL/Data/food_coded.csv",
        "Data/food_coded.csv",
        "food_coded.csv",
        "../THKTDL/Data/food_coded.csv"
    ]
    fpath = find_file(candidates)
    if fpath:
        return pd.read_csv(fpath)
    return None

@st.cache_data
def run_tv1_preprocessing(df_raw):
    """Tái hiện toàn bộ luồng tiền xử lý của Thành viên 1"""
    df = df_raw.copy()
    
    # 1. Loại bỏ trùng lặp trên dữ liệu gốc (trước khi lọc 13 cột)
    df = df.drop_duplicates()
    
    # 2. Chọn 13 thuộc tính quan trọng
    selected_cols = [
        "breakfast", "eating_out", "fruit_day", "veggies_day", "calories_day",
        "cook", "healthy_feeling", "nutritional_check", "italian_food",
        "greek_food", "indian_food", "thai_food", "ethnic_food"
    ]
    cols_exist = [c for c in selected_cols if c in df.columns]
    df = df[cols_exist]
    
    # 3. Xử lý Missing Values bằng Median
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # 4. Rời rạc hóa các cột thang điểm 1-5
    cols_to_bin = [
        "eating_out", "fruit_day", "veggies_day", "cook", "healthy_feeling",
        "nutritional_check", "italian_food", "greek_food", "indian_food",
        "thai_food", "ethnic_food"
    ]
    for col in cols_to_bin:
        if col in df.columns:
            df[col] = df[col].apply(categorize_1_5)
            
    # 5. Rời rạc hóa calories_day
    if "calories_day" in df.columns:
        df["calories_day"] = pd.cut(df["calories_day"], bins=3, labels=["Low", "Medium", "High"])
        
    # 6. Chuyển đổi breakfast
    if "breakfast" in df.columns:
        df["breakfast"] = df["breakfast"].map({1: "Breakfast_No", 2: "Breakfast_Yes"})
        
    # 7. One-hot encoding
    encoded_df = pd.get_dummies(df).astype(int)
    return df, encoded_df

@st.cache_data
def load_prepared_data():
    """Tải file food_association_ready.csv sẵn có"""
    candidates = [
        "THKTDL/Data/food_association_ready.csv",
        "THKTDL/Source_Code/food_association_ready.csv",
        "food_association_ready.csv",
        "Data/food_association_ready.csv"
    ]
    fpath = find_file(candidates)
    if fpath:
        return pd.read_csv(fpath)
    return None

# ==============================================================================
# 2. HÀM CỦA THÀNH VIÊN 2: THUẬT TOÁN KHAI THÁC TẬP PHỔ BIẾN
# ==============================================================================
def it_tree_mining(df, minsup=0.1):
    """Thuật toán IT-Tree (Itemset Tree mining) từ Thành viên 2"""
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["support", "itemsets"])

    items = [(col, set(df.index[df[col]])) for col in df.columns]
    results = []

    def search(prefix, current_items):
        for i, (name, tids) in enumerate(current_items):
            sup = len(tids) / n
            if sup < minsup:
                continue

            new_set = prefix | {name}
            results.append((sup, frozenset(new_set)))

            next_items = [
                (name2, tids & tids2)
                for name2, tids2 in current_items[i + 1:]
            ]
            search(new_set, next_items)

    search(set(), items)
    return pd.DataFrame(results, columns=["support", "itemsets"])

@st.cache_data
def mine_frequent_itemsets(df_bool, algorithm_name, min_support):
    """Khai thác tập phổ biến theo thuật toán được chọn"""
    start_time = time.perf_counter()
    
    if algorithm_name == "Apriori":
        itemsets = apriori(df_bool, min_support=min_support, use_colnames=True)
    elif algorithm_name == "FP-Growth":
        itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
    elif algorithm_name == "IT-Tree":
        itemsets = it_tree_mining(df_bool, minsup=min_support)
    else:
        itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
        
    exec_time = time.perf_counter() - start_time
    return itemsets, exec_time

# ==============================================================================
# 3. HÀM CỦA THÀNH VIÊN 3: SINH LUẬT KẾT HỢP & GIẢI THÍCH TỰ ĐỘNG
# ==============================================================================
@st.cache_data
def generate_rules_from_itemsets(itemsets_df, min_confidence, min_lift=1.0):
    if itemsets_df is None or itemsets_df.empty:
        return pd.DataFrame()
    try:
        rules = association_rules(itemsets_df, metric="confidence", min_threshold=min_confidence)
        if rules.empty:
            return pd.DataFrame()
        # Lọc lift
        filtered_rules = rules[rules["lift"] >= min_lift].copy()
        return filtered_rules
    except Exception:
        return pd.DataFrame()

def generate_natural_explanation(antecedents, consequents, support, confidence, lift):
    """Tạo câu giải thích tự động theo ngôn ngữ tự nhiên"""
    ante_str = format_items(antecedents, human_readable=True)
    cons_str = format_items(consequents, human_readable=True)
    conf_pct = round(confidence * 100, 1)
    sup_pct = round(support * 100, 1)
    lift_val = round(lift, 2)
    
    explanation = (
        f"Sinh viên có thói quen **[{ante_str}]** thì có **{conf_pct}%** khả năng cũng có **[{cons_str}]**. "
        f"Mối liên hệ này chiếm **{sup_pct}%** toàn bộ sinh viên khảo sát và có độ nâng **Lift = {lift_val}** "
        f"(cao gấp **{lift_val} lần** so với việc xuất hiện ngẫu nhiên)."
    )
    return explanation

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (STREAMLIT APP)
# ==============================================================================
# Tải dữ liệu sẵn
raw_df = load_raw_data()
encoded_df = load_prepared_data()

if encoded_df is None and raw_df is not None:
    _, encoded_df = run_tv1_preprocessing(raw_df)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/salad.png", width=80)
st.sidebar.title("Khai Phá Dữ Liệu")
st.sidebar.caption("Đồ án Khai thác Tập phổ biến & Luật kết hợp")

# Menu điều hướng
menu = st.sidebar.radio(
    "📌 Điều hướng trang:",
    [
        "1. Dữ liệu & Tiền xử lý (TV1)",
        "2. Khai thác Luật kết hợp (TV2 & TV3)",
        "3. So sánh 3 Thuật toán (Benchmark)"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Cấu hình khai phá")

selected_algo = st.sidebar.selectbox(
    "Thuật toán:",
    ["FP-Growth", "Apriori", "IT-Tree"],
    help="FP-Growth tối ưu cây, Apriori duyệt BFS, IT-Tree duyệt giao TID"
)

minsup_val = st.sidebar.slider(
    "Ngưỡng Min Support (Tỷ lệ ủng hộ):",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01,
    help="Tỷ lệ sinh viên tối thiểu chứa tập item"
)

minconf_val = st.sidebar.slider(
    "Ngưỡng Min Confidence (Độ tin cậy):",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05,
    help="Xác suất có B khi đã có A: P(B|A)"
)

minlift_val = st.sidebar.slider(
    "Ngưỡng Min Lift (Độ nâng):",
    min_value=1.0,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Lift > 1 biểu thị mối tương quan đồng biến (tích cực)"
)

# ==============================================================================
# TRANG 1: DỮ LIỆU & TIỀN XỬ LÝ (THÀNH VIÊN 1)
# ==============================================================================
if menu == "1. Dữ liệu & Tiền xử lý (TV1)":
    st.header("🥗 Quy Trình Tiền Xử Lý Dữ Liệu Khảo Sát Dinh Dưỡng")
    st.markdown("""
    Trang này thể hiện toàn bộ công việc tiền xử lý dữ liệu của **Thành viên 1** (`TienXuLyDL.py`), 
    chuyển đổi từ dữ liệu khảo sát gốc của sinh viên thành bảng giao dịch nhị phân (One-hot encoding) 
    sẵn sàng cho các thuật toán khai phá.
    """)
    
    col1, col2, col3 = st.columns(3)
    if raw_df is not None:
        col1.metric("Số dòng khảo sát gốc", len(raw_df))
        col2.metric("Số cột khảo sát gốc", len(raw_df.columns))
    if encoded_df is not None:
        col3.metric("Số thuộc tính sau mã hóa (Items)", len(encoded_df.columns))

    st.divider()
    tab_raw, tab_steps, tab_encoded = st.tabs([
        "📄 Dữ liệu gốc (food_coded.csv)",
        "🛠️ Các bước xử lý của TV1",
        "✅ Dữ liệu sẵn sàng khai phá (food_association_ready.csv)"
    ])

    with tab_raw:
        if raw_df is not None:
            st.subheader("Dữ liệu gốc chưa qua xử lý")
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.caption(f"Tổng cộng {raw_df.shape[0]} dòng và {raw_df.shape[1]} cột thuộc tính.")
        else:
            st.warning("Chưa tìm thấy file `food_coded.csv` trong thư mục `THKTDL/Data/`.")

    with tab_steps:
        st.subheader("Quy trình 5 bước tiền xử lý dữ liệu")
        st.markdown("""
        1. **Loại bỏ trùng lặp (Deduplication)**: Loại bỏ các dòng ghi nhận trùng trên dữ liệu gốc.
        2. **Lựa chọn thuộc tính mục tiêu (13 cột)**:
           - *Ăn uống:* `breakfast`, `eating_out`, `cook`, `calories_day`.
           - *Dinh dưỡng & Thể chất:* `fruit_day`, `veggies_day`, `healthy_feeling`, `nutritional_check`.
           - *Sở thích ẩm thực:* `italian_food`, `greek_food`, `indian_food`, `thai_food`, `ethnic_food`.
        3. **Xử lý giá trị khuyết (Missing Values)**: Chuyển sang số và điền bằng giá trị trung vị (`median`).
        4. **Rời rạc hóa (Discretization)**:
           - Thang điểm 1-5 được nhóm thành 3 mức: `Low` (<= 2), `Medium` (3-4), `High` (5).
           - `calories_day` được chia 3 bins: `Low`, `Medium`, `High`.
           - `breakfast`: Mã hóa thành `Breakfast_No` (1) và `Breakfast_Yes` (2).
        5. **One-Hot Encoding**: Tạo các cột nhị phân (0/1) đại diện cho từng itemset.
        """)

    with tab_encoded:
        if encoded_df is not None:
            st.subheader("Bảng giao dịch nhị phân hoàn chỉnh (37 Items)")
            st.dataframe(encoded_df.head(10), use_container_width=True)
            
            # Biểu đồ tần suất xuất hiện của các item
            st.subheader("📊 Tỷ lệ xuất hiện của các thói quen (Item Support)")
            item_freq = (encoded_df.mean() * 100).sort_values(ascending=False).reset_index()
            item_freq.columns = ["Mã Item", "Tỷ lệ xuất hiện (%)"]
            item_freq["Tên Thói Quen (Tiếng Việt)"] = item_freq["Mã Item"].map(ITEM_NAME_MAP).fillna(item_freq["Mã Item"])

            fig_bar = px.bar(
                item_freq.head(15),
                x="Tỷ lệ xuất hiện (%)",
                y="Tên Thói Quen (Tiếng Việt)",
                orientation="h",
                text="Tỷ lệ xuất hiện (%)",
                color="Tỷ lệ xuất hiện (%)",
                color_continuous_scale="Viridis",
                title="Top 15 Thói Quen & Sở Thích Phổ Biến Nhất Trong Sinh Viên"
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.error("Chưa có dữ liệu `food_association_ready.csv`.")

# ==============================================================================
# TRANG 2: KHAI THÁC LUẬT KẾT HỢP (THÀNH VIÊN 2 & 3)
# ==============================================================================
elif menu == "2. Khai thác Luật kết hợp (TV2 & TV3)":
    st.header("⚡ Khai Thác Tập Phổ Biến & Luật Kết Hợp")

    if encoded_df is None:
        st.error("Không tìm thấy dữ liệu để khai phá. Vui lòng kiểm tra lại file `food_association_ready.csv`!")
        st.stop()

    df_bool = encoded_df.astype(bool)

    # Khai thác tập phổ biến
    with st.spinner(f"Đang chạy thuật toán {selected_algo} với MinSup = {minsup_val}..."):
        frequent_itemsets, run_time = mine_frequent_itemsets(df_bool, selected_algo, minsup_val)

    # Hiển thị số đo KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Thuật toán", selected_algo)
    c2.metric("Thời gian chạy", f"{run_time:.4f} s")
    c3.metric("Số tập phổ biến", len(frequent_itemsets))

    if frequent_itemsets.empty:
        c4.metric("Số luật kết hợp", 0)
        st.warning(f"⚠️ Không tìm thấy tập phổ biến nào với ngưỡng **Min Support = {minsup_val}**! Hãy thử kéo thanh trượt Min Support xuống thấp hơn (ví dụ: 0.05 hoặc 0.1).")
        st.stop()

    # Sinh luật kết hợp
    rules_df = generate_rules_from_itemsets(frequent_itemsets, minconf_val, minlift_val)
    c4.metric("Số luật sinh ra", len(rules_df))

    st.divider()

    # BỘ LỌC THÓI QUEN Ở VẾ TRÁI (ANTECEDENTS) - THEO YÊU CẦU NGƯỜI DÙNG
    st.subheader("🔍 Lọc Luật Theo Thói Quen Vế Trái (Antecedents)")
    
    # Lấy danh sách tất cả các item có mặt
    all_raw_items = sorted(list(encoded_df.columns))
    item_display_options = {item: f"{ITEM_NAME_MAP.get(item, item)} ({item})" for item in all_raw_items}

    # Định nghĩa tuple tiền tố sở thích ẩm thực & tách danh sách thói quen
    PREF_PREFIXES = ("italian_food", "greek_food", "indian_food", "thai_food", "ethnic_food")
    habit_only_items = [col for col in all_raw_items if not col.startswith(PREF_PREFIXES)]

    strict_direction = st.checkbox(
        "🎯 Chỉ lọc các luật có Vế trái là [Thói quen] -> Vế phải là [Sở thích ẩm thực] (Đúng trọng tâm đề bài)",
        value=False
    )

    multiselect_options = habit_only_items if strict_direction else all_raw_items

    selected_raw_antecedents = st.multiselect(
        "Chọn một hoặc nhiều thói quen ở vế trái mà bạn muốn khảo sát:",
        options=multiselect_options,
        format_func=lambda x: item_display_options[x],
        help="Hệ thống sẽ lọc ra các luật có chứa thói quen bạn đã chọn ở vế trái (Antecedents)."
    )

    col_filter_mode, col_sort_by = st.columns([2, 1])
    with col_filter_mode:
        filter_mode = st.radio(
            "Chế độ lọc thói quen:",
            ["Chứa ít nhất một trong các thói quen đã chọn (OR)", "Chứa tất cả các thói quen đã chọn (AND)"],
            horizontal=True
        )
    with col_sort_by:
        sort_metric = st.selectbox("Sắp xếp bảng theo:", ["lift", "confidence", "support"], index=0)

    # Lọc luật theo các điều kiện
    filtered_rules = rules_df.copy()

    # 1. Nếu bật chế độ nghiêm ngặt: Vế trái toàn bộ là Thói quen, Vế phải có ít nhất 1 Sở thích ẩm thực
    if not filtered_rules.empty and strict_direction:
        filtered_rules = filtered_rules[
            filtered_rules.apply(
                lambda row: all(not item.startswith(PREF_PREFIXES) for item in row["antecedents"]) and
                            any(item.startswith(PREF_PREFIXES) for item in row["consequents"]),
                axis=1
            )
        ]

    # 2. Lọc theo thói quen được chọn ở multiselect (OR / AND)
    if not filtered_rules.empty and selected_raw_antecedents:
        selected_set = set(selected_raw_antecedents)
        if "ít nhất một" in filter_mode:
            filtered_rules = filtered_rules[filtered_rules["antecedents"].apply(lambda a: bool(set(a) & selected_set))]
        else:
            filtered_rules = filtered_rules[filtered_rules["antecedents"].apply(lambda a: selected_set.issubset(set(a)))]

    # 3. Sắp xếp
    if not filtered_rules.empty:
        filtered_rules = filtered_rules.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)

    # XỬ LÝ TÌNH HUỐNG KHÔNG CÓ LUẬT THỎA NGƯỠNG
    if rules_df.empty:
        st.warning(
            f"⚠️ **Không có luật kết hợp nào** thỏa mãn ngưỡng "
            f"**Min Support = {minsup_val}**, **Min Confidence = {minconf_val}** và **Min Lift = {minlift_val}**.\n\n"
            f"👉 **Gợi ý:** Hãy thử kéo giảm *Min Confidence* xuống (ví dụ 0.3 - 0.4) hoặc giảm *Min Support* trên thanh bên trái."
        )
    elif filtered_rules.empty:
        st.info(
            f"ℹ️ **Không tìm thấy luật nào** có vế trái phù hợp với các thói quen bạn vừa chọn. "
            f"Hãy thử chuyển sang chế độ *'Chứa ít nhất một'* hoặc chọn các thói quen khác phổ biến hơn."
        )
    else:
        st.success(f"🎉 Đã tìm thấy **{len(filtered_rules)}** luật kết hợp phù hợp (Được sắp xếp theo `{sort_metric}` giảm dần)!")

        # Chuẩn bị bảng hiển thị đẹp
        display_table = pd.DataFrame()
        display_table["Vế Trái (Antecedents)"] = filtered_rules["antecedents"].apply(lambda x: format_items(x, human_readable=True))
        display_table["Vế Phải (Consequents)"] = filtered_rules["consequents"].apply(lambda x: format_items(x, human_readable=True))
        display_table["Support (%)"] = (filtered_rules["support"] * 100).round(2)
        display_table["Confidence (%)"] = (filtered_rules["confidence"] * 100).round(2)
        display_table["Lift"] = filtered_rules["lift"].round(2)
        display_table["Conviction"] = filtered_rules["conviction"].replace([np.inf, -np.inf], 999.99).round(2)

        st.dataframe(
            display_table.head(50),
            use_container_width=True,
            height=350
        )
        if len(display_table) > 50:
            st.caption("*(Chỉ hiển thị top 50 luật cao nhất để tối ưu tốc độ trình diễn)*")

        # ======================================================================
        # CÂU GIẢI THÍCH TỰ ĐỘNG (THEO YÊU CẦU NGƯỜI DÙNG)
        # ======================================================================
        st.subheader("💡 Diễn Giải Tri Thức Tự Động (Top Luật Tiêu Biểu)")
        top_k = min(5, len(filtered_rules))
        for idx in range(top_k):
            row = filtered_rules.iloc[idx]
            explanation_text = generate_natural_explanation(
                row["antecedents"],
                row["consequents"],
                row["support"],
                row["confidence"],
                row["lift"]
            )
            with st.chat_message("assistant", avatar="💡"):
                st.markdown(f"**Luật #{idx + 1}:** {explanation_text}")

        # Biểu đồ Scatter Plot phân tán các luật
        st.subheader("📊 Trực Quan Hóa Mối Quan Hệ (Scatter Plot: Support vs Confidence)")
        scatter_df = filtered_rules.copy()
        scatter_df["Antecedents_Text"] = scatter_df["antecedents"].apply(lambda x: format_items(x, human_readable=True))
        scatter_df["Consequents_Text"] = scatter_df["consequents"].apply(lambda x: format_items(x, human_readable=True))
        scatter_df["Support (%)"] = (scatter_df["support"] * 100).round(2)
        scatter_df["Confidence (%)"] = (scatter_df["confidence"] * 100).round(2)
        scatter_df["Lift"] = scatter_df["lift"].round(2)

        fig_scatter = px.scatter(
            scatter_df,
            x="Support (%)",
            y="Confidence (%)",
            color="Lift",
            size="Lift",
            hover_data=["Antecedents_Text", "Consequents_Text", "Lift"],
            color_continuous_scale="Turbo",
            title="Biểu đồ phân tán Luật kết hợp (Màu sắc và kích thước thể hiện giá trị Lift)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# TRANG 3: SO SÁNH HIỆU NĂNG 3 THUẬT TOÁN (BENCHMARK)
# ==============================================================================
elif menu == "3. So sánh 3 Thuật toán (Benchmark)":
    st.header("📊 So Sánh Thực Nghiệm 3 Thuật Toán (Apriori vs FP-Growth vs IT-Tree)")
    st.markdown("""
    Trang này tổng hợp kết quả thực nghiệm của **Thành viên 2** đo đạc trên nhiều ngưỡng `min_support` khác nhau.
    """)

    # Tải file thực nghiệm sẵn có hoặc tạo mẫu
    benchmark_candidates = [
        "THKTDL/Data/minsup_experiment_3_algorithms.csv",
        "Data/minsup_experiment_3_algorithms.csv",
        "THKTDL/Source_Code/minsup_experiment_3_algorithms.csv",
        "minsup_experiment_3_algorithms.csv"
    ]
    bm_file = find_file(benchmark_candidates)
    
    if bm_file:
        bm_df = pd.read_csv(bm_file)
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("⏱️ Thời gian thực thi (Execution Time)")
            fig_time = px.line(
                bm_df,
                x="MinSup",
                y=["Apriori_Time", "FP_Growth_Time", "IT_Tree_Time"],
                markers=True,
                labels={"value": "Thời gian (giây)", "variable": "Thuật toán", "MinSup": "Ngưỡng Min Support"},
                title="So sánh Thời gian chạy của 3 Thuật toán theo MinSup"
            )
            st.plotly_chart(fig_time, use_container_width=True)

        with c_right:
            st.subheader("📦 Số lượng tập phổ biến sinh ra")
            fig_items = px.line(
                bm_df,
                x="MinSup",
                y=["Apriori_Itemsets", "FP_Growth_Itemsets", "IT_Tree_Itemsets"],
                markers=True,
                labels={"value": "Số lượng tập phổ biến", "variable": "Thuật toán", "MinSup": "Ngưỡng Min Support"},
                title="Số tập phổ biến theo từng ngưỡng MinSup (Đồng nhất)"
            )
            st.plotly_chart(fig_items, use_container_width=True)

        st.subheader("📋 Bảng số liệu chi tiết thực nghiệm")
        st.dataframe(bm_df, use_container_width=True)

        st.info("""
        📌 **Nhận xét chuyên môn:**
        - **Tính đúng đắn**: Cả 3 thuật toán đều cho ra **chính xác cùng số lượng tập phổ biến** ở mọi ngưỡng `min_support`, khẳng định tính chính xác của thuật toán tự cài đặt (`IT-Tree`) so với thư viện chuẩn.
        - **Về hiệu năng thời gian**:
          - Khi `min_support` lớn (>= 0.2): Số tập ứng viên ít, thời gian chạy của cả 3 đều rất nhanh (< 0.01 giây).
          - Khi `min_support` giảm sâu (0.05 - 0.1): Số lượng tập phổ biến tăng vọt (lên hơn 11,000 tập). Lúc này thuật toán **FP-Growth** và **IT-Tree** thể hiện ưu thế vượt trội nhờ không phải sinh tổ hợp ứng viên khổng lồ như Apriori.
        """)
    else:
        st.warning("Chưa tìm thấy file số liệu thực nghiệm `minsup_experiment_3_algorithms.csv`.")
