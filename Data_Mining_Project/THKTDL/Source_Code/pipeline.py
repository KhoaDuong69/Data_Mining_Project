"""
PIPELINE: KHAI THÁC TẬP PHỔ BIẾN & LUẬT KẾT HỢP - THÓI QUEN ĂN UỐNG SINH VIÊN
=============================================================================
Chạy toàn bộ luồng chỉ với 1 lệnh:

    python pipeline.py                      # chạy tất cả các bước
    python pipeline.py --steps eda,preprocess
    python pipeline.py --minsup 0.05 --minconf 0.6

Các bước (theo thứ tự):
    1. eda         Khảo sát dữ liệu gốc            (TV1 - TienXuLyDL.ipynb)
    2. preprocess  Tiền xử lý -> bảng nhị phân     (TV1 - TienXuLyDL.ipynb)
    3. mine        Apriori / FP-Growth / IT-Tree   (TV2 - Khai_thac_tap_pho_bien.ipynb)
    4. benchmark   Thực nghiệm MinSup vs thời gian (TV2)
    5. rules       Sinh luật + độ nhạy confidence  (TV3)
    6. interpret   Top mẫu hành vi tiêu biểu       (TV3)

Mặc định mọi thứ nằm CẠNH pipeline.py (Data/, app.py) nên copy cả thư mục sang máy khác
là chạy được, không cần sửa đường dẫn. Kết quả ghi vào --data-dir (mặc định: Data/).
Chạy xong pipeline TỰ ĐỘNG mở giao diện Streamlit (app.py). Thêm --no-app để tắt.
"""

import argparse
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # không cần màn hình, chạy được cả trên Colab / server
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
RAW_URL = (
    "https://raw.githubusercontent.com/KhoaDuong69/Data_Mining_Project/main/"
    "Data_Mining_Project/THKTDL/Data/food_coded.csv"
)

SELECTED_COLS = [
    "breakfast", "eating_out", "fruit_day", "veggies_day", "calories_day",
    "cook", "healthy_feeling", "nutritional_check", "italian_food",
    "greek_food", "indian_food", "thai_food", "ethnic_food",
]
COLS_TO_BIN = [
    "eating_out", "fruit_day", "veggies_day", "cook", "healthy_feeling",
    "nutritional_check", "italian_food", "greek_food", "indian_food",
    "thai_food", "ethnic_food",
]

ALL_STEPS = ["eda", "preprocess", "mine", "benchmark", "rules", "interpret"]

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # chạy bằng exec / một số môi trường notebook
    BASE_DIR = os.getcwd()
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "Data")   # luôn nằm cạnh pipeline.py


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==============================================================================
# BƯỚC 0: NẠP DỮ LIỆU GỐC
# ==============================================================================
def load_raw(data_dir):
    """Ưu tiên file local trong data_dir, nếu chưa có thì tải từ GitHub và lưu lại."""
    path = os.path.join(data_dir, "food_coded.csv")
    if os.path.exists(path):
        print(f"[load] Đọc file local: {path}")
        return pd.read_csv(path)
    print(f"[load] Chưa có {path}, tải từ GitHub...")
    df = pd.read_csv(RAW_URL)
    df.to_csv(path, index=False)
    print(f"[load] Đã lưu bản sao: {path}")
    return df


# ==============================================================================
# BƯỚC 1: KHẢO SÁT DỮ LIỆU GỐC
# ==============================================================================
def step_eda(df_raw, data_dir):
    banner("1. KHẢO SÁT DỮ LIỆU GỐC")
    print("Kích thước:", df_raw.shape)
    print("Dòng trùng lặp:", df_raw.duplicated().sum())
    n_miss = int(df_raw.isnull().sum().sum())
    print(f"Tổng ô thiếu: {n_miss} ({n_miss / df_raw.size:.1%})")

    sub = df_raw[SELECTED_COLS].apply(pd.to_numeric, errors="coerce")

    summary = pd.DataFrame({
        "so_thieu": sub.isnull().sum(),
        "ty_le_thieu_%": (sub.isnull().mean() * 100).round(1),
        "min": sub.min(),
        "max": sub.max(),
        "so_gia_tri_khac_nhau": sub.nunique(),
    })
    print("\nTóm tắt 13 thuộc tính được chọn:")
    print(summary.to_string())

    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    axes = axes.ravel()
    for i, c in enumerate(SELECTED_COLS):
        vc = sub[c].value_counts(dropna=False).sort_index()
        axes[i].bar([str(k) for k in vc.index], vc.values)
        axes[i].set_title(c)
    for j in range(len(SELECTED_COLS), len(axes)):
        axes[j].axis("off")
    plt.suptitle("Phân bố giá trị 13 thuộc tính (dữ liệu gốc)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_distribution.png"), dpi=120)
    plt.close()

    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    plt.colorbar(im)
    plt.title("Tương quan giữa 13 thuộc tính")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_correlation.png"), dpi=120)
    plt.close()
    print(f"\n[eda] Đã lưu biểu đồ vào {fig_dir}/")


# ==============================================================================
# BƯỚC 2: TIỀN XỬ LÝ  (giữ nguyên logic của Thành viên 1)
# ==============================================================================
def categorize_1_5(x):
    if x <= 2:
        return "Low"
    elif x <= 4:
        return "Medium"
    return "High"


def step_preprocess(df_raw, data_dir):
    banner("2. TIỀN XỬ LÝ DỮ LIỆU")
    df = df_raw[SELECTED_COLS].copy()
    print("Shape sau khi chọn 13 cột:", df.shape)

    # Missing values -> median
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print("Missing trước khi điền:\n", df.isnull().sum()[df.isnull().sum() > 0].to_string())
    for col in df.columns:
        # Dùng phép gán (không dùng inplace) để chạy đúng cả pandas 2.x lẫn 3.x
        df[col] = df[col].fillna(df[col].median())

    # Trùng lặp
    n_dup = df.duplicated().sum()
    print("Duplicate rows:", n_dup)
    df = df.drop_duplicates()

    # Rời rạc hóa
    # TODO (nhóm cân nhắc): healthy_feeling là thang 1-10, không phải 1-5;
    # italian_food chỉ có giá trị 3-5. Hiện giữ nguyên logic gốc của TV1.
    for col in COLS_TO_BIN:
        df[col] = df[col].apply(categorize_1_5)
    df["calories_day"] = pd.cut(df["calories_day"], bins=3, labels=["Low", "Medium", "High"])
    df["breakfast"] = df["breakfast"].map({1: "Breakfast_No", 2: "Breakfast_Yes"})

    # One-hot
    transaction_df = pd.get_dummies(df).astype(int)
    print("Shape sau encoding:", transaction_df.shape)

    out = os.path.join(data_dir, "food_association_ready.csv")
    transaction_df.to_csv(out, index=False)
    print(f"[preprocess] Đã lưu: {out}")
    return transaction_df


def validate_and_to_bool(df_encoded):
    """Kiểm tra dữ liệu TRƯỚC khi astype(bool) (như cell kiểm tra của TV2)."""
    if df_encoded.empty:
        raise ValueError("File không có dữ liệu!")
    if df_encoded.columns.duplicated().any():
        raise ValueError("Tên cột bị trùng!")
    if df_encoded.isna().sum().sum() > 0:
        raise ValueError("Dữ liệu có giá trị thiếu!")
    if not df_encoded.isin([0, 1]).all().all():
        raise ValueError("Dữ liệu phải chỉ chứa 0 và 1!")
    return df_encoded.astype(bool)


# ==============================================================================
# BƯỚC 3: KHAI THÁC TẬP PHỔ BIẾN  (Thành viên 2)
# ==============================================================================
def it_tree_mining(df, minsup=0.1):
    n = len(df)
    if n == 0:
        raise ValueError("Dữ liệu rỗng")

    items = [(col, set(df.index[df[col]])) for col in df.columns]
    results = []

    def search(prefix, cur_items):
        for i, (name, tids) in enumerate(cur_items):
            if len(tids) / n < minsup:
                continue
            new_set = prefix | {name}
            results.append((len(tids) / n, frozenset(new_set)))
            next_items = [(n2, tids & t2) for n2, t2 in cur_items[i + 1:]]
            search(new_set, next_items)

    search(set(), items)
    return pd.DataFrame(results, columns=["support", "itemsets"])


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    return res, time.perf_counter() - start


def step_mine(df_bool, data_dir, minsup):
    banner(f"3. KHAI THÁC TẬP PHỔ BIẾN (MinSup = {minsup})")
    fi_ap, t_ap = timed(apriori, df_bool, min_support=minsup, use_colnames=True)
    fi_fp, t_fp = timed(fpgrowth, df_bool, min_support=minsup, use_colnames=True)
    fi_it, t_it = timed(it_tree_mining, df_bool, minsup)

    print(f"Apriori   : {len(fi_ap):>6} tập | {t_ap:.4f}s")
    print(f"FP-Growth : {len(fi_fp):>6} tập | {t_fp:.4f}s")
    print(f"IT-Tree   : {len(fi_it):>6} tập | {t_it:.4f}s")

    same = (set(fi_ap["itemsets"]) == set(fi_fp["itemsets"]) == set(fi_it["itemsets"]))
    print("=> 3 thuật toán cho CÙNG tập phổ biến" if same
          else "=> CẢNH BÁO: 3 thuật toán cho kết quả KHÁC NHAU")
    if not same:
        raise RuntimeError("Kết quả 3 thuật toán không khớp, dừng pipeline.")

    fi_ap.to_csv(os.path.join(data_dir, "frequent_itemsets_apriori.csv"), index=False)
    fi_fp.to_csv(os.path.join(data_dir, "frequent_itemsets_fpgrowth.csv"), index=False)
    fi_it.to_csv(os.path.join(data_dir, "frequent_itemsets_ittree.csv"), index=False)
    print("[mine] Đã lưu 3 file frequent_itemsets_*.csv")
    return fi_fp


# ==============================================================================
# BƯỚC 4: THỰC NGHIỆM MINSUP VS THỜI GIAN  (Thành viên 2)
# ==============================================================================
def step_benchmark(df_bool, data_dir, repeats=3):
    banner("4. THỰC NGHIỆM MINSUP VÀ THỜI GIAN CHẠY")
    minsup_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    for ms in minsup_values:
        t_ap, t_fp, t_it = [], [], []
        for _ in range(repeats):
            ap, d = timed(apriori, df_bool, min_support=ms, use_colnames=True)
            t_ap.append(d)
        for _ in range(repeats):
            fp, d = timed(fpgrowth, df_bool, min_support=ms, use_colnames=True)
            t_fp.append(d)
        for _ in range(repeats):
            it, d = timed(it_tree_mining, df_bool, ms)
            t_it.append(d)
        rows.append({
            "MinSup": ms,
            "Apriori_Time": float(np.median(t_ap)),
            "FP_Growth_Time": float(np.median(t_fp)),
            "IT_Tree_Time": float(np.median(t_it)),
            "Apriori_Itemsets": len(ap),
            "FP_Growth_Itemsets": len(fp),
            "IT_Tree_Itemsets": len(it),
        })
        print(f"  minsup={ms:.2f} xong")

    exp_df = pd.DataFrame(rows)
    print(exp_df.round(4).to_string(index=False))
    exp_df.to_csv(os.path.join(data_dir, "minsup_experiment_3_algorithms.csv"), index=False)

    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for col, mk, lb in [("Apriori_Itemsets", "o", "Apriori"),
                        ("FP_Growth_Itemsets", "s", "FP-Growth"),
                        ("IT_Tree_Itemsets", "^", "IT-Tree")]:
        plt.plot(exp_df["MinSup"], exp_df[col], marker=mk, label=lb)
    plt.xlabel("MinSup"); plt.ylabel("Số tập phổ biến")
    plt.title("MinSup và số tập phổ biến"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "benchmark_itemsets.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(10, 5))
    for col, mk, lb in [("Apriori_Time", "o", "Apriori"),
                        ("FP_Growth_Time", "s", "FP-Growth"),
                        ("IT_Tree_Time", "^", "IT-Tree")]:
        plt.plot(exp_df["MinSup"], exp_df[col], marker=mk, label=lb)
    plt.xlabel("MinSup"); plt.ylabel("Thời gian chạy (giây)")
    plt.title("So sánh thời gian chạy theo MinSup"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "benchmark_time.png"), dpi=120)
    plt.close()
    print(f"[benchmark] Đã lưu CSV + biểu đồ")
    return exp_df


# ==============================================================================
# BƯỚC 5: SINH LUẬT KẾT HỢP  (Thành viên 3)
# ==============================================================================
def make_rules(itemsets, n_rows, min_conf):
    """Tương thích mọi phiên bản mlxtend (bản mới cần num_itemsets)."""
    try:
        return association_rules(itemsets, num_itemsets=n_rows,
                                 metric="confidence", min_threshold=min_conf)
    except TypeError:
        return association_rules(itemsets, metric="confidence", min_threshold=min_conf)


def _stringify_sets(df):
    out = df.copy()
    for c in ("antecedents", "consequents"):
        if c in out.columns:
            out[c] = out[c].apply(lambda s: " + ".join(sorted(s)))
    return out


def step_rules(itemsets, n_rows, data_dir, minconf):
    banner(f"5. SINH LUẬT KẾT HỢP (MinConf = {minconf})")
    rules = make_rules(itemsets, n_rows, minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    meaningful = rules[rules["lift"] > 1.0].copy()
    print(f"Tổng số luật (conf >= {minconf}): {len(rules)}")
    print(f"Số luật có ý nghĩa (Lift > 1.0): {len(meaningful)}")

    top = meaningful.sort_values("lift", ascending=False).head(10)
    print("\nTop 10 luật theo Lift:")
    print(_stringify_sets(top)[["antecedents", "consequents", "support",
                                "confidence", "lift"]].round(3).to_string(index=False))

    _stringify_sets(meaningful).sort_values("lift", ascending=False).to_csv(
        os.path.join(data_dir, "association_rules.csv"), index=False)

    # Độ nhạy theo min_confidence
    sens = []
    for conf in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tmp = make_rules(itemsets, n_rows, conf)
        m = tmp[tmp["lift"] > 1.0]
        sens.append({
            "Min_Confidence": conf,
            "Total_Rules": len(tmp),
            "Meaningful_Rules (Lift > 1)": len(m),
            "Avg_Lift": m["lift"].mean() if len(m) else 0,
            "Max_Lift": m["lift"].max() if len(m) else 0,
        })
    df_sens = pd.DataFrame(sens)
    print("\nĐộ nhạy tham số Min_Confidence:")
    print(df_sens.round(3).to_string(index=False))
    df_sens.to_csv(os.path.join(data_dir, "confidence_sensitivity.csv"), index=False)
    print("[rules] Đã lưu association_rules.csv, confidence_sensitivity.csv")
    return rules


# ==============================================================================
# BƯỚC 6: DIỄN GIẢI TRI THỨC  (Thành viên 3)
# ==============================================================================
def step_interpret(itemsets, n_rows, data_dir):
    banner("6. TOP MẪU HÀNH VI TIÊU BIỂU")
    fi = itemsets.copy()
    fi["length"] = fi["itemsets"].apply(len)
    multi = fi[fi["length"] >= 2].copy()
    multi["student_count"] = (multi["support"] * n_rows).round().astype(int)
    top = multi.sort_values("support", ascending=False).head(10).copy()
    top["Tập hành vi / sở thích"] = top["itemsets"].apply(lambda s: " + ".join(sorted(s)))
    res = top[["Tập hành vi / sở thích", "support", "student_count"]].rename(
        columns={"support": "Support", "student_count": f"So_sinh_vien_tren_{n_rows}"})
    print(res.round(3).to_string(index=False))
    res.to_csv(os.path.join(data_dir, "top_patterns.csv"), index=False)
    print("[interpret] Đã lưu top_patterns.csv")


# ==============================================================================
# BƯỚC CUỐI: TỰ ĐỘNG MỞ GIAO DIỆN STREAMLIT
# ==============================================================================
def _scan_for_app(roots, max_depth=5):
    """Tìm file app.py có dùng streamlit trong các thư mục con của roots."""
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", "env", "site-packages"}
    seen, found = set(), []
    for root in roots:
        root = os.path.abspath(root)
        if root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            if dirpath.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []
            if "app.py" in filenames:
                f = os.path.join(dirpath, "app.py")
                try:
                    if "streamlit" in open(f, encoding="utf-8", errors="ignore").read():
                        found.append(f)
                except OSError:
                    pass
        if found:
            break
    if len(found) > 1:
        print("[app] Tìm thấy nhiều app.py, dùng file đầu tiên:")
        for f in found:
            print("   -", f)
    return found[0] if found else None


def launch_app(data_dir, app_path=None, port=8501):
    """Mở app.py bằng Streamlit. Chặn tại đây đến khi bạn nhấn Ctrl+C."""
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    data_parent = os.path.dirname(os.path.abspath(data_dir))

    app_path_given = bool(app_path)
    if app_path:
        app_path = os.path.abspath(os.path.expanduser(app_path))
        if os.path.isdir(app_path):          # cho phép truyền cả thư mục chứa app.py
            app_path = os.path.join(app_path, "app.py")
        candidates = [app_path]
    else:
        # Tự tìm app.py ở các chỗ thường gặp (cạnh pipeline.py, Source_Code/, thư mục cha của Data...)
        candidates = []
        for base in (here, cwd, data_parent, os.path.dirname(here)):
            candidates.append(os.path.join(base, "app.py"))
            candidates.append(os.path.join(base, "Source_Code", "app.py"))
        candidates = list(dict.fromkeys(os.path.abspath(c) for c in candidates))

    app_path = next((c for c in candidates if os.path.exists(c)), None)

    # Không thấy ở các chỗ quen thuộc -> quét đệ quy các thư mục con (tối đa 5 cấp)
    if app_path is None and not app_path_given:
        app_path = _scan_for_app([here, data_parent, cwd])

    if app_path is None:
        print("[app] Không tìm thấy app.py. Đã tìm ở:")
        for c in candidates:
            print("   -", c)
        print("   (và quét đệ quy trong:", ", ".join(dict.fromkeys([here, data_parent, cwd])) + ")")
        print('[app] Chỉ rõ bằng:  python pipeline.py --app "đường\\dẫn\\app.py"')
        return
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("[app] Chưa cài streamlit. Chạy:  pip install streamlit plotly")
        return

    banner("MỞ GIAO DIỆN STREAMLIT")
    print(f"App : {app_path}")
    print(f"Data: {os.path.abspath(data_dir)}")
    print(f"Địa chỉ: http://localhost:{port}   (nhấn Ctrl+C để dừng)")
    env = dict(os.environ, FOOD_DATA_DIR=os.path.abspath(data_dir))
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)]
    try:
        subprocess.run(cmd, env=env, cwd=os.path.dirname(app_path))
    except KeyboardInterrupt:
        print("\n[app] Đã dừng giao diện.")


# ==============================================================================
# ĐIỀU PHỐI
# ==============================================================================
def run_all(data_dir=DEFAULT_DATA_DIR, minsup=0.10, minconf=0.50, repeats=3, steps=None):
    steps = steps or ALL_STEPS
    os.makedirs(data_dir, exist_ok=True)
    t0 = time.perf_counter()

    df_raw = load_raw(data_dir)

    if "eda" in steps:
        step_eda(df_raw, data_dir)

    # Các bước sau cần bảng nhị phân: tự tạo nếu chưa có / hoặc vừa chạy preprocess
    encoded_path = os.path.join(data_dir, "food_association_ready.csv")
    if "preprocess" in steps or not os.path.exists(encoded_path):
        encoded = step_preprocess(df_raw, data_dir)
    else:
        encoded = pd.read_csv(encoded_path)
        print(f"[load] Dùng lại {encoded_path}")

    need_mining = any(s in steps for s in ("mine", "benchmark", "rules", "interpret"))
    if not need_mining:
        return

    df_bool = validate_and_to_bool(encoded)
    n_rows = len(df_bool)
    print(f"\nSố transaction: {n_rows} | Số item: {df_bool.shape[1]}")

    itemsets = step_mine(df_bool, data_dir, minsup)   # luôn cần cho rules/interpret
    if "benchmark" in steps:
        step_benchmark(df_bool, data_dir, repeats)
    if "rules" in steps:
        step_rules(itemsets, n_rows, data_dir, minconf)
    if "interpret" in steps:
        step_interpret(itemsets, n_rows, data_dir)

    banner(f"HOÀN TẤT PIPELINE trong {time.perf_counter() - t0:.1f}s")
    print(f"Kết quả nằm trong: {os.path.abspath(data_dir)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Pipeline khai thác luật kết hợp")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="Thư mục dữ liệu vào/ra (mặc định: thư mục Data cạnh pipeline.py)")
    p.add_argument("--minsup", type=float, default=0.10)
    p.add_argument("--minconf", type=float, default=0.50)
    p.add_argument("--repeats", type=int, default=3, help="Số lần lặp đo thời gian benchmark")
    p.add_argument("--steps", default="all",
                   help=f"'all' hoặc danh sách cách nhau bởi dấu phẩy: {','.join(ALL_STEPS)}")
    p.add_argument("--no-app", action="store_true", help="Không tự mở Streamlit sau khi chạy xong")
    p.add_argument("--app", default=None, help="Đường dẫn app.py (mặc định: app.py cạnh pipeline.py)")
    p.add_argument("--port", type=int, default=8501)
    # parse_known_args để chạy được cả trong Jupyter/Colab (có thêm -f kernel.json)
    args, _ = p.parse_known_args(argv)

    steps = ALL_STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        p.error(f"Bước không hợp lệ: {bad}. Hợp lệ: {ALL_STEPS}")

    run_all(args.data_dir, args.minsup, args.minconf, args.repeats, steps)

    if not args.no_app:
        launch_app(args.data_dir, args.app, args.port)


if __name__ == "__main__":
    main()
