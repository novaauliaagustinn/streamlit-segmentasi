import streamlit as st
import pandas as pd
import joblib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Kelas DecisionTree dan RandomForest
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data, target, features, depth=0):
        if depth == self.max_depth or len(np.unique(data[target])) == 1:
            most_common_label = Counter(data[target]).most_common(1)[0][0]
            return most_common_label

        best_feature = features[0]
        tree = {best_feature: {}}

        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            if subset.empty:
                most_common_label = Counter(data[target]).most_common(1)[0][0]
                tree[best_feature][value] = most_common_label
            else:
                tree[best_feature][value] = self.fit(subset, target, [f for f in features if f != best_feature], depth + 1)

        self.tree = tree
        return tree

    def predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        if row[feature] in tree[feature]:
            return self.predict_row(row, tree[feature][row[feature]])
        else:
            return Counter(self.tree).most_common(1)[0][0]

    def predict(self, data):
        return data.apply(lambda row: self.predict_row(row, self.tree), axis=1)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def bootstrap_sample(self, data):
        indices = np.random.choice(data.index, size=int(len(data) * self.sample_size), replace=True)
        return data.loc[indices]

    def fit(self, data, target, features):
        for _ in range(self.n_trees):
            sample = self.bootstrap_sample(data)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sample, target, features)
            self.trees.append(tree)

    def predict(self, data):
        tree_preds = np.array([tree.predict(data) for tree in self.trees])
        majority_votes = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(data.shape[0])]
        return majority_votes

# Function for preprocessing data
def preprocess_data(df):
    mappings = {
        'Gender': {'Female': 0, 'Male': 1},
        'City': {'Chicago': 0, 'Houston': 1, 'Los Angeles': 2, 'Miami': 3, 'New York': 4, 'San Francisco': 5},
        'Membership Type': {'Bronze': 0, 'Gold': 1, 'Silver': 2},
        'Discount Applied': {False: 0, True: 1},
        'Satisfaction Level': {'Neutral': 0, 'Satisfied': 1, 'Unsatisfied': 2}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = df.fillna(df.median())
    return df

# Load model
model_file = r'random_forest_model_90.pkl'
try:
    rf_model = joblib.load(model_file)
except Exception as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# Sidebar untuk mengunggah file
st.sidebar.title("Segment Pro")
uploaded_file = st.sidebar.file_uploader("Unggah File CSV", type="csv")

# Fungsi untuk menampilkan rekomendasi
def show_recommendations(segment):
    if segment == 'Neutral':
        st.write("""
        **Rekomendasi untuk Pelanggan dengan Kepuasan Netral:**
        - **Program Loyalitas**: Tawarkan insentif seperti diskon atau program poin untuk meningkatkan pengeluaran dan keterlibatan pelanggan.
        - **Penyesuaian Penawaran**: Personalisasikan penawaran berdasarkan data preferensi untuk menarik perhatian mereka.
        - **Feedback Langsung**: Kumpulkan umpan balik untuk memahami kebutuhan yang belum terpenuhi.
        """)
    elif segment == 'Satisfied':
        st.write("""
        **Rekomendasi untuk Pelanggan dengan Kepuasan Tinggi:**
        - **Retensi Pelanggan**: Pastikan pengalaman tetap konsisten dengan menawarkan layanan premium atau program keanggotaan eksklusif.
        - **Upselling dan Cross-selling**: Promosikan produk atau layanan tambahan untuk meningkatkan nilai transaksi.
        - **Ambassador Program**: Jadikan pelanggan dalam segmen ini sebagai duta merek dengan memberikan penghargaan atas rekomendasi mereka.
        """)
    elif segment == 'Unsatisfied':
        st.write("""
        **Rekomendasi untuk Pelanggan Tidak Puas:**
        - **Strategi Pemulihan**: Hubungi pelanggan secara proaktif dengan penawaran khusus untuk menarik mereka kembali.
        - **Analisis Masalah**: Identifikasi dan atasi penyebab ketidakpuasan berdasarkan data dan umpan balik.
        - **Komunikasi Personal**: Tunjukkan perhatian dengan mengirimkan pesan atau email yang menawarkan solusi untuk pengalaman mereka yang buruk.
        """)

# Fungsi untuk menampilkan ID pelanggan per segmen
def display_customer_ids(segment, data):
    if segment == 'Neutral':
        customer_ids = data[data['Prediction'] == 'Neutral']['Customer ID']
    elif segment == 'Satisfied':
        customer_ids = data[data['Prediction'] == 'Satisfied']['Customer ID']
    elif segment == 'Unsatisfied':
        customer_ids = data[data['Prediction'] == 'Unsatisfied']['Customer ID']
    st.write(f"### ID Pelanggan {segment}:")
    st.write(customer_ids)

# Main Interface
st.title("Segmentasi Pelanggan Berdasarkan Tingkat Kepuasan")
st.write("""
Aplikasi ini menampilkan segmen pelanggan berdasarkan prediksi kepuasan dan rekomendasi untuk setiap segmen.
Silakan pilih segmen untuk melihat detail ID pelanggan dan rekomendasi.
""")

# Memuat data dan prediksi sebelumnya
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("Dataset yang diunggah kosong.")
        else:
            # Preprocessing
            data_preprocessed = preprocess_data(data)

            # Melakukan prediksi
            try:
                features = ['Gender', 'Membership Type', 'Total Spend', 'Discount Applied', 'City']
                predictions = rf_model.predict(data_preprocessed[features])

                # Konversi hasil prediksi ke label yang sesuai
                label_mapping = {0: 'Neutral', 1: 'Satisfied', 2: 'Unsatisfied'}
                predictions_labels = [label_mapping.get(pred, 'Neutral') for pred in predictions]

                # Menambahkan kolom Customer ID dan Prediksi
                data_preprocessed['Prediction'] = predictions_labels
                data_preprocessed['Customer ID'] = data['Customer ID']

                # Pilihan untuk mengunduh hasil
                csv = data_preprocessed[['Customer ID', 'Prediction']].to_csv(index=False)
                st.download_button(
                    label="Unduh Hasil Prediksi",
                    data=csv,
                    file_name="hasil_klasifikasi_segmentasi.csv",
                    mime="text/csv",
                )

                # Analisis Chart: Distribusi Segmentasi
                st.write("### Analisis Distribusi Segmentasi:")
                segment_counts = data_preprocessed['Prediction'].value_counts()
                st.write(segment_counts)

                # Plot Pie Chart
                fig, ax = plt.subplots()
                ax.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3', len(segment_counts)))
                ax.axis('equal')
                st.pyplot(fig)

                # Membuat Bar Chart dengan warna yang sama seperti Pie Chart
                fig, ax = plt.subplots()
                colors = sns.color_palette('Set3', len(segment_counts))
                ax.bar(segment_counts.index, segment_counts.values, color=colors)
                ax.set_xlabel('Segmen Kepuasan')
                ax.set_ylabel('Jumlah Pelanggan')
                ax.set_title('Distribusi Pelanggan Berdasarkan Kepuasan')
                st.pyplot(fig)

                # Pilih Segmen
                segment_choice = st.selectbox("Pilih Segmen untuk Melihat Detail", ['Neutral', 'Satisfied', 'Unsatisfied'])

                # Tampilkan ID pelanggan untuk segmen yang dipilih
                display_customer_ids(segment_choice, data_preprocessed)

                # Tampilkan rekomendasi untuk segmen yang dipilih
                show_recommendations(segment_choice)

            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {e}")
    except Exception as e:
        st.error(f"Error saat membaca file CSV: {e}")
