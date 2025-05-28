# Laporan Proyek Machine Learning - Khansa Maritza A

## Project Overview

Seiring berkembangnya internet dan melimpahnya informasi digital, kebutuhan akan sistem rekomendasi yang efektif semakin penting untuk membantu pengguna menyaring informasi yang relevan. Salah satu pendekatan yang banyak digunakan adalah content-based filtering, yang merekomendasikan item berdasarkan kemiripan konten dengan preferensi pengguna sebelumnya. Menurut Lü et al. (2012), sistem rekomendasi telah menjadi fokus penting di berbagai bidang karena kemampuannya menyederhanakan pengambilan keputusan dalam ruang informasi yang luas.

Selain itu, collaborative filtering juga menjadi teknik dominan karena kemampuannya memanfaatkan pola penilaian pengguna lain untuk memberikan rekomendasi yang lebih personal. Ekstrand et al. (2011) menyatakan bahwa sistem ini tidak bersifat satu-ukuran-untuk-semua, dan efektivitasnya sangat tergantung pada konteks pengguna serta tugas yang ingin didukung. Oleh karena itu, proyek ini mengembangkan dua pendekatan tersebut untuk menyediakan rekomendasi film yang relevan dan disesuaikan dengan perilaku serta preferensi pengguna.

Referensi:
- Lü, L., Medo, M., Yeung, C. H., Zhang, Y. C., Zhang, Z. K., & Zhou, T. (2012). Recommender systems. Physics Reports, 519(1), 1–49.
- Ekstrand, M. D., Riedl, J. T., & Konstan, J. A. (2011). Collaborative filtering recommender systems. Foundations and Trends® in Human–Computer Interaction, 4(2), 81–173.

## Business Understanding

### Problem Statements
- Bagaimana cara merekomendasikan film-film yang memiliki kemiripan karakteristik dengan film yang disukai pengguna, meskipun tanpa data riwayat pengguna?
- Bagaimana sistem dapat menyajikan daftar film yang relevan berdasarkan preferensi pengguna lain yang memiliki pola penilaian serupa?

### Goals
- Mengembangkan sistem rekomendasi berbasis content-based filtering untuk menyarankan film berdasarkan metadata seperti genre.
- Mengembangkan sistem rekomendasi berbasis collaborative filtering untuk menyarankan film berdasarkan pola penilaian pengguna lain.
- Menyediakan top-N rekomendasi film (misalnya 5 film) yang relevan berdasarkan masukan pengguna atau perilaku pengguna lain.

### Solution approach
- Menggunakan Content-Based Filtering
Menggunakan genre sebagai fitur utama, dilakukan ekstraksi dan vektorisasi menggunakan TF-IDF untuk kemudian dihitung similarity antar film menggunakan cosine similarity. Dengan pendekatan ini, sistem dapat menyarankan film yang mirip secara konten dengan film yang disukai pengguna.

- Menggunakan Collaborative Filtering
Menggunakan Neural Collaborative Filtering dengan embedding user dan item (film), dilatih untuk memprediksi skor rating dan memberikan rekomendasi berdasarkan pola rating pengguna lain. Model ini efektif menangkap hubungan tersembunyi antara pengguna dan film.

## Data Understanding
**Sumber Dataset**: [Movie Recommendation System - parasharmanas](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)  
- Tidak ada data missing values
- Dataset yang digunakan terdiri dari dua file utama, yaitu:
    - ratings.csv: Data rating pengguna terhadap film.
    - movies.csv: Informasi mengenai film.

- Struktur Dataset
    - movies.csv
  
| Dataset    | Kolom     | Tipe Data | Non-Null Count | Deskripsi                           |   
| ---------- | --------- | --------- | -------------- | ----------------------------------- | 
| **movies** | `movieId` | int64     | 62,423         | ID unik untuk tiap film             |  
|            | `title`   | object    | 62,423         | Judul film                          |   
|            | `genres`  | object    | 62,423         | Genre film                          | 

<br>
    - ratings.csv
    
| Dataset     | Kolom       | Tipe Data | Non-Null Count | Deskripsi                                                |
| ----------- | ----------- | --------- | -------------- | -------------------------------------------------------- |
| **ratings** | `userId`    | int64     | 25,000,095     | ID pengguna                                              |
|             | `movieId`   | int64     | 25,000,095     | ID film (relasi ke `movies`)                             |
|             | `rating`    | float64   | 25,000,095     | Rating yang diberikan pengguna terhadap film (0.5 - 5.0) |
|             | `timestamp` | int64     | 25,000,095     | Waktu rating diberikan (format epoch timestamp)          |


## Data Preparation
Tahapan data preparation dilakukan untuk mempersiapkan data sebelum diterapkan ke dalam model sistem rekomendasi. Berikut ini adalah langkah-langkah yang dilakukan secara berurutan:

### Menggabungkan Dataset ratings dan movies
Dataset ratings dan movies digabungkan berdasarkan kolom movieId menggunakan metode left join. Tujuannya adalah untuk mengaitkan setiap rating dengan informasi judul dan genre film terkait. Selain itu, kolom `timestamp` dihapus karena tidak diperlukan dalam proses analisis dan tidak memberikan informasi relevan terhadap sistem rekomendasi.
```
df = pd.merge(ratings, movies, on='movieId', how='left')
df = df.drop(columns=['timestamp'])
```
### Pengecekan Missing Value
Langkah ini dilakukan untuk memastikan bahwa tidak terdapat nilai yang hilang (missing value) pada data yang akan digunakan. Hal ini penting agar model tidak gagal membaca atau memproses data.
```
df.isnull().sum()
```

### Pengurutan dan Pengambilan Sampel Data
Data diurutkan berdasarkan movieId secara menaik untuk memudahkan pengelompokan dan analisis selanjutnya.
Kemudian, diambil sampel sebanyak 100.000 baris secara acak dari data untuk mengurangi penggunaan memori dan mempercepat proses pemodelan tanpa mengorbankan keragaman data.
```
df_fix = df.sort_values('movieId', ascending=True)
df_fix = df.sample(100000, random_state=42)
```
### Mengecek Jumlah Film Unik dan Kategori Genre
Untuk memahami keberagaman data, dilakukan pengecekan terhadap jumlah film unik serta genre film yang terdapat pada dataset hasil sampel `df_fix`.
Langkah ini membantu dalam memahami dimensi data dan variasi genre sebagai salah satu fitur penting dalam sistem rekomendasi berbasis konten.
```
len(df_fix.movieId.unique())
df_fix.genres.unique()
```
### Persiapan Data untuk Analisis Selanjutnya
Data `df_fix` disalin ke dalam variabel preparation dan diurutkan berdasarkan `movieId`.
Selanjutnya, dilakukan penghapusan data duplikat berdasarkan kolom `movieId` agar setiap film hanya direpresentasikan satu kali. Langkah ini penting agar model tidak bias terhadap film yang muncul lebih dari sekali.
```
preparation = df_fix
preparation.sort_values('movieId')
preparation = preparation.drop_duplicates('movieId')
```
### Konversi Data ke Bentuk List dan Pembuatan DataFrame Baru
Kolom movieId, title, dan genres dikonversi ke dalam bentuk list agar lebih mudah diolah.
Kemudian, list tersebut digunakan untuk membentuk DataFrame baru movie_new yang memiliki tiga kolom utama: id, title, dan genre.
```
movie_id = preparation['movieId'].tolist()
movie_name = preparation['title'].tolist()
movie_genre = preparation['genres'].tolist()

movie_new = pd.DataFrame({
    'id': movie_id,
    'title': movie_name,
    'genre': movie_genre
})
```
## Modeling

Dalam membangun sistem rekomendasi berbasis konten (*Content-Based Filtering*), terdapat beberapa metode utama yang digunakan untuk mengukur kemiripan antar item (dalam hal ini: film).

### 1. Content-Based Filtering dengan TF-IDF dan Cosine Similarity

Content-Based Filtering adalah pendekatan sistem rekomendasi yang menyarankan item (misalnya film) berdasarkan kemiripan kontennya dengan item yang disukai pengguna sebelumnya. Dalam konteks proyek ini, metadata seperti genre film digunakan untuk membangun representasi fitur setiap film.

#### Langkah-Langkah Modeling:

- **Preprocessing Genre**

Genre yang memiliki label tidak jelas seperti `(no genres listed)` diganti menjadi `'Unknown'`, dan format diseragamkan (contoh: `'Sci-Fi'` diubah menjadi `'SciFi'`).

- **TF-IDF Vectorizer**  
 
TF-IDF (*Term Frequency–Inverse Document Frequency*) merupakan metode representasi teks ke dalam bentuk numerik yang mempertimbangkan seberapa sering suatu kata muncul dalam satu dokumen (*term frequency*) dan seberapa jarang kata tersebut muncul di seluruh dokumen (*inverse document frequency*).  
Pada sistem rekomendasi ini, TF-IDF digunakan untuk mengubah data genre film dari bentuk teks menjadi bentuk vektor numerik. Setiap genre diberikan bobot berdasarkan tingkat kepentingannya dalam membedakan satu film dengan film lainnya.

  ```
  # Fit dan transform genre menjadi matriks TF-IDF
  tfidf_matrix = tf.fit_transform(data['genre'])
  ```
 - **Cosine Similarity**

Cosine similarity adalah metode pengukuran kemiripan antar dua vektor dalam ruang berdimensi banyak dengan cara menghitung nilai cosinus dari sudut antar vektor tersebut.
Dalam konteks ini, vektor yang dibandingkan merupakan representasi TF-IDF dari genre film. Nilai cosine similarity berkisar antara 0 hingga 1: <br>
    - Nilai mendekati 1 menunjukkan bahwa dua film memiliki genre yang sangat mirip. <br>
    - Nilai mendekati 0 menunjukkan bahwa dua film memiliki genre yang sangat berbeda.

```
    # Menghitung cosine similarity pada matriks TF-IDF
    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim
```

- **Fungsi Rekomendasi**

Fungsi `movie_recommendations()` dibuat untuk mengembalikan rekomendasi top-N (default: 5) film yang memiliki nilai kemiripan tertinggi terhadap film input.
```
def movie_recommendations(movie_name, similarity_data=cosine_sim_df, items=movie_new[['movie_name', 'genre']], k=5):
    """
    Rekomendasi film berdasarkan kemiripan genre film.

    Parameter:
    ---
    movie_name : str
        Judul film yang ingin dicari rekomendasinya (harus ada di index similarity dataframe)
    similarity_data : pd.DataFrame
        Dataframe kemiripan (cosine similarity) simetrik antar film
    items : pd.DataFrame
        Data film berisi nama film dan genre
    k : int
        Jumlah rekomendasi yang ingin ditampilkan
    """

    # Ambil posisi k film dengan similarity terbesar terhadap movie_name
    index = similarity_data.loc[:, movie_name].to_numpy().argpartition(range(-1, -k-1, -1))

    # Ambil nama film dengan similarity tertinggi
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Hapus movie_name agar tidak muncul di rekomendasi
    closest = closest.drop(movie_name, errors='ignore')

    # Merge dengan data film asli agar dapat informasi genre
    return pd.DataFrame(closest, columns=['movie_name']).merge(items, on='movie_name').head(k)
```

- **Top N Recommendation Output**

Di sini user memilih film Grudge 3 dengan genre "Horror" dan sistem akan merekomendasikan top 5 dari film dengan genre yang sama. Contoh output tersebut dapat dilihat di Gambar berikut: <br>
![image](https://github.com/user-attachments/assets/5f8a8900-563c-4416-a8cb-1e279fbda647)

####  Kelebihan dan Kekurangan Content-Based Filtering
**Kelebihan:**
- Tidak bergantung pada data pengguna lain (tidak memerlukan interaksi pengguna seperti Collaborative Filtering).
- Dapat memberikan rekomendasi untuk film baru selama informasinya tersedia (misalnya genre).
- Cocok untuk sistem rekomendasi awal (cold-start problem pada pengguna dapat diminimalisasi).

**Kekurangan:**
- Terbatas hanya pada fitur yang digunakan (dalam hal ini genre). Jika dua film memiliki genre yang sama namun sangat berbeda dalam nuansa atau kualitas, mereka tetap dianggap mirip.
- Kurang mampu menangkap konteks atau selera kompleks seperti preferensi berdasarkan aktor, sutradara, atau rating.

### 2. Collaborative Filtering dengan Matrix Factorization (Neural Network Embedding)

Collaborative Filtering adalah pendekatan yang merekomendasikan item berdasarkan interaksi (rating) antara user dan item. Di sini, sistem belajar dari pola penilaian pengguna lain untuk membuat rekomendasi, tanpa melihat isi konten filmnya.

#### Langkah-Langkah Modeling:
- **Encoding User dan Movie**

Untuk dapat memanfaatkan embedding layer pada neural network, ID pengguna dan ID film perlu diubah ke dalam bentuk integer (numerik). Proses encoding ini dilakukan dengan membuat mapping dari ID asli ke indeks numerik.

```
# Mapping userId dan movieId ke index numerik
user_ids = ratings['userId'].unique().tolist()
movie_ids = ratings['movieId'].unique().tolist()

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}

ratings['user'] = ratings['userId'].map(user_to_user_encoded)
ratings['movie'] = ratings['movieId'].map(movie_to_movie_encoded)
```
Alasan:

Encoding diperlukan karena embedding layer hanya dapat memproses input dalam bentuk integer. Teknik ini memungkinkan model mempelajari representasi vektor dari pengguna dan film, sehingga bisa menangkap pola-pola kompleks dalam preferensi pengguna terhadap film yang tidak langsung terlihat dari fitur eksplisit seperti genre.
	
- **Normalisasi Rating**

Rating film dinormalisasi ke rentang 0 hingga 1 agar sesuai dengan output dari fungsi aktivasi sigmoid yang digunakan pada model.
```
min_rating = ratings['rating'].min()
max_rating = ratings['rating'].max()

ratings['rating'] = ratings['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
```
Alasan:
Karena fungsi aktivasi sigmoid digunakan pada output layer model, maka normalisasi rating sangat penting agar nilai target berada dalam rentang [0, 1]. Ini membantu proses pelatihan model menjadi lebih stabil dan akurat.

- **Pemisahan Data**

Data diacak dan kemudian dipisahkan menjadi data pelatihan dan validasi dengan rasio 80:20.
```
x = ratings[['user', 'movie']].values
y = ratings['rating'].values

train_indices = int(0.8 * ratings.shape[0])
x_train, x_val, y_train, y_val = x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:]
```
- **Arsitektur Model Neural Network**

Model rekomendasi dibuat dengan arsitektur berbasis tf.keras.Model subclassing. Model ini memiliki dua embedding layer, masing-masing untuk pengguna dan film. Dot product dari kedua embedding tersebut akan menghasilkan estimasi rating.
```
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_size)
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.movie_bias = tf.keras.layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        user_bias = self.user_bias(inputs[:, 0])
        movie_bias = self.movie_bias(inputs[:, 1])
        
        dot_product = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        x = dot_product + user_bias + movie_bias
        return tf.nn.sigmoid(x)
```
Alasan:

Model ini menerapkan teknik Matrix Factorization dengan neural network, yang efektif dalam menangkap interaksi laten antara pengguna dan item. Penggunaan embedding layer dan bias membantu model memahami preferensi personal pengguna dan popularitas film tertentu. Dot product digunakan untuk mengukur kesesuaian antara representasi pengguna dan film.

- **Kompilasi Model**

Model di-compile menggunakan fungsi loss BinaryCrossentropy, optimizer Adam, dan metrik evaluasi RootMeanSquaredError.
```
model = RecommenderNet(num_users, num_movies, 50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
Alasan:
- BinaryCrossentropy digunakan karena rating sudah dinormalisasi menjadi nilai antara 0 dan 1.
- Adam dipilih sebagai optimizer karena efisien dan mampu menangani masalah sparse gradients.
- RMSE dipakai karena sesuai dengan tujuan sistem rekomendasi yang ingin meminimalkan selisih antara rating prediksi dan rating aktual.

- **Proses Pelatihan Model**

Model dilatih menggunakan data training dan divalidasi dengan data validasi selama 50 epoch dengan ukuran batch sebesar 8.
```
history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    batch_size=8,
    epochs=50
)
```
- **Visualisasi Hasil Pelatihan**
Visualisasi grafik RMSE (Root Mean Squared Error) pada data pelatihan dan validasi digunakan untuk memantau performa model sepanjang proses pelatihan dan validasi.
![image](https://github.com/user-attachments/assets/49d2a3a2-3e1a-421f-8a0e-50fe31e7e461)


Alasan:

Grafik RMSE membantu mengidentifikasi kapan model mulai overfitting, stagnan, atau masih bisa ditingkatkan. Ini juga berguna untuk menentukan epoch optimal jika diperlukan early stopping.

- **Prediksi dan Rekomendasi Film**
    - Prediksi Rating Film Belum Ditonton
      Setelah model collaborative filtering dilatih, sistem akan memprediksi rating terhadap film yang belum pernah ditonton oleh seorang user berdasarkan pola interaksi user lain. Prosesnya dilakukan dengan membentuk pasangan [user_id, movie_id] dan memasukkannya ke dalam model. Model kemudian mengeluarkan skor prediksi (rating estimasi).

    - Top-N Recommendation Output
      Setelah seluruh prediksi dibuat, sistem akan memilih 10 film dengan rating tertinggi sebagai rekomendasi personal untuk user tersebut. Output tersebut dapat dilihat pada Gambar di bawah: <br>
![image](https://github.com/user-attachments/assets/0c493fbf-4582-45ac-b082-ac7199efc42d)

#### Kelebihan dan Kekurangan Collaborative Filtering
**Kelebihan:**
- Dapat menangkap pola kompleks berdasarkan interaksi historis pengguna.
- Tidak memerlukan informasi fitur tambahan dari item (misalnya genre atau sinopsis).

**Kekurangan:**
- Rentan terhadap masalah cold-start (untuk pengguna atau film baru yang belum memiliki interaksi).
- Performa sangat tergantung pada kualitas dan kuantitas data interaksi.

## Evaluation
### Evaluasi Model Content-Based Filtering
#### Metrik Evaluasi: Precision@K
Precision@K adalah salah satu metrik evaluasi yang umum digunakan dalam sistem rekomendasi, khususnya untuk pendekatan top-N recommendation. Metrik ini mengukur proporsi item yang relevan dalam daftar rekomendasi yang diberikan kepada pengguna.
Rumus perhitungannya adalah sebagai berikut: <br>
![image](https://github.com/user-attachments/assets/ef88c7db-8072-4119-83c9-2868dc449b51)

Sebuah item dianggap relevan apabila memiliki karakteristik yang sesuai dengan preferensi pengguna atau item acuan, dalam konteks ini yaitu genre film.
#### Hasil Evaluasi
Model content-based filtering memberikan rekomendasi berdasarkan kesamaan fitur antar film, khususnya genre. Misalnya, ketika pengguna memilih film Grudge 3, The (2009) yang bergenre Horror, sistem memberikan lima rekomendasi film lain yang juga bergenre Horror:
- KM 31: Kilometro 31 (2006)
- Circus of Horrors (1960)
- Requiem for a Vampire (1971)
- Halloween 5: The Revenge of Michael Myers (1989)
- Black Sabbath (1963)

Kelima film memiliki genre yang relevan dengan film yang dipilih, sehingga:
Precision@5= 1.0 atau 100%

#### Kesimpulan:
Model content-based filtering menunjukkan performa yang sangat baik dalam menjaga konsistensi genre dalam rekomendasinya. Precision@5 yang mencapai 100% menunjukkan bahwa sistem ini efektif dalam merekomendasikan film dengan karakteristik yang serupa, cocok untuk pengguna dengan preferensi genre yang spesifik.
### Evaluasi Model Collaborative Filtering
#### Metrik Evaluasi: Root Mean Squared Error (RMSE)
RMSE merupakan metrik yang umum digunakan untuk mengukur performa model dalam memprediksi nilai numerik, seperti rating. Metrik ini mengukur rata-rata kesalahan antara rating sebenarnya dan rating yang diprediksi oleh model.
Rumus RMSE adalah sebagai berikut: <br>
![image](https://github.com/user-attachments/assets/e8085153-d6c8-49b0-b9d3-3f5b1afaee94)

#### Hasil Evaluasi
- Nilai RMSE pada data pelatihan akhir: sekitar 0.12
- Nilai RMSE pada data validasi: sekitar 0.23
- Kisaran RMSE selama pelatihan: 0.12 – 0.26

Model collaborative filtering menunjukkan proses pelatihan yang stabil dan konvergen pada sekitar epoch ke-50. Nilai RMSE yang cukup rendah menunjukkan bahwa model mampu memprediksi rating pengguna dengan akurasi yang baik.
#### Kesimpulan:
Dengan hasil RMSE yang rendah, model collaborative filtering terbukti efektif dalam memahami pola preferensi pengguna berdasarkan interaksi historis. Model ini sangat berguna dalam memberikan prediksi rating yang akurat, yang menjadi dasar dalam menyusun daftar rekomendasi.

### Evaluasi terhadap Goals Proyek
- Goals 1: Mengembangkan sistem rekomendasi berbasis content-based filtering untuk menyarankan film berdasarkan metadata seperti genre.
<br> ✅ Tercapai. Sistem berhasil mengidentifikasi kesamaan antar film berdasarkan genre dan metadata lainnya, sehingga mampu merekomendasikan film yang mirip secara konten dengan film yang disukai pengguna.

- Goals 2: Mengembangkan sistem rekomendasi berbasis collaborative filtering untuk menyarankan film berdasarkan pola penilaian pengguna lain.
<br> ✅ Tercapai. Model dengan pendekatan matrix factorization berbasis neural network embedding berhasil mempelajari pola interaksi antar user dan film. Sistem mampu memberikan rekomendasi yang personalized berdasarkan preferensi pengguna lain yang serupa. Validasi performa dilakukan melalui metrik RMSE dan visualisasi hasil pelatihan.

- Goals 3: Menyediakan top-N rekomendasi film (misalnya 10 film) yang relevan berdasarkan masukan pengguna atau perilaku pengguna lain.
<br> ✅ Tercapai. Sistem menghasilkan 10 rekomendasi film untuk user tertentu berdasarkan film yang memiliki rating tinggi oleh user tersebut. Film yang direkomendasikan memiliki karakteristik serupa, baik dari segi genre maupun gaya cerita, menunjukkan kemampuan sistem dalam menangkap preferensi pengguna secara akurat.
