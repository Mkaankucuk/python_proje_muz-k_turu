# Müzik Türü Sınıflandırma

Bu proje, ses kayıtlarından müzik türü tahmini yapmak için hazırlanmış bir derin öğrenme uygulamasıdır. GTZAN veri setindeki müzik parçalarından MFCC özellikleri çıkarılmış, farklı model mimarileri eğitilmiş ve seçilen modeller Flask tabanlı web arayüzüne entegre edilmiştir.

## Proje Özeti

- Veri seti: GTZAN Music Genre Dataset
- Yöntem: MFCC özellik çıkarımı + derin öğrenme modelleri
- Modeller: CNN, CNN + LSTM, Simple RNN, DenseNet, Vision Transformer
- Arayüz: Flask, HTML, CSS
- Girdi: Ses dosyası veya YouTube bağlantısı
- Çıktı: Tahmin edilen müzik türü, güven oranı ve tür olasılıkları
- Mobil uygulama: `mobil_uygulama/` klasöründe Flutter projesi olarak yer alır

## Mobil Uygulama

Flutter mobil uygulaması `mobil_uygulama/` klasöründedir. Bu uygulama, kullanıcının seçtiği ses dosyasını analiz ederek eğitilmiş TensorFlow Lite modeliyle müzik türü tahmini yapar.

Mobil uygulamayı çalıştırmak için:

```bash
cd mobil_uygulama
flutter pub get
flutter run
```

Mobil uygulamanın kullandığı model dosyaları:

```text
mobil_uygulama/assets/models/genre_classifier.tflite
mobil_uygulama/assets/models/labels.json
mobil_uygulama/assets/models/feature_engineering_normalization.json
```

## Veri Seti

Projede GTZAN Music Genre Dataset kullanılmıştır.

Veri seti bağlantısı:

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Veri setinde 10 müzik türü bulunur:

```text
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
```

Projede beklenen veri seti yolu:

```text
dataset/genres_original/
```

Her müzik dosyası yaklaşık 30 saniyedir. Ön işleme sırasında her dosya yaklaşık 3 saniyelik 10 segmente bölünür. Her segmentten MFCC özellikleri çıkarılır ve aşağıdaki dosyaya kaydedilir:

```text
dataset/features_3.0_sec.json
```

Not: `jazz.00054.wav` dosyası okunamadığı için ön işleme sırasında atlanmıştır. Toplam `9990` segment oluşturulmuştur.

## Kullanılan Kütüphaneler

Projede kullanılan temel kütüphaneler `requirements.txt` dosyasında verilmiştir.

| Kütüphane | Kullanım amacı |
|---|---|
| TensorFlow / Keras | Derin öğrenme modellerini oluşturma, eğitme ve kaydetme |
| NumPy | Sayısal işlemler ve dizi yapıları |
| Librosa | Ses dosyalarını okuma, MFCC ve spectrogram çıkarma |
| SciPy | Frekans analizi ve yardımcı bilimsel işlemler |
| Scikit-learn | Train-test ayrımı, metrikler ve confusion matrix |
| Pandas | Model sonuç tablolarını oluşturma |
| Matplotlib | Accuracy/loss grafikleri ve görselleştirme |
| h5py | `.h5` model dosyalarıyla uyumluluk |
| Flask | Web uygulaması |
| yt-dlp | YouTube bağlantısından ses indirme |
| imageio-ffmpeg | YouTube sesini WAV formatına dönüştürme desteği |

## Donanım ve Yazılım Gereksinimleri

Önerilen yazılım:

- Windows 10 veya Windows 11
- Python 3.12
- Git
- Git LFS

Önerilen donanım:

- En az 8 GB RAM
- Eğitim için tercihen 16 GB RAM
- CPU ile çalışabilir
- GPU zorunlu değildir, ancak eğitim süresini kısaltır
- Veri seti ve modeller için en az 4 GB boş disk alanı önerilir

## Git LFS Gereksinimi

Veri seti, `features_3.0_sec.json` ve bazı model dosyaları büyük boyutlu olduğu için Git LFS ile takip edilmektedir. Projeyi başka bir bilgisayarda eksiksiz çalıştırmak için Git LFS kurulmalıdır.

Git LFS kurulumu:

https://git-lfs.com/

Kurulumdan sonra terminalde:

```powershell
git lfs install
```

## Projeyi İndirme

Önerilen yöntem Git ile klonlamaktır. GitHub üzerinden ZIP indirmek, Git LFS dosyalarını eksik veya pointer dosyası olarak indirebilir.

```powershell
git lfs install
git clone -b muhammed-dev https://github.com/Mkaankucuk/python_proje_muz-k_turu.git
cd python_proje_muz-k_turu
git lfs pull
```

Bu işlemden sonra veri seti, özellik dosyası ve model dosyaları da indirilmiş olur.

## Kurulum

Proje klasöründe PowerShell açıp aşağıdaki komutları çalıştırın:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Kurulum tamamlandıktan sonra web uygulamasını başlatın:

```powershell
python app.py
```

Tarayıcıdan şu adresi açın:

```text
http://127.0.0.1:5000
```

## Çalıştırma Komutu

Kısa çalıştırma komutları:

```powershell
cd python_proje_muz-k_turu
.\.venv\Scripts\activate
python app.py
```

## Örnek Veri

Projede örnek ses dosyaları veri seti içinde bulunmaktadır. Örneğin:

```text
dataset/genres_original/blues/blues.00000.wav
dataset/genres_original/classical/classical.00000.wav
```

Web arayüzünde bu dosyalardan biri seçilerek tür tahmini yapılabilir.

## Eğitilmiş Model Dosyaları

Eğitilmiş model dosyaları `models/` klasöründe yer alır.

Web uygulamasında kullanılan temel modeller:

```text
models/best_model.h5
models/best_rnn_lstm_model.h5
models/feature_engineered_densenet.h5
models/feature_engineered_vit.h5
```

Ayrıca Keras formatındaki karşılıkları da bulunmaktadır:

```text
models/best_model.keras
models/best_rnn_lstm_model.keras
models/feature_engineered_densenet.keras
models/feature_engineered_vit.keras
```

## Veritabanı Durumu

Bu projede veritabanı kullanılmamıştır. Bu nedenle veritabanı oluşturma adımı, dump dosyası veya demo kullanıcı bilgisi yoktur.

Tahmin işlemleri dosya yükleme ve model çıkarımı üzerinden yapılır. Kullanıcı hesabı veya oturum sistemi bulunmaz.

## `.env` Dosyası Durumu

Bu projede API anahtarı veya gizli ortam değişkeni kullanılmamıştır. Bu nedenle `.env` veya `.env.example` dosyasına ihtiyaç yoktur.

## Klasör Yapısı

```text
music-genre-classification/
│
├── app.py
├── prediction_service.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── genres_original/
│   └── features_3.0_sec.json
│
├── models/
│   ├── best_model.h5
│   ├── best_rnn_lstm_model.h5
│   ├── feature_engineered_densenet.h5
│   └── feature_engineered_vit.h5
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── cnn_model.ipynb
│   ├── rnn_lstm_model.ipynb
│   ├── densenet_vit_model.ipynb
│   ├── feature_engineering_model.ipynb
│   └── load_model_cnn.ipynb
│
├── static/
│   └── styles.css
│
├── templates/
│   └── index.html
│
└── uploads/
```

Klasör açıklamaları:

- `dataset/`: Veri seti ve çıkarılmış MFCC özellikleri
- `models/`: Eğitilmiş model dosyaları, grafikler ve karşılaştırma tabloları
- `notebooks/`: Ön işleme, eğitim ve model yükleme notebook'ları
- `static/`: CSS dosyaları
- `templates/`: HTML arayüz dosyaları
- `uploads/`: Web arayüzünden yüklenen geçici ses dosyaları

## Ön İşleme

Ön işleme notebook'u:

```text
notebooks/preprocessing.ipynb
```

Bu notebook:

1. Veri setindeki tür klasörlerini kontrol eder.
2. Örnek ses dosyası için waveform, frekans spektrumu, mel spectrogram ve MFCC görselleştirmesi üretir.
3. Her 30 saniyelik ses dosyasını 10 parçaya böler.
4. Her 3 saniyelik segmentten 13 MFCC katsayısı çıkarır.
5. Çıkarılan özellikleri `dataset/features_3.0_sec.json` dosyasına kaydeder.

## Model Eğitimi

Notebook dosyaları:

```text
notebooks/cnn_model.ipynb
notebooks/rnn_lstm_model.ipynb
notebooks/densenet_vit_model.ipynb
notebooks/feature_engineering_model.ipynb
```

Eğitim notebook'larında:

- Training accuracy
- Validation accuracy
- Test accuracy
- Loss
- Validation loss
- Accuracy/loss grafikleri
- Confusion matrix
- Model karşılaştırma tabloları

bulunmaktadır.

## Model Sonuçları

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| CNN baseline | 0.7005 | 0.6723 |
| CNN BatchNorm + Dropout | 0.6855 | 0.6657 |
| Simple RNN | 0.3975 | 0.3919 |
| CNN + LSTM | 0.7863 | 0.7684 |
| Compact DenseNet | 0.7598 | 0.7421 |
| Compact Vision Transformer | 0.5368 | 0.5138 |
| Feature Engineered DenseNet | 0.7748 | 0.7618 |
| Feature Engineered Vision Transformer | 0.5833 | 0.5542 |

Model karşılaştırmasında temel ölçüt olarak validation accuracy kullanılmıştır. Web arayüzünde birden fazla model seçilebildiği için farklı modellerin tahmin sonuçları karşılaştırılabilir.

## Web Arayüzü

Flask web arayüzünde:

- Model seçimi yapılabilir.
- Bilgisayardan ses dosyası yüklenebilir.
- YouTube şarkı bağlantısı girilebilir.
- Tahmin edilen tür gösterilir.
- Güven oranı ve tüm tür olasılıkları listelenir.
- Açık tema ve koyu tema kullanılabilir.

## Notlar

- Veri seti ve büyük dosyalar Git LFS ile indirildiği için `git lfs pull` komutu önemlidir.
- Sanal ortam klasörü `.venv/` repoya eklenmemiştir. Her bilgisayarda yeniden oluşturulmalıdır.
- YouTube tahmini için internet bağlantısı gerekir.
- Veritabanı, demo kullanıcı ve `.env` dosyası gerektiren bir yapı bulunmamaktadır.
