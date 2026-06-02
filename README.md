# Müzik Türü Sınıflandırma

Bu proje, ses kayıtlarından müzik türü tahmini yapmak amacıyla geliştirilmiş bir derin öğrenme uygulamasıdır. GTZAN veri setindeki müzik parçaları üzerinde farklı model mimarileri eğitilmiş, sonuçlar karşılaştırılmış ve seçilen modeller Flask tabanlı bir web arayüzüne entegre edilmiştir.

## Kullanılan Veri Seti

Projede GTZAN Music Genre Dataset kullanılmıştır. Veri setinde 10 müzik türü bulunmaktadır:

```text
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
```

Ses dosyaları aşağıdaki klasörde yer alır:

```text
dataset/genres_original/
```

Her parça yaklaşık 30 saniyedir. Ön işleme sırasında parçalar yaklaşık 3 saniyelik segmentlere ayrılır. `jazz.00054.wav` dosyası okunamadığı için atlanmış ve toplam `9990` segment üretilmiştir.

## Özellik Çıkarımı

Ses sinyalleri doğrudan modele verilmez. Öncelikle MFCC (Mel-Frequency Cepstral Coefficients) özellikleri çıkarılır. MFCC, bir ses kaydının frekans karakteristiğini insan işitme algısına yakın bir biçimde temsil eder.

Ön işleme notebook'u:

```text
notebooks/preprocessing.ipynb
```

Üretilen özellik dosyası:

```text
dataset/features_3.0_sec.json
```

DenseNet ve Vision Transformer denemelerinde MFCC özelliklerine ek olarak sesin zamansal değişimini temsil eden `delta` ve `delta-delta` özellikleri kullanılmıştır:

```text
MFCC + delta + delta-delta
```

## Eğitilen Modeller

Projede farklı mimarilerin performansını incelemek için şu modeller eğitilmiştir:

- CNN
- CNN + LSTM
- Simple RNN
- Compact DenseNet
- Compact Vision Transformer
- Feature Engineered DenseNet
- Feature Engineered Vision Transformer

CNN ve CNN + LSTM ses verilerindeki yerel örüntüleri öğrenmek için kullanılmıştır. RNN ve LSTM katmanları zamana bağlı ilişkileri incelemek amacıyla eklenmiştir. DenseNet ve Vision Transformer modellerinde özellik mühendisliğinin sonuçlara etkisi ayrıca değerlendirilmiştir.

## Model Sonuçları

### CNN Denemeleri

| Model | Training accuracy | Validation accuracy | Test accuracy | Loss | Validation loss | Test loss |
|---|---:|---:|---:|---:|---:|---:|
| CNN baseline | 0.9106 | 0.7005 | 0.6723 | 0.3071 | 1.0059 | 1.0449 |
| CNN BatchNorm + Dropout | 0.7277 | 0.6855 | 0.6657 | 0.8365 | 1.0857 | 1.0901 |

### RNN ve LSTM Denemeleri

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Simple RNN | 0.3975 | 0.3919 |
| CNN + LSTM | 0.7863 | 0.7684 |

### DenseNet ve Vision Transformer Denemeleri

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| Compact DenseNet | 0.7598 | 0.7421 |
| Compact Vision Transformer | 0.5368 | 0.5138 |
| Feature Engineered DenseNet | 0.7748 | 0.7618 |
| Feature Engineered Vision Transformer | 0.5833 | 0.5542 |

Model karşılaştırmalarında en yüksek validation accuracy değeri dikkate alınmıştır. Web uygulamasında birden fazla eğitilmiş model seçilebildiği için farklı mimarilerin tahmin sonuçları doğrudan karşılaştırılabilir.

## Kaydedilen Modeller

Model dosyaları `models/` klasöründe bulunur. Hem güncel Keras biçimi hem de uyumluluk amacıyla H5 biçimi kullanılmıştır.

Web uygulamasında seçilebilen temel modeller:

```text
models/best_model.h5
models/best_rnn_lstm_model.h5
models/feature_engineered_densenet.h5
models/feature_engineered_vit.h5
```

Eğitim notebook'larında epoch çıktıları, accuracy-loss grafikleri, confusion matrix görselleri ve model karşılaştırma tabloları yer almaktadır.

## Notebook Dosyaları

```text
notebooks/preprocessing.ipynb
notebooks/cnn_model.ipynb
notebooks/rnn_lstm_model.ipynb
notebooks/densenet_vit_model.ipynb
notebooks/feature_engineering_model.ipynb
notebooks/load_model_cnn.ipynb
```

## Web Uygulaması

Flask tabanlı web arayüzünde:

- Eğitilmiş model seçilebilir.
- Bilgisayardan ses dosyası yüklenebilir.
- YouTube bağlantısı ile müzik türü tahmini yapılabilir.
- Tahmin edilen tür, güven oranı ve tüm türlere ait olasılıklar görüntülenebilir.
- Aydınlık ve koyu tema arasında geçiş yapılabilir.

## Kurulum ve Çalıştırma

Python 3.12 kurulu olmalıdır. Proje klasöründe PowerShell açarak aşağıdaki komutları çalıştırın:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Uygulama başladıktan sonra tarayıcıdan şu adresi açın:

```text
http://127.0.0.1:5000
```

## Modelleri Yeniden Eğitme

Notebook'ları çalıştırmak için:

```powershell
.\.venv\Scripts\activate
pip install notebook ipykernel
jupyter notebook
```

Öncelikle `notebooks/preprocessing.ipynb` dosyasını çalıştırarak MFCC özelliklerini oluşturun. Daha sonra denemek istediğiniz model notebook'unu açıp `Run All` seçeneğini kullanın.

Temel CNN akışı için sıralama:

```text
notebooks/preprocessing.ipynb
notebooks/cnn_model.ipynb
notebooks/load_model_cnn.ipynb
```

## Git LFS Notu

Veri seti ve `dataset/features_3.0_sec.json` dosyası boyutları nedeniyle Git LFS ile takip edilmektedir. Projeyi GitHub üzerinden indirecek bilgisayarda Git LFS kurulu olmalıdır:

```powershell
git lfs install
git clone -b muhammed-dev https://github.com/Mkaankucuk/python_proje_muz-k_turu.git
```
