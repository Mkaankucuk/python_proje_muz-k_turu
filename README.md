# Music Genre Classification

## Başka Bir Bilgisayarda Çalıştırma

Projeyi zip dosyasından çıkardıktan sonra proje klasöründe PowerShell açın.

Python 3.12 kurulu olmalıdır. Ardından sırasıyla şu komutları çalıştırın:

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

Web uygulamasında `CNN` veya `CNN + LSTM` modeli seçilebilir. Bilgisayardan ses dosyası yüklenebilir veya YouTube şarkı bağlantısı girilebilir.

Not: `.venv` klasörünü zip dosyasına eklemeyin. Her bilgisayarda yukarıdaki komutlarla yeniden oluşturun.

## Projenin Amacı

Bu proje, GTZAN müzik türü veri setindeki ses dosyalarını derin öğrenme ile sınıflandırmak için hazırlanmıştır. Amaç eski hazır `.h5` modellerini doğrudan kullanmak değil, modelleri bu bilgisayarda yeniden eğitmek, yeni accuracy oranlarını görmek, en başarılı modeli kaydetmek ve sunuma hazır hale getirmektir.

## Kullanılan Veri Seti

Veri seti klasörü:

```text
dataset/genres_original/
```

Beklenen tür klasörleri:

```text
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
```

Bu projede `dataset/genres_original` yolu, bilgisayardaki mevcut veri seti konumuna bağlanmıştır:

```text
C:\Users\Muham\Desktop\Data (1)\genres_original
```

## Kullanılan Yöntem: MFCC + CNN

`notebooks/preprocessing.ipynb` her 30 saniyelik parçayı 10 segmente böler. Her segment yaklaşık 3 saniyedir. Her segment için 13 MFCC özelliği çıkarılır ve sonuç şu dosyaya yazılır:

```text
dataset/features_3.0_sec.json
```

`notebooks/cnn_model.ipynb` bu MFCC özelliklerini CNN modeline giriş olarak verir.

## Neden İki Model Denendi?

Eski projedeki `cnn_model.ipynb` içinde 3 aktif CNN denemesi vardı. Eski kaydedilmiş çıktılar incelendi ve aktif eğitim akışında sadece iki model bırakıldı:

1. Önceki model: Baseline CNN.
2. En başarılı model: BatchNormalization ve Dropout kullanan CNN.

Üçüncü eski deneme silinmedi; eğitim notebookunda not olarak bırakıldı. Aktif olarak yeniden eğitilen model sayısı 2'dir.

## Eski Referans Model Karşılaştırma Tablosu

Bu tablo eski notebook çıktılarından referans olarak çıkarılmıştır. Yeni oranlar, bu bilgisayarda `cnn_model.ipynb` çalıştırılınca tekrar üretilecektir.

| Model | Eski max validation accuracy | Eski test accuracy | Durum |
|---|---:|---:|---|
| CNN baseline | 0.7312 | 0.7040 | Aktif, önceki model |
| CNN BatchNorm + Dropout | 0.8456 | 0.8075 | Aktif, en başarılı model |
| Eski CNN 3 | 0.8177 | 0.8005 | Aktif eğitimden çıkarıldı |

Yeni eğitimden sonra karşılaştırma tablosu otomatik olarak burada oluşur:

```text
models/model_comparison.csv
```

## Bu Bilgisayarda Alınan Yeni Eğitim Sonuçları

Eğitim 1 Haziran 2026 tarihinde `C:\Users\Muham\Desktop\Data (1)\music-genre-classification` klasöründe yeniden çalıştırıldı. `jazz.00054.wav` dosyası okunamadığı için preprocessing sırasında atlandı ve toplam `9990` segment üretildi.

| Model | Training accuracy | Validation accuracy | Test accuracy | Loss | Validation loss | Test loss | Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Önceki model - CNN baseline | 0.9106 | 0.7005 | 0.6723 | 0.3071 | 1.0059 | 1.0449 | 57 |
| En başarılı model - CNN batchnorm/dropout | 0.7277 | 0.6855 | 0.6657 | 0.8365 | 1.0857 | 1.0901 | 100 |

Bu yeni çalıştırmada en yüksek validation accuracy değerini `Önceki model - CNN baseline` aldığı için otomatik olarak en iyi model seçildi.

## En İyi Modelin Nasıl Seçildiği

`cnn_model.ipynb` iki modeli de yeniden eğitir ve şu metrikleri hesaplar:

- training accuracy
- validation accuracy
- test accuracy
- loss
- validation loss
- test loss

En iyi model otomatik olarak en yüksek `validation_accuracy` değerine göre seçilir.

## Modelin Kaydedildiği Yer

En başarılı model iki formatta kaydedilir:

```text
models/best_model.keras
models/best_model.h5
```

`.keras` güncel Keras formatıdır. `.h5` dosyası ek uyumluluk için ayrıca kaydedilir.

## Projeyi Baştan Çalıştırma Adımları

1. Python 3.10 ortamı oluşturun ve proje klasöründe açın.

```powershell
cd "C:\Users\Muham\Desktop\Data (1)\music-genre-classification"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Terminalden baştan çalıştırmak için:

```powershell
python 01_preprocessing.py
python 02_train_cnn.py
python 03_load_model_cnn.py
```

3. Notebook ile çalıştırmak isterseniz Jupyter kurup açın.

```powershell
pip install notebook ipykernel
python -m ipykernel install --user --name music-genre-classification --display-name "Music Genre Classification"
jupyter notebook
```

4. Notebooklarda sırasıyla Run All yapın:

```text
notebooks/preprocessing.ipynb
notebooks/cnn_model.ipynb
notebooks/load_model_cnn.ipynb
```

5. `preprocessing.ipynb` veya `01_preprocessing.py` şu dosyayı üretir:

```text
dataset/features_3.0_sec.json
```

6. `cnn_model.ipynb` veya `02_train_cnn.py` iki modeli yeniden eğitir, accuracy/loss grafiklerini çizer, karşılaştırma tablosunu oluşturur ve en iyi modeli kaydeder.

7. `load_model_cnn.ipynb` veya `03_load_model_cnn.py` kaydedilen en iyi modeli yükler, test accuracy gösterir, confusion matrix çizer ve tek bir müzik dosyası için tür tahmini örneği çalıştırır.

## Flask Web Uygulaması

Model seçmeli web ekranını açmak için:

```powershell
cd "C:\Users\Muham\Desktop\Data (1)\music-genre-classification"
.\.venv\Scripts\python.exe app.py
```

Tarayıcı adresi:

```text
http://127.0.0.1:5000
```

Web ekranında `CNN` veya `CNN + LSTM` modeli seçilebilir. Ses dosyası yüklenebilir veya bir YouTube şarkı bağlantısı girilebilir. YouTube bağlantısı verildiğinde ses otomatik olarak indirilir, WAV formatına çevrilir ve seçilen model ile müzik türü tahmin edilir.

## DenseNet ve Vision Transformer Denemeleri

Ek model denemeleri şu notebook içinde çalıştırılmıştır:

```text
notebooks/densenet_vit_model.ipynb
```

Notebook içinde epoch çıktıları, training/validation/test metrikleri, accuracy-loss grafikleri, confusion matrix görselleri ve karşılaştırma tablosu bulunur.

| Model | Training accuracy | Validation accuracy | Test accuracy | Loss | Validation loss | Test loss | Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compact DenseNet | 0.7692 | 0.7598 | 0.7421 | 0.6686 | 0.7172 | 0.7417 | 20 |
| Compact Vision Transformer | 0.5399 | 0.5368 | 0.5138 | 1.2615 | 1.2612 | 1.3150 | 20 |

Modeller ayrı ayrı kaydedilmiştir:

```text
models/compact_densenet.keras
models/compact_densenet.h5
models/compact_vit.keras
models/compact_vit.h5
```

En yüksek validation accuracy değerini alan model ayrıca kaydedilmiştir:

```text
models/best_vision_model.keras
models/best_vision_model.h5
```

## DenseNet ve Vision Transformer Feature Engineering Denemesi

Feature engineering denemesi yalnızca DenseNet ve Vision Transformer modellerine uygulanmıştır:

```text
notebooks/feature_engineering_model.ipynb
```

Ham MFCC verisine ek olarak MFCC değerlerinin zamansal değişimini gösteren `delta` ve `delta-delta` özellikleri çıkarılmıştır. Üç kanal yalnızca training verisinden hesaplanan ortalama ve standart sapma değerleriyle normalize edilmiştir:

```text
MFCC + delta + delta-delta
```

| Model | Önceki validation accuracy | Yeni validation accuracy | Önceki test accuracy | Yeni test accuracy |
|---|---:|---:|---:|---:|
| Compact DenseNet | 0.7598 | 0.7748 | 0.7421 | 0.7618 |
| Compact Vision Transformer | 0.5368 | 0.5833 | 0.5138 | 0.5542 |

Feature engineered modeller ayrı ayrı kaydedilmiştir:

```text
models/feature_engineered_densenet.keras
models/feature_engineered_densenet.h5
models/feature_engineered_vit.keras
models/feature_engineered_vit.h5
```

Bu ikili içinde en yüksek validation accuracy değerini alan model ayrıca kaydedilmiştir:

```text
models/best_feature_engineered_vision_model.keras
models/best_feature_engineered_vision_model.h5
```
