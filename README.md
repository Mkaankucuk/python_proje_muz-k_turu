# Müzik Türü Tespit Projesi

Bu repo iki parçadan oluşur:

- Python tarafı: Müzik türü modelini eğitmek veya eğitilmiş Keras modelini TensorFlow Lite formatına çevirmek için kullanılır.
- Mobil uygulama: Flutter ile geliştirilmiştir. Kullanıcıdan ses dosyası alır, model dosyasını kullanarak müzik türünü tahmin eder.

Mobil uygulama şu türleri tahmin eder:

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

## Proje Yapısı

```text
python_proje_muz-k_turu/
  README.md
  pubspec.yaml
  lib/
    main.dart
    services/
      audio_feature_extractor.dart
      genre_classifier.dart
      media_converter.dart
  assets/
    models/
      genre_classifier.tflite
      labels.json
      feature_engineering_normalization.json
  tools/
    requirements.txt
    train_gtzan_model.py
    convert_project_model_to_tflite.py
  android/
  ios/
  test/
```

## Gerekli Programlar

Projeyi tamamen çalıştırmak için bilgisayarında şunlar kurulu olmalıdır:

1. Git
2. Python 3.9 veya daha yeni bir Python sürümü
3. Flutter SDK
4. Android Studio veya Xcode
5. Android emülatörü, iOS simülatörü ya da bağlı telefon

Kurulumları kontrol etmek için:

```bash
python3 --version
flutter doctor
```

`flutter doctor` eksik bir şey gösterirse önce onu düzelt.

## Projeyi İndirme

Terminali aç ve projeyi indirmek istediğin klasöre git:

```bash
cd ~/Desktop
```

Projeyi indir:

```bash
git clone https://github.com/Mkaankucuk/python_proje_muz-k_turu.git
cd python_proje_muz-k_turu
```

Eğer ekip branch'i üzerinde çalışacaksan:

```bash
git checkout bengisu-dev
```

## 1. Mobil Uygulamayı Çalıştırma

Normal kullanım için Python tarafını çalıştırmana gerek yoktur. Hazır model dosyaları `assets/models/` klasöründe bulunur.

Flutter paketlerini indir:

```bash
flutter pub get
```

Bağlı cihazları gör:

```bash
flutter devices
```

Uygulamayı çalıştır:

```bash
flutter run
```

Birden fazla cihaz varsa cihaz ID'si ile çalıştırabilirsin:

```bash
flutter run -d CİHAZ_ID
```

## iOS İçin Ek Adım

iOS için çalıştıracaksan pod dosyalarını kur:

```bash
cd ios
pod install
cd ..
```

Xcode ile açacaksan bu dosyayı aç:

```text
ios/Runner.xcworkspace
```

Şunu açma:

```text
ios/Runner.xcodeproj
```

Yanlış dosya açılırsa `Pods_Runner.framework not found` veya benzeri linker hataları alınabilir.

## 2. Python Ortamını Hazırlama

Modeli yeniden eğitmek veya Keras modelini tekrar TFLite'a dönüştürmek istersen Python ortamını kur:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r tools/requirements.txt
```

Windows kullanıyorsan sanal ortamı şu şekilde aç:

```bash
.venv\Scripts\activate
```

## 3. Eğitilmiş Keras Modelini Mobil Uygulamaya Aktarma

Elinde daha önce eğitilmiş `.keras` uzantılı model varsa onu mobil uygulamanın kullanacağı `.tflite` formatına çevirebilirsin.

Örnek:

```bash
python3 tools/convert_project_model_to_tflite.py \
  --model /MODELİN_BULUNDUĞU_YOL/feature_engineered_densenet.keras
```

Bu komut şu dosyaları günceller:

```text
assets/models/genre_classifier.tflite
assets/models/labels.json
```

Sonra mobil uygulamayı tekrar çalıştır:

```bash
flutter pub get
flutter run
```

## 4. Modeli GTZAN Datasetiyle Yeniden Eğitme

Eğer modeli sıfırdan eğitmek istersen GTZAN veri seti şu yapıda olmalıdır:

```text
genres_original/
  blues/
    blues.00000.wav
  classical/
    classical.00000.wav
  country/
  disco/
  hiphop/
  jazz/
  metal/
  pop/
  reggae/
  rock/
```

Eğitimi başlat:

```bash
python3 tools/train_gtzan_model.py \
  --dataset /DATASETİN_BULUNDUĞU_YOL/genres_original
```

Epoch sayısını değiştirmek istersen:

```bash
python3 tools/train_gtzan_model.py \
  --dataset /DATASETİN_BULUNDUĞU_YOL/genres_original \
  --epochs 150
```

Eğitim tamamlanınca model dosyaları otomatik olarak `assets/models/` klasörüne yazılır. Sonrasında mobil uygulama yeni modeli kullanır.

## 5. Uygulamayı Test Etme

Kod kalitesini kontrol et:

```bash
flutter analyze
```

Testleri çalıştır:

```bash
flutter test
```

iOS build kontrolü:

```bash
flutter build ios --debug --no-codesign
```

Android APK üretmek:

```bash
flutter build apk
```

## Uygulama Nasıl Kullanılır?

1. Mobil uygulamayı aç.
2. `Dosya seç` butonuna bas.
3. Bir müzik dosyası seç.
4. Uygulama dosyayı analiz eder.
5. En olası müzik türlerini güven oranlarıyla birlikte gösterir.

Desteklenen dosya türleri:

- WAV
- MP3
- MP4
- M4A
- AAC
- FLAC

## Model Dosyaları

Mobil uygulamanın çalışması için bu dosyalar gereklidir:

```text
assets/models/genre_classifier.tflite
assets/models/labels.json
assets/models/feature_engineering_normalization.json
```

Bu dosyalardan biri eksikse uygulama tahmin yapamaz.

## GitHub'a Gönderirken Dikkat Edilecekler

Şu klasörler repoya eklenmemelidir:

```text
.venv/
.tf-env/
build/
.dart_tool/
ios/Pods/
ios/.symlinks/
```

Bu klasörler yerel olarak oluşur ve GitHub'a gönderilmez. Özellikle `.venv/` klasörü içinde TensorFlow gibi büyük paketler olduğu için GitHub'ın 100 MB dosya sınırına takılabilir.

Değişiklikleri göndermek için:

```bash
git status
git add .
git commit -m "Açıklayıcı commit mesajı"
git push origin bengisu-dev
```

## Sık Karşılaşılan Sorunlar

### `flutter: command not found`

Flutter kurulu değildir veya PATH ayarına eklenmemiştir. Flutter SDK'yı kurup terminali yeniden aç.

### `python3: command not found`

Python kurulu değildir veya terminal Python'u bulamıyordur. Python 3 kurup terminali yeniden aç.

### `Pods_Runner.framework not found`

Xcode'da yanlış dosya açılmış olabilir. `ios/Runner.xcworkspace` dosyasını aç.

### GitHub `File size limit exceeded` hatası veriyor

Muhtemelen `.venv/`, `build/` veya benzeri yerel klasörlerden biri git'e eklenmiştir. Önce kontrol et:

```bash
git status
```

Bu klasörleri commit'e ekleme. Eğer daha önce commit'e girdiyse git geçmişinden temizlemek gerekir.

### Uygulama dosya seçiyor ama tahmin yapmıyor

Model dosyalarının var olduğundan emin ol:

```bash
ls assets/models
```

Şu dosyaları görmelisin:

```text
genre_classifier.tflite
labels.json
feature_engineering_normalization.json
```

## En Kısa Çalıştırma Özeti

Sadece mobil uygulamayı çalıştırmak için:

```bash
git clone https://github.com/Mkaankucuk/python_proje_muz-k_turu.git
cd python_proje_muz-k_turu
git checkout bengisu-dev
flutter pub get
flutter run
```

Python ile model dönüştürmek için:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt
python3 tools/convert_project_model_to_tflite.py --model /MODEL_YOLU/model.keras
flutter run
```
