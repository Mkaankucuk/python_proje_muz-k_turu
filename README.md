# Müzik Türü Tespit Uygulaması

Bu Flutter uygulaması, seçilen bir müzik dosyasını analiz ederek müzik türünü tahmin eder.

Uygulama içinde eğitilmiş TensorFlow Lite modeli hazır olarak gelir. Kullanıcı bir ses dosyası seçer, uygulama ses özelliklerini çıkarır ve şu türlerden en yakın olanları gösterir:

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

## Kimler İçin?

Bu README, projeyi daha önce hiç çalıştırmamış biri için hazırlanmıştır. Aşağıdaki adımları sırayla izleyerek uygulamayı bilgisayarına indirebilir ve çalıştırabilirsin.

## Gerekenler

Bilgisayarında şunlar kurulu olmalı:

1. Git
2. Flutter SDK
3. Android Studio veya Xcode
4. Bir Android emülatörü, iOS simülatörü ya da bağlı telefon

Mac kullanıyorsan iOS için ayrıca Xcode kurulu olmalıdır.

Flutter kurulumunu kontrol etmek için terminalde şunu çalıştır:

```bash
flutter doctor
```

`flutter doctor` eksik bir şey gösterirse önce onu düzelt.

## Projeyi İndirme

Terminali aç ve projeyi indirmek istediğin klasöre git. Örneğin:

```bash
cd ~/Desktop
```

Projeyi GitHub’dan indir:

```bash
git clone https://github.com/bengisudemirdev/muziktespituygulamassi.git
```

Proje klasörüne gir:

```bash
cd muziktespituygulamassi
```

## Bağımlılıkları Kurma

Flutter paketlerini indir:

```bash
flutter pub get
```

iOS için çalıştıracaksan pod dosyalarını da kur:

```bash
cd ios
pod install
cd ..
```

## Uygulamayı Çalıştırma

Bağlı cihazları görmek için:

```bash
flutter devices
```

Uygulamayı çalıştırmak için:

```bash
flutter run
```

Birden fazla cihaz görünüyorsa belirli cihaz seçerek çalıştırabilirsin:

```bash
flutter run -d CİHAZ_ID
```

`CİHAZ_ID` değerini `flutter devices` çıktısından alabilirsin.

## iOS İçin Önemli Not

Xcode ile açacaksan şu dosyayı aç:

```text
ios/Runner.xcworkspace
```

Şunu açma:

```text
ios/Runner.xcodeproj
```

`Runner.xcodeproj` açılırsa `Pods_Runner.framework not found` gibi hatalar alabilirsin.

## Uygulama Nasıl Kullanılır?

1. Uygulamayı aç.
2. `Dosya seç` butonuna bas.
3. Bir müzik dosyası seç.
4. Uygulama dosyayı analiz eder.
5. En yakın müzik türlerini ve güven oranlarını gösterir.

Desteklenen dosya türleri:

- WAV
- MP3
- MP4
- M4A
- AAC
- FLAC

## Model Dosyaları

Uygulamanın kullandığı model dosyaları bu klasördedir:

```text
assets/models/
```

Önemli dosyalar:

```text
assets/models/genre_classifier.tflite
assets/models/labels.json
assets/models/feature_engineering_normalization.json
```

Bu dosyalar uygulamanın müzik türü tahmini yapabilmesi için gereklidir.

## Modeli Yeniden Dönüştürmek İstersen

Normal kullanım için bunu yapman gerekmez. Model zaten uygulamaya eklenmiştir.

Eğer eğitim projesindeki Keras modelinden tekrar TensorFlow Lite dosyası üretmek istersen:

```bash
python3 -m venv .tf-env
source .tf-env/bin/activate
pip install tensorflow numpy
python3 tools/convert_project_model_to_tflite.py --model /path/to/feature_engineered_densenet.keras
```

Bu komut şu dosyayı günceller:

```text
assets/models/genre_classifier.tflite
```

## Test Etme

Kodun çalıştığını kontrol etmek için:

```bash
flutter analyze
flutter test
```

iOS build kontrolü için:

```bash
flutter build ios --debug --no-codesign
```

Android build kontrolü için:

```bash
flutter build apk
```

## Sık Karşılaşılan Sorunlar

### `flutter: command not found`

Flutter kurulu değildir veya PATH ayarına eklenmemiştir. Flutter SDK’yı kurup terminali yeniden aç.

### `Pods_Runner.framework not found`

Xcode’da yanlış dosya açılmış olabilir. `ios/Runner.xcworkspace` dosyasını aç.

### iOS simülatörde `ffmpeg_kit_flutter_new` uyarısı

Apple Silicon ve bazı iOS simülatörlerinde bu paket uyarı verebilir. Gerçek iPhone cihazında çalıştırmak daha sağlıklı olabilir.

### Uygulama dosya seçiyor ama tahmin yapmıyor

Şu dosyaların var olduğundan emin ol:

```text
assets/models/genre_classifier.tflite
assets/models/labels.json
assets/models/feature_engineering_normalization.json
```

## Proje Yapısı

```text
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
test/
tools/
ios/
android/
```

## Kısa Özet

Projeyi çalıştırmak için en kısa yol:

```bash
git clone https://github.com/bengisudemirdev/muziktespituygulamassi.git
cd muziktespituygulamassi
flutter pub get
flutter run
```
