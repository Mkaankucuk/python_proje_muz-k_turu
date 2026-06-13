import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:music_genre_detector/services/audio_feature_extractor.dart';
import 'package:music_genre_detector/services/genre_classifier.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('loads trained TFLite model and returns genre predictions', () async {
    final classifier = GenreClassifier();

    final loaded = await classifier.load();
    expect(loaded, isTrue);

    final segment = List.generate(
      AudioFeatureExtractor.mfccFrameCount,
      (_) => List.generate(
        AudioFeatureExtractor.mfccCount,
        (_) => List.filled(AudioFeatureExtractor.featureChannelCount, 0.0),
      ),
    );

    final predictions = await classifier.predict([segment]);

    expect(predictions, hasLength(5));
    expect(
      predictions.map((prediction) => prediction.label),
      everyElement(
        isIn([
          'blues',
          'classical',
          'country',
          'disco',
          'hiphop',
          'jazz',
          'metal',
          'pop',
          'reggae',
          'rock',
        ]),
      ),
    );
    expect(
      predictions.map((prediction) => prediction.confidence),
      everyElement(allOf(isNonNegative, isA<double>())),
    );
    expect(
      predictions.map((prediction) => prediction.confidence),
      everyElement(
        predicate<double>((value) => !value.isNaN && value.isFinite),
      ),
    );
    expect(
      predictions.first.confidence,
      greaterThanOrEqualTo(predictions.last.confidence),
    );

    classifier.close();
  });

  test(
    'extracts engineered audio features and runs the trained model',
    () async {
      final extractor = AudioFeatureExtractor();
      final classifier = GenreClassifier();

      final loaded = await classifier.load();
      expect(loaded, isTrue);

      final analysis = await extractor.extract(_sineWaveWavBytes());
      expect(analysis.segments, isNotEmpty);
      expect(
        analysis.segments.first,
        hasLength(AudioFeatureExtractor.mfccFrameCount),
      );
      expect(
        analysis.segments.first.first,
        hasLength(AudioFeatureExtractor.mfccCount),
      );
      expect(
        analysis.segments.first.first.first,
        hasLength(AudioFeatureExtractor.featureChannelCount),
      );

      final predictions = await classifier.predict(analysis.segments);
      expect(predictions, hasLength(5));
      expect(predictions.first.confidence, isA<double>());

      classifier.close();
    },
  );
}

Uint8List _sineWaveWavBytes() {
  const sampleRate = AudioFeatureExtractor.targetSampleRate;
  const durationSeconds = 3;
  const sampleCount = sampleRate * durationSeconds;
  const headerSize = 44;
  final byteCount = sampleCount * 2;
  final bytes = Uint8List(headerSize + byteCount);
  final data = ByteData.sublistView(bytes);

  _writeAscii(bytes, 0, 'RIFF');
  data.setUint32(4, 36 + byteCount, Endian.little);
  _writeAscii(bytes, 8, 'WAVE');
  _writeAscii(bytes, 12, 'fmt ');
  data.setUint32(16, 16, Endian.little);
  data.setUint16(20, 1, Endian.little);
  data.setUint16(22, 1, Endian.little);
  data.setUint32(24, sampleRate, Endian.little);
  data.setUint32(28, sampleRate * 2, Endian.little);
  data.setUint16(32, 2, Endian.little);
  data.setUint16(34, 16, Endian.little);
  _writeAscii(bytes, 36, 'data');
  data.setUint32(40, byteCount, Endian.little);

  for (var i = 0; i < sampleCount; i++) {
    final sample = (sin(2 * pi * 440 * i / sampleRate) * 12000).round();
    data.setInt16(headerSize + i * 2, sample, Endian.little);
  }
  return bytes;
}

void _writeAscii(Uint8List bytes, int offset, String value) {
  for (var i = 0; i < value.length; i++) {
    bytes[offset + i] = value.codeUnitAt(i);
  }
}
