import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/genre_prediction.dart';
import 'audio_feature_extractor.dart';

class GenreClassifier {
  static const _modelPath = 'assets/models/genre_classifier.tflite';
  static const _labelsPath = 'assets/models/labels.json';

  Interpreter? _interpreter;
  List<String> _labels = const [
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
  ];

  Future<bool> load() async {
    try {
      final labelText = await rootBundle.loadString(_labelsPath);
      _labels = (jsonDecode(labelText) as List<dynamic>).cast<String>();
      _interpreter = await Interpreter.fromAsset(_modelPath);
      return true;
    } on Object catch (error) {
      debugPrint('Model yüklenemedi: $error');
      _interpreter = null;
      return false;
    }
  }

  Future<List<GenrePrediction>> predict(
    List<List<List<List<double>>>> segments,
  ) async {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('Model yüklenmedi.');
    }
    if (segments.isEmpty) {
      throw ArgumentError('Tahmin için MFCC segmenti bulunamadı.');
    }
    final firstSegment = segments.first;
    if (firstSegment.length != AudioFeatureExtractor.mfccFrameCount ||
        firstSegment.first.length != AudioFeatureExtractor.mfccCount ||
        firstSegment.first.first.length !=
            AudioFeatureExtractor.featureChannelCount) {
      throw ArgumentError(
        'MFCC boyutu uyumsuz; beklenen '
        '${AudioFeatureExtractor.mfccFrameCount}x'
        '${AudioFeatureExtractor.mfccCount}x'
        '${AudioFeatureExtractor.featureChannelCount}.',
      );
    }

    final meanOutput = List<double>.filled(_labels.length, 0);
    for (final segment in segments) {
      final input = [segment];
      final output = [List<double>.filled(_labels.length, 0)];
      interpreter.run(input, output);
      for (var i = 0; i < _labels.length; i++) {
        meanOutput[i] += output.first[i];
      }
    }
    for (var i = 0; i < meanOutput.length; i++) {
      meanOutput[i] /= segments.length;
    }

    final predictions = <GenrePrediction>[
      for (var i = 0; i < _labels.length; i++)
        GenrePrediction(label: _labels[i], confidence: meanOutput[i]),
    ]..sort((a, b) => b.confidence.compareTo(a.confidence));

    return predictions.take(5).toList();
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}
