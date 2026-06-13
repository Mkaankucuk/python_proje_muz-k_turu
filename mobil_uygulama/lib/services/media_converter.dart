import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class MediaConverter {
  static const _channel = MethodChannel('music_genre_detector/audio_converter');

  static const supportedExtensions = [
    'wav',
    'mp3',
    'mp4',
    'm4a',
    'aac',
    'flac',
  ];

  Future<File> toWav(String inputPath) async {
    final extension = inputPath.split('.').last.toLowerCase();
    if (extension == 'wav') {
      return File(inputPath);
    }

    if (defaultTargetPlatform == TargetPlatform.iOS) {
      final outputPath = await _channel.invokeMethod<String>('convertToWav', {
        'path': inputPath,
      });
      if (outputPath == null || outputPath.isEmpty) {
        throw const FormatException('Ses dosyası WAV formatına çevrilemedi.');
      }
      return File(outputPath);
    }

    throw const FormatException(
      'Bu platformda yalnızca WAV dosyaları destekleniyor.',
    );
  }
}
