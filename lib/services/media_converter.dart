import 'dart:io';

import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';
import 'package:path_provider/path_provider.dart';

class MediaConverter {
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

    final tempDir = await getTemporaryDirectory();
    final outputPath =
        '${tempDir.path}/genre_input_${DateTime.now().microsecondsSinceEpoch}.wav';
    final command = [
      '-y',
      '-i',
      _quote(inputPath),
      '-vn',
      '-ac',
      '1',
      '-ar',
      '22500',
      '-sample_fmt',
      's16',
      _quote(outputPath),
    ].join(' ');

    final session = await FFmpegKit.execute(command);
    final returnCode = await session.getReturnCode();
    if (!ReturnCode.isSuccess(returnCode)) {
      final output = await session.getOutput();
      throw FormatException(
        'Dosya WAV formatına dönüştürülemedi. ${output ?? ''}'.trim(),
      );
    }

    return File(outputPath);
  }

  String _quote(String value) {
    return "'${value.replaceAll("'", "'\\''")}'";
  }
}
