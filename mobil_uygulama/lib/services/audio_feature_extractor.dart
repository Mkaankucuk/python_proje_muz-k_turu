import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';

class AudioFeatureAnalysis {
  const AudioFeatureAnalysis({required this.segments, required this.summary});

  final List<List<List<List<double>>>> segments;
  final AudioFeatureSummary summary;
}

class AudioFeatureSummary {
  const AudioFeatureSummary({
    required this.sampleRate,
    required this.durationSeconds,
    required this.rms,
    required this.zeroCrossingRate,
    required this.spectralCentroid,
    required this.spectralBandwidth,
    required this.spectralRolloff,
    required this.segmentCount,
  });

  final int sampleRate;
  final double durationSeconds;
  final double rms;
  final double zeroCrossingRate;
  final double spectralCentroid;
  final double spectralBandwidth;
  final double spectralRolloff;
  final int segmentCount;
}

class AudioFeatureExtractor {
  static const int targetSampleRate = 22500;
  static const int mfccFrameCount = 132;
  static const int mfccCount = 13;
  static const int featureChannelCount = 3;
  static const String _normalizationPath =
      'assets/models/feature_engineering_normalization.json';

  static const int _trackDurationSeconds = 30;
  static const double _segmentDurationSeconds = 3.0;
  static const int _nFft = 2048;
  static const int _hopLength = 512;
  static const int _melBandCount = 40;

  Future<AudioFeatureAnalysis> extract(Uint8List wavBytes) async {
    final wav = _WavDecoder.decode(wavBytes);
    var samples = wav.samples;
    if (wav.sampleRate != targetSampleRate) {
      samples = _resampleLinear(samples, wav.sampleRate, targetSampleRate);
    }
    final maxSamples = targetSampleRate * _trackDurationSeconds;
    if (samples.length > maxSamples) {
      samples = samples.sublist(0, maxSamples);
    }
    if (samples.isEmpty) {
      throw const FormatException('Ses dosyası okunamadı.');
    }

    final segmentLength = (targetSampleRate * _segmentDurationSeconds).round();
    final segmentCount = max(1, (samples.length / segmentLength).ceil());
    final melFilters = _buildMelFilters();
    final dct = _buildDctMatrix();
    final normalization = await _loadNormalization();

    final segments = <List<List<List<double>>>>[];
    for (var segmentIndex = 0; segmentIndex < segmentCount; segmentIndex++) {
      final start = segmentIndex * segmentLength;
      if (start >= samples.length) break;
      final segment = List<double>.filled(segmentLength, 0);
      final end = min(start + segmentLength, samples.length);
      for (var i = start; i < end; i++) {
        segment[i - start] = samples[i];
      }
      final mfcc = _extractMfcc(segment, melFilters, dct);
      segments.add(_applyFeatureEngineering(mfcc, normalization));
    }
    if (segments.isEmpty) {
      throw const FormatException('Tahmin için geçerli ses segmenti yok.');
    }

    final summary = _summarize(samples, targetSampleRate, segments.length);
    return AudioFeatureAnalysis(segments: segments, summary: summary);
  }

  static List<List<double>> _extractMfcc(
    List<double> samples,
    List<List<double>> melFilters,
    List<List<double>> dct,
  ) {
    final window = List<double>.generate(
      _nFft,
      (i) => 0.5 - 0.5 * cos(2 * pi * i / (_nFft - 1)),
    );
    final frames = <List<double>>[];

    for (var start = 0; start < samples.length; start += _hopLength) {
      final frame = List<double>.filled(_nFft, 0);
      for (var i = 0; i < _nFft; i++) {
        final sampleIndex = start + i;
        if (sampleIndex < samples.length) {
          frame[i] = samples[sampleIndex] * window[i];
        }
      }

      final powerSpectrum = _powerSpectrum(frame);
      final melEnergies = <double>[];
      for (final filter in melFilters) {
        var energy = 0.0;
        for (var i = 0; i < filter.length; i++) {
          energy += powerSpectrum[i] * filter[i];
        }
        melEnergies.add(log(max(energy, 1e-10)));
      }

      final coeffs = List<double>.filled(mfccCount, 0);
      for (var coeff = 0; coeff < mfccCount; coeff++) {
        var value = 0.0;
        for (var mel = 0; mel < _melBandCount; mel++) {
          value += dct[coeff][mel] * melEnergies[mel];
        }
        coeffs[coeff] = value;
      }
      frames.add(coeffs);
      if (frames.length == mfccFrameCount) break;
    }

    while (frames.length < mfccFrameCount) {
      frames.add(List<double>.filled(mfccCount, 0));
    }

    return frames.take(mfccFrameCount).toList();
  }

  static List<List<List<double>>> _applyFeatureEngineering(
    List<List<double>> mfcc,
    _FeatureNormalization normalization,
  ) {
    final delta = _delta(mfcc);
    final delta2 = _delta(delta);

    return [
      for (var frame = 0; frame < mfccFrameCount; frame++)
        [
          for (var coeff = 0; coeff < mfccCount; coeff++)
            [
              (mfcc[frame][coeff] - normalization.mean[coeff][0]) /
                  normalization.std[coeff][0],
              (delta[frame][coeff] - normalization.mean[coeff][1]) /
                  normalization.std[coeff][1],
              (delta2[frame][coeff] - normalization.mean[coeff][2]) /
                  normalization.std[coeff][2],
            ],
        ],
    ];
  }

  static List<List<double>> _delta(List<List<double>> values) {
    const radius = 4;
    const denominator = 60.0;
    return List<List<double>>.generate(values.length, (frame) {
      return List<double>.generate(mfccCount, (coeff) {
        var sum = 0.0;
        for (var offset = 1; offset <= radius; offset++) {
          final previous = max(0, frame - offset);
          final next = min(values.length - 1, frame + offset);
          sum += offset * (values[next][coeff] - values[previous][coeff]);
        }
        return sum / denominator;
      });
    });
  }

  static Future<_FeatureNormalization> _loadNormalization() async {
    final text = await rootBundle.loadString(_normalizationPath);
    final data = jsonDecode(text) as Map<String, dynamic>;
    return _FeatureNormalization(
      mean: _parseNormalizationMatrix(data['mean'] as List<dynamic>),
      std: _parseNormalizationMatrix(data['std'] as List<dynamic>),
    );
  }

  static List<List<double>> _parseNormalizationMatrix(List<dynamic> rows) {
    return [
      for (final row in rows)
        [for (final value in row as List<dynamic>) (value as num).toDouble()],
    ];
  }

  static List<double> _powerSpectrum(List<double> frame) {
    final real = List<double>.from(frame);
    final imag = List<double>.filled(frame.length, 0);
    _fft(real, imag);
    final bins = frame.length ~/ 2 + 1;
    return List<double>.generate(
      bins,
      (i) => (real[i] * real[i] + imag[i] * imag[i]) / frame.length,
    );
  }

  static List<List<double>> _buildMelFilters() {
    final fftBins = _nFft ~/ 2 + 1;
    final minMel = _hzToMel(0);
    final maxMel = _hzToMel(targetSampleRate / 2);
    final melPoints = List<double>.generate(
      _melBandCount + 2,
      (i) => minMel + (maxMel - minMel) * i / (_melBandCount + 1),
    );
    final hzPoints = melPoints.map(_melToHz).toList();
    final bins = hzPoints
        .map((hz) => ((fftBins - 1) * hz / (targetSampleRate / 2)).floor())
        .toList();

    return List<List<double>>.generate(_melBandCount, (filterIndex) {
      final filter = List<double>.filled(fftBins, 0);
      final left = bins[filterIndex];
      final center = bins[filterIndex + 1];
      final right = bins[filterIndex + 2];

      for (var i = left; i < center; i++) {
        if (center != left && i >= 0 && i < fftBins) {
          filter[i] = (i - left) / (center - left);
        }
      }
      for (var i = center; i < right; i++) {
        if (right != center && i >= 0 && i < fftBins) {
          filter[i] = (right - i) / (right - center);
        }
      }
      return filter;
    });
  }

  static List<List<double>> _buildDctMatrix() {
    return List<List<double>>.generate(mfccCount, (k) {
      final scale = k == 0 ? sqrt(1 / _melBandCount) : sqrt(2 / _melBandCount);
      return List<double>.generate(
        _melBandCount,
        (n) => scale * cos(pi * k * (2 * n + 1) / (2 * _melBandCount)),
      );
    });
  }

  static AudioFeatureSummary _summarize(
    List<double> samples,
    int sampleRate,
    int segmentCount,
  ) {
    final rms = sqrt(
      samples.fold<double>(0, (sum, value) => sum + value * value) /
          samples.length,
    );
    final zcr = _zeroCrossingRate(samples);
    final frameLength = min(_nFft, samples.length);
    final frame = List<double>.filled(_nFft, 0);
    for (var i = 0; i < frameLength; i++) {
      frame[i] = samples[i];
    }
    final spectrum = _powerSpectrum(frame).sublist(1);
    final centroid = _spectralCentroid(spectrum, sampleRate);
    final bandwidth = _spectralBandwidth(spectrum, sampleRate, centroid);
    final rolloff = _spectralRolloff(spectrum, sampleRate);

    return AudioFeatureSummary(
      sampleRate: sampleRate,
      durationSeconds: samples.length / sampleRate,
      rms: rms,
      zeroCrossingRate: zcr,
      spectralCentroid: centroid,
      spectralBandwidth: bandwidth,
      spectralRolloff: rolloff,
      segmentCount: segmentCount,
    );
  }

  static List<double> _resampleLinear(
    List<double> samples,
    int sourceRate,
    int targetRate,
  ) {
    if (samples.isEmpty || sourceRate == targetRate) return samples;
    final outputLength = max(
      1,
      (samples.length * targetRate / sourceRate).round(),
    );
    return List<double>.generate(outputLength, (i) {
      final sourceIndex = i * sourceRate / targetRate;
      final left = sourceIndex.floor();
      final right = min(left + 1, samples.length - 1);
      final fraction = sourceIndex - left;
      return samples[left] * (1 - fraction) + samples[right] * fraction;
    });
  }

  static double _hzToMel(num hz) => 2595 * log(1 + hz / 700) / ln10;

  static double _melToHz(num mel) => 700 * (pow(10, mel / 2595) - 1);

  static void _fft(List<double> real, List<double> imag) {
    final n = real.length;
    var j = 0;
    for (var i = 1; i < n; i++) {
      var bit = n >> 1;
      while ((j & bit) != 0) {
        j ^= bit;
        bit >>= 1;
      }
      j ^= bit;
      if (i < j) {
        final tr = real[i];
        final ti = imag[i];
        real[i] = real[j];
        imag[i] = imag[j];
        real[j] = tr;
        imag[j] = ti;
      }
    }

    for (var len = 2; len <= n; len <<= 1) {
      final angle = -2 * pi / len;
      final wLenR = cos(angle);
      final wLenI = sin(angle);
      for (var i = 0; i < n; i += len) {
        var wr = 1.0;
        var wi = 0.0;
        for (var k = 0; k < len ~/ 2; k++) {
          final uR = real[i + k];
          final uI = imag[i + k];
          final vR = real[i + k + len ~/ 2] * wr - imag[i + k + len ~/ 2] * wi;
          final vI = real[i + k + len ~/ 2] * wi + imag[i + k + len ~/ 2] * wr;
          real[i + k] = uR + vR;
          imag[i + k] = uI + vI;
          real[i + k + len ~/ 2] = uR - vR;
          imag[i + k + len ~/ 2] = uI - vI;
          final nextWr = wr * wLenR - wi * wLenI;
          wi = wr * wLenI + wi * wLenR;
          wr = nextWr;
        }
      }
    }
  }

  static double _zeroCrossingRate(List<double> samples) {
    var crossings = 0;
    for (var i = 1; i < samples.length; i++) {
      if ((samples[i - 1] >= 0 && samples[i] < 0) ||
          (samples[i - 1] < 0 && samples[i] >= 0)) {
        crossings++;
      }
    }
    return crossings / max(1, samples.length - 1);
  }

  static double _spectralCentroid(List<double> spectrum, int sampleRate) {
    var weighted = 0.0;
    var total = 0.0;
    for (var i = 0; i < spectrum.length; i++) {
      final frequency = (i + 1) * sampleRate / (2 * spectrum.length);
      weighted += frequency * spectrum[i];
      total += spectrum[i];
    }
    return total <= 1e-12 ? 0 : weighted / total;
  }

  static double _spectralBandwidth(
    List<double> spectrum,
    int sampleRate,
    double centroid,
  ) {
    var weighted = 0.0;
    var total = 0.0;
    for (var i = 0; i < spectrum.length; i++) {
      final frequency = (i + 1) * sampleRate / (2 * spectrum.length);
      weighted += pow(frequency - centroid, 2) * spectrum[i];
      total += spectrum[i];
    }
    return total <= 1e-12 ? 0 : sqrt(weighted / total);
  }

  static double _spectralRolloff(List<double> spectrum, int sampleRate) {
    final total = spectrum.fold<double>(0, (sum, value) => sum + value);
    final threshold = total * 0.85;
    var cumulative = 0.0;
    for (var i = 0; i < spectrum.length; i++) {
      cumulative += spectrum[i];
      if (cumulative >= threshold) {
        return (i + 1) * sampleRate / (2 * spectrum.length);
      }
    }
    return sampleRate / 2;
  }
}

class _FeatureNormalization {
  const _FeatureNormalization({required this.mean, required this.std});

  final List<List<double>> mean;
  final List<List<double>> std;
}

class _DecodedWav {
  const _DecodedWav({required this.sampleRate, required this.samples});

  final int sampleRate;
  final List<double> samples;
}

class _WavDecoder {
  static _DecodedWav decode(Uint8List bytes) {
    final data = ByteData.sublistView(bytes);
    if (_fourCc(bytes, 0) != 'RIFF' || _fourCc(bytes, 8) != 'WAVE') {
      throw const FormatException('Yalnızca PCM WAV dosyaları desteklenir.');
    }

    var offset = 12;
    int? audioFormat;
    int? channels;
    int? sampleRate;
    int? bitsPerSample;
    int? dataOffset;
    int? dataSize;

    while (offset + 8 <= bytes.length) {
      final chunkId = _fourCc(bytes, offset);
      final chunkSize = data.getUint32(offset + 4, Endian.little);
      final chunkStart = offset + 8;
      if (chunkId == 'fmt ') {
        audioFormat = data.getUint16(chunkStart, Endian.little);
        channels = data.getUint16(chunkStart + 2, Endian.little);
        sampleRate = data.getUint32(chunkStart + 4, Endian.little);
        bitsPerSample = data.getUint16(chunkStart + 14, Endian.little);
      } else if (chunkId == 'data') {
        dataOffset = chunkStart;
        dataSize = chunkSize;
      }
      offset = chunkStart + chunkSize + (chunkSize.isOdd ? 1 : 0);
    }

    if (audioFormat == null ||
        channels == null ||
        sampleRate == null ||
        bitsPerSample == null ||
        dataOffset == null ||
        dataSize == null) {
      throw const FormatException('WAV başlık bilgisi eksik.');
    }
    if (audioFormat != 1 && audioFormat != 3) {
      throw const FormatException('Sadece PCM veya float WAV desteklenir.');
    }

    final bytesPerSample = bitsPerSample ~/ 8;
    final frameCount = dataSize ~/ (bytesPerSample * channels);
    final samples = <double>[];

    for (var frame = 0; frame < frameCount; frame++) {
      var mono = 0.0;
      for (var ch = 0; ch < channels; ch++) {
        final sampleOffset =
            dataOffset + (frame * channels + ch) * bytesPerSample;
        mono += _readSample(data, sampleOffset, bitsPerSample, audioFormat);
      }
      samples.add(mono / channels);
    }

    return _DecodedWav(sampleRate: sampleRate, samples: samples);
  }

  static double _readSample(
    ByteData data,
    int offset,
    int bitsPerSample,
    int audioFormat,
  ) {
    if (audioFormat == 3 && bitsPerSample == 32) {
      return data.getFloat32(offset, Endian.little).clamp(-1.0, 1.0);
    }
    switch (bitsPerSample) {
      case 8:
        return (data.getUint8(offset) - 128) / 128;
      case 16:
        return data.getInt16(offset, Endian.little) / 32768;
      case 24:
        final b0 = data.getUint8(offset);
        final b1 = data.getUint8(offset + 1);
        final b2 = data.getUint8(offset + 2);
        var value = b0 | (b1 << 8) | (b2 << 16);
        if (value & 0x800000 != 0) value |= 0xFF000000;
        return value.toSigned(32) / 8388608;
      case 32:
        return data.getInt32(offset, Endian.little) / 2147483648;
      default:
        throw FormatException('$bitsPerSample bit WAV desteklenmiyor.');
    }
  }

  static String _fourCc(Uint8List bytes, int offset) {
    return String.fromCharCodes(bytes.sublist(offset, offset + 4));
  }
}
