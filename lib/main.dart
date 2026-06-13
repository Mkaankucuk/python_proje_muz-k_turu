import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import 'models/genre_prediction.dart';
import 'services/audio_feature_extractor.dart';
import 'services/genre_classifier.dart';
import 'services/media_converter.dart';

void main() {
  runApp(const MusicGenreApp());
}

class MusicGenreApp extends StatelessWidget {
  const MusicGenreApp({super.key});

  @override
  Widget build(BuildContext context) {
    const ink = Color(0xFF14201F);
    const teal = Color(0xFF1F7A70);
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Müzik Türü Tespiti',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: teal,
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF6F8F7),
        textTheme: Theme.of(
          context,
        ).textTheme.apply(bodyColor: ink, displayColor: ink),
        useMaterial3: true,
      ),
      home: const GenreDetectorScreen(),
    );
  }
}

class GenreDetectorScreen extends StatefulWidget {
  const GenreDetectorScreen({super.key});

  @override
  State<GenreDetectorScreen> createState() => _GenreDetectorScreenState();
}

class _GenreDetectorScreenState extends State<GenreDetectorScreen> {
  final _classifier = GenreClassifier();
  final _extractor = AudioFeatureExtractor();
  final _converter = MediaConverter();

  String? _fileName;
  String? _status;
  bool _busy = false;
  bool _modelReady = false;
  List<GenrePrediction> _predictions = const [];
  AudioFeatureSummary? _summary;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    final ready = await _classifier.load();
    if (!mounted) return;
    setState(() {
      _modelReady = ready;
      _status = ready
          ? 'Model hazır. Bir dosya seçerek analize başlayabilirsin.'
          : 'Model dosyası bekleniyor: önce GTZAN eğitimini çalıştır.';
    });
  }

  Future<void> _pickAndAnalyze() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: MediaConverter.supportedExtensions,
      allowMultiple: false,
    );
    final path = result?.files.single.path;
    if (path == null) return;

    setState(() {
      _busy = true;
      _fileName = result!.files.single.name;
      _status = 'Dosya hazırlanıyor...';
      _predictions = const [];
      _summary = null;
    });

    try {
      final wavFile = await _converter.toWav(path);
      if (mounted) {
        setState(() => _status = 'Ses özellikleri çıkarılıyor...');
      }
      final bytes = await wavFile.readAsBytes();
      final analysis = await _extractor.extract(bytes);
      final predictions = _modelReady
          ? await _classifier.predict(analysis.segments)
          : <GenrePrediction>[];

      if (!mounted) return;
      setState(() {
        _summary = analysis.summary;
        _predictions = predictions;
        _status = _modelReady
            ? 'Analiz tamamlandı.'
            : 'Özellikler hazır. Tahmin için modeli eğitmen gerekiyor.';
      });
    } on Object catch (error) {
      if (!mounted) return;
      setState(() {
        _status = 'Bu dosya analiz edilemedi. $error';
      });
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  void dispose() {
    _classifier.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final topPrediction = _predictions.isEmpty ? null : _predictions.first;
    return Scaffold(
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final isWide = constraints.maxWidth >= 860;
            return ListView(
              padding: EdgeInsets.fromLTRB(
                isWide ? 32 : 16,
                18,
                isWide ? 32 : 16,
                28,
              ),
              children: [
                Center(
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 1060),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        _HeaderBar(modelReady: _modelReady),
                        const SizedBox(height: 18),
                        if (isWide)
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                flex: 6,
                                child: _AnalyzerPanel(
                                  busy: _busy,
                                  fileName: _fileName,
                                  status: _status,
                                  topPrediction: topPrediction,
                                  onPickFile: _pickAndAnalyze,
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                flex: 5,
                                child: _SideStack(
                                  predictions: _predictions,
                                  summary: _summary,
                                  modelReady: _modelReady,
                                ),
                              ),
                            ],
                          )
                        else ...[
                          _AnalyzerPanel(
                            busy: _busy,
                            fileName: _fileName,
                            status: _status,
                            topPrediction: topPrediction,
                            onPickFile: _pickAndAnalyze,
                          ),
                          const SizedBox(height: 14),
                          _SideStack(
                            predictions: _predictions,
                            summary: _summary,
                            modelReady: _modelReady,
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _HeaderBar extends StatelessWidget {
  const _HeaderBar({required this.modelReady});

  final bool modelReady;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: const Color(0xFF18302D),
            borderRadius: BorderRadius.circular(8),
          ),
          child: const Icon(Icons.graphic_eq, color: Color(0xFFF2C14E)),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Müzik Türü Tespiti',
                style: Theme.of(
                  context,
                ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w900),
              ),
              const SizedBox(height: 2),
              Text(
                'Feature-engineered DenseNet modeli',
                style: Theme.of(
                  context,
                ).textTheme.bodySmall?.copyWith(color: const Color(0xFF65716F)),
              ),
            ],
          ),
        ),
        _ModelBadge(modelReady: modelReady),
      ],
    );
  }
}

class _ModelBadge extends StatelessWidget {
  const _ModelBadge({required this.modelReady});

  final bool modelReady;

  @override
  Widget build(BuildContext context) {
    final color = modelReady
        ? const Color(0xFF1C7C54)
        : const Color(0xFFC47B18);
    return Tooltip(
      message: modelReady ? 'Model yüklendi' : 'Model bekleniyor',
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.11),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: color.withValues(alpha: 0.26)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              modelReady ? Icons.check_circle : Icons.info_outline,
              color: color,
              size: 18,
            ),
            const SizedBox(width: 6),
            Text(
              modelReady ? 'Hazır' : 'Model yok',
              style: TextStyle(color: color, fontWeight: FontWeight.w800),
            ),
          ],
        ),
      ),
    );
  }
}

class _AnalyzerPanel extends StatelessWidget {
  const _AnalyzerPanel({
    required this.busy,
    required this.fileName,
    required this.status,
    required this.topPrediction,
    required this.onPickFile,
  });

  final bool busy;
  final String? fileName;
  final String? status;
  final GenrePrediction? topPrediction;
  final VoidCallback onPickFile;

  @override
  Widget build(BuildContext context) {
    final top = topPrediction;
    return _Surface(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _UploadTarget(busy: busy, fileName: fileName, onPickFile: onPickFile),
          const SizedBox(height: 16),
          _ResultPanel(prediction: top, busy: busy),
          if (status != null) ...[
            const SizedBox(height: 14),
            _InlineStatus(text: status!, busy: busy),
          ],
        ],
      ),
    );
  }
}

class _UploadTarget extends StatelessWidget {
  const _UploadTarget({
    required this.busy,
    required this.fileName,
    required this.onPickFile,
  });

  final bool busy;
  final String? fileName;
  final VoidCallback onPickFile;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFEAF4F1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFCFE3DE)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(
                  Icons.library_music,
                  color: Color(0xFF1F7A70),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Şarkının türünü bul',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      fileName ?? 'MP3, MP4, M4A, WAV, AAC veya FLAC seç.',
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: const Color(0xFF53615E),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          FilledButton.icon(
            onPressed: busy ? null : onPickFile,
            icon: busy
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.upload_file),
            label: Text(busy ? 'Analiz ediliyor' : 'Dosya seç'),
            style: FilledButton.styleFrom(
              minimumSize: const Size(double.infinity, 52),
              backgroundColor: const Color(0xFF1F7A70),
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          ),
          const SizedBox(height: 12),
          const _FormatRow(),
        ],
      ),
    );
  }
}

class _FormatRow extends StatelessWidget {
  const _FormatRow();

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 7,
      runSpacing: 7,
      children: [
        for (final extension in MediaConverter.supportedExtensions)
          _FormatPill(extension: extension),
      ],
    );
  }
}

class _FormatPill extends StatelessWidget {
  const _FormatPill({required this.extension});

  final String extension;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFD9E6E2)),
      ),
      child: Text(
        extension.toUpperCase(),
        style: const TextStyle(
          color: Color(0xFF1E5F59),
          fontSize: 12,
          fontWeight: FontWeight.w900,
        ),
      ),
    );
  }
}

class _ResultPanel extends StatelessWidget {
  const _ResultPanel({required this.prediction, required this.busy});

  final GenrePrediction? prediction;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    final confidence = ((prediction?.confidence ?? 0) * 100).clamp(0, 100);
    final hasResult = prediction != null;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF172326),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          SizedBox.square(
            dimension: 62,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CircularProgressIndicator(
                  value: hasResult ? confidence / 100 : null,
                  strokeWidth: 7,
                  backgroundColor: Colors.white.withValues(alpha: 0.12),
                  color: const Color(0xFFF2C14E),
                ),
                Center(
                  child: Icon(
                    hasResult ? Icons.auto_graph : Icons.equalizer,
                    color: Colors.white,
                    size: 24,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  hasResult
                      ? prediction!.label.toUpperCase()
                      : busy
                      ? 'ANALİZ'
                      : 'SONUÇ BEKLENİYOR',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  hasResult
                      ? 'Güven: ${confidence.toStringAsFixed(1)}%'
                      : 'Ses dosyası seçildiğinde model tahmini burada görünür.',
                  style: const TextStyle(color: Color(0xFFD1DEDA)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _InlineStatus extends StatelessWidget {
  const _InlineStatus({required this.text, required this.busy});

  final String text;
  final bool busy;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFF7F0DD),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFEADDAF)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (busy)
            const Padding(
              padding: EdgeInsets.only(top: 2),
              child: SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            )
          else
            const Icon(Icons.info, color: Color(0xFFC47B18), size: 18),
          const SizedBox(width: 10),
          Expanded(child: Text(text)),
        ],
      ),
    );
  }
}

class _SideStack extends StatelessWidget {
  const _SideStack({
    required this.predictions,
    required this.summary,
    required this.modelReady,
  });

  final List<GenrePrediction> predictions;
  final AudioFeatureSummary? summary;
  final bool modelReady;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (predictions.isNotEmpty)
          _PredictionPanel(predictions: predictions)
        else
          _EmptyPanel(modelReady: modelReady),
        if (summary != null) ...[
          const SizedBox(height: 14),
          _FeaturePanel(summary: summary!),
        ],
        if (!modelReady) ...[
          const SizedBox(height: 14),
          const _ModelHelpPanel(),
        ],
      ],
    );
  }
}

class _EmptyPanel extends StatelessWidget {
  const _EmptyPanel({required this.modelReady});

  final bool modelReady;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      title: 'Tahminler',
      icon: Icons.insights,
      child: Text(
        modelReady
            ? 'Bir ses dosyası seçildiğinde en yakın türler burada listelenir.'
            : 'Model yüklendiğinde tahmin listesi aktif olur.',
        style: Theme.of(
          context,
        ).textTheme.bodyMedium?.copyWith(color: const Color(0xFF5F6D6A)),
      ),
    );
  }
}

class _PredictionPanel extends StatelessWidget {
  const _PredictionPanel({required this.predictions});

  final List<GenrePrediction> predictions;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      title: 'En Yakın Türler',
      icon: Icons.insights,
      child: Column(
        children: [
          for (var index = 0; index < predictions.length; index++)
            _PredictionRow(prediction: predictions[index], rank: index + 1),
        ],
      ),
    );
  }
}

class _PredictionRow extends StatelessWidget {
  const _PredictionRow({required this.prediction, required this.rank});

  final GenrePrediction prediction;
  final int rank;

  @override
  Widget build(BuildContext context) {
    final percent = (prediction.confidence * 100).clamp(0, 100);
    final color = rank == 1
        ? const Color(0xFF1F7A70)
        : rank == 2
        ? const Color(0xFF6B6FAE)
        : const Color(0xFFC16C46);
    return Padding(
      padding: EdgeInsets.only(bottom: rank == 5 ? 0 : 12),
      child: Row(
        children: [
          Container(
            width: 30,
            height: 30,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              '$rank',
              style: TextStyle(color: color, fontWeight: FontWeight.w900),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        prediction.label,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontWeight: FontWeight.w800),
                      ),
                    ),
                    Text(
                      '${percent.toStringAsFixed(1)}%',
                      style: const TextStyle(fontWeight: FontWeight.w800),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: LinearProgressIndicator(
                    value: prediction.confidence.clamp(0, 1),
                    minHeight: 9,
                    color: color,
                    backgroundColor: const Color(0xFFE4ECE9),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _FeaturePanel extends StatelessWidget {
  const _FeaturePanel({required this.summary});

  final AudioFeatureSummary summary;

  @override
  Widget build(BuildContext context) {
    final values = {
      'Süre': '${summary.durationSeconds.toStringAsFixed(1)} sn',
      'Örnekleme': '${summary.sampleRate} Hz',
      'Segment': '${summary.segmentCount}',
      'RMS': summary.rms.toStringAsFixed(4),
      'ZCR': summary.zeroCrossingRate.toStringAsFixed(4),
      'Centroid': '${summary.spectralCentroid.toStringAsFixed(0)} Hz',
      'Rolloff': '${summary.spectralRolloff.toStringAsFixed(0)} Hz',
    };

    return _Panel(
      title: 'Ses Özeti',
      icon: Icons.tune,
      child: Wrap(
        spacing: 9,
        runSpacing: 9,
        children: [
          for (final entry in values.entries)
            _MetricChip(label: entry.key, value: entry.value),
        ],
      ),
    );
  }
}

class _ModelHelpPanel extends StatelessWidget {
  const _ModelHelpPanel();

  @override
  Widget build(BuildContext context) {
    return _Panel(
      title: 'Model Gerekli',
      icon: Icons.school,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Tahmin almak için bu projedeki eğitilmiş Keras modelini TFLite formatına çevir.',
          ),
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFF172326),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const SelectableText(
              'python3 tools/convert_project_model_to_tflite.py --model /Users/bengisudemir/Documents/GitHub/python_proje_muz-k_turu/models/feature_engineered_densenet.keras',
              style: TextStyle(
                color: Colors.white,
                fontFamily: 'monospace',
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Panel extends StatelessWidget {
  const _Panel({required this.title, required this.icon, required this.child});

  final String title;
  final IconData icon;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return _Surface(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: const Color(0xFF1F7A70), size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  title,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          child,
        ],
      ),
    );
  }
}

class _Surface extends StatelessWidget {
  const _Surface({required this.child, required this.padding});

  final Widget child;
  final EdgeInsetsGeometry padding;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFE0E8E5)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: child,
    );
  }
}

class _MetricChip extends StatelessWidget {
  const _MetricChip({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: const BoxConstraints(minWidth: 112),
      padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFF4F6F5),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFE2E8E6)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: Theme.of(context).textTheme.labelSmall?.copyWith(
              color: const Color(0xFF63716E),
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(
              context,
            ).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w900),
          ),
        ],
      ),
    );
  }
}
