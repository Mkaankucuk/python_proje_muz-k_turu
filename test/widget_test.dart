import 'package:flutter_test/flutter_test.dart';
import 'package:music_genre_detector/main.dart';

void main() {
  testWidgets('shows detector screen', (WidgetTester tester) async {
    await tester.pumpWidget(const MusicGenreApp());
    await tester.pump();

    expect(find.text('Şarkının türünü bul'), findsOneWidget);
    expect(find.text('Dosya seç'), findsOneWidget);
  });
}
