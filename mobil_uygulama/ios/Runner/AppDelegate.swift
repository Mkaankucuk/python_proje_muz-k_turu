import Flutter
import AVFoundation
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    let channel = FlutterMethodChannel(
      name: "music_genre_detector/audio_converter",
      binaryMessenger: engineBridge.applicationRegistrar.messenger()
    )
    channel.setMethodCallHandler { call, result in
      guard call.method == "convertToWav" else {
        result(FlutterMethodNotImplemented)
        return
      }
      guard
        let arguments = call.arguments as? [String: Any],
        let path = arguments["path"] as? String
      else {
        result(FlutterError(code: "INVALID_ARGUMENT", message: "Dosya yolu eksik.", details: nil))
        return
      }

      DispatchQueue.global(qos: .userInitiated).async {
        do {
          let outputPath = try Self.convertToWav(inputPath: path)
          DispatchQueue.main.async {
            result(outputPath)
          }
        } catch {
          DispatchQueue.main.async {
            result(FlutterError(code: "CONVERSION_FAILED", message: error.localizedDescription, details: nil))
          }
        }
      }
    }
  }

  private static func convertToWav(inputPath: String) throws -> String {
    let inputURL = URL(fileURLWithPath: inputPath)
    let inputFile = try AVAudioFile(forReading: inputURL)
    guard
      let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 22500,
        channels: 1,
        interleaved: true
      ),
      let converter = AVAudioConverter(
        from: inputFile.processingFormat,
        to: outputFormat
      )
    else {
      throw NSError(
        domain: "MusicGenreDetector",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Ses dönüştürücü başlatılamadı."]
      )
    }

    let outputURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("genre_input_\(UUID().uuidString).wav")
    let outputFile = try AVAudioFile(
      forWriting: outputURL,
      settings: outputFormat.settings,
      commonFormat: .pcmFormatInt16,
      interleaved: true
    )

    let maxDurationSeconds = 30.0
    let inputFrameLimit = min(
      inputFile.length,
      AVAudioFramePosition(inputFile.processingFormat.sampleRate * maxDurationSeconds)
    )
    guard inputFrameLimit > 0 else {
      throw NSError(
        domain: "MusicGenreDetector",
        code: 2,
        userInfo: [NSLocalizedDescriptionKey: "Ses dosyası okunamadı."]
      )
    }

    guard
      let inputBuffer = AVAudioPCMBuffer(
        pcmFormat: inputFile.processingFormat,
        frameCapacity: AVAudioFrameCount(inputFrameLimit)
      )
    else {
      throw NSError(
        domain: "MusicGenreDetector",
        code: 3,
        userInfo: [NSLocalizedDescriptionKey: "Giriş ses tamponu oluşturulamadı."]
      )
    }
    try inputFile.read(into: inputBuffer, frameCount: AVAudioFrameCount(inputFrameLimit))

    let outputFrameCapacity = AVAudioFrameCount(outputFormat.sampleRate * maxDurationSeconds) + 4096
    guard
      let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: outputFrameCapacity
      )
    else {
      throw NSError(
        domain: "MusicGenreDetector",
        code: 4,
        userInfo: [NSLocalizedDescriptionKey: "Çıkış ses tamponu oluşturulamadı."]
      )
    }

    var didProvideInput = false
    var conversionError: NSError?
    let status = converter.convert(to: outputBuffer, error: &conversionError) {
      _, outStatus in
      if didProvideInput {
        outStatus.pointee = .endOfStream
        return nil
      }
      didProvideInput = true
      outStatus.pointee = .haveData
      return inputBuffer
    }

    if let conversionError {
      throw conversionError
    }
    if status == .error || outputBuffer.frameLength == 0 {
      throw NSError(
        domain: "MusicGenreDetector",
        code: 5,
        userInfo: [NSLocalizedDescriptionKey: "Ses WAV formatına çevrilemedi."]
      )
    }
    try outputFile.write(from: outputBuffer)

    return outputURL.path
  }
}
