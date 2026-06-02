from pathlib import Path
import os
import shutil
import uuid

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from prediction_service import MODEL_OPTIONS, predict_audio_file


PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "music-genre-classification-local")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def allowed_audio_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def download_youtube_audio(url: str, work_dir: Path) -> Path:
    import imageio_ffmpeg
    import yt_dlp

    output_template = str(work_dir / "youtube_audio.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.extract_info(url, download=True)

    wav_files = list(work_dir.glob("*.wav"))
    if not wav_files:
        raise RuntimeError("YouTube sesi WAV formatına dönüştürülemedi.")
    return wav_files[0]


@app.get("/")
def index():
    return render_template("index.html", models=MODEL_OPTIONS)


@app.post("/predict")
def predict():
    selected_model = request.form.get("model_key", "")
    source_type = request.form.get("source_type", "file")

    if selected_model not in MODEL_OPTIONS:
        flash("Geçerli bir model seçin.", "error")
        return redirect(url_for("index"))

    request_dir = UPLOAD_DIR / uuid.uuid4().hex
    request_dir.mkdir(parents=True)

    try:
        if source_type == "youtube":
            youtube_url = request.form.get("youtube_url", "").strip()
            if not youtube_url:
                raise ValueError("YouTube bağlantısı girin.")
            audio_path = download_youtube_audio(youtube_url, request_dir)
            source_label = youtube_url
        else:
            uploaded_file = request.files.get("audio_file")
            if uploaded_file is None or not uploaded_file.filename:
                raise ValueError("Bir ses dosyası seçin.")
            if not allowed_audio_file(uploaded_file.filename):
                raise ValueError("Desteklenen bir ses dosyası yükleyin.")

            filename = secure_filename(uploaded_file.filename)
            audio_path = request_dir / filename
            uploaded_file.save(audio_path)
            source_label = uploaded_file.filename

        result = predict_audio_file(audio_path, selected_model)
        return render_template(
            "index.html",
            models=MODEL_OPTIONS,
            result=result,
            selected_model=selected_model,
            selected_source=source_type,
            source_label=source_label,
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


@app.errorhandler(413)
def upload_too_large(_error):
    flash("Dosya boyutu 100 MB sınırını aşıyor.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
