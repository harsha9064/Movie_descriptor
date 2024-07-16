from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import shutil
from forms import UploadForm
from models import generate_video_caption
from utils import convert_caption_to_audio, combine_audio_video

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'static/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        # Clear the uploads and processed folders
        clear_folder(app.config['UPLOAD_FOLDER'])
        clear_folder(app.config['PROCESSED_FOLDER'])

        file = form.video.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Generate video caption
        caption = generate_video_caption(filepath)

        # Convert caption to audio
        audio_filepath = convert_caption_to_audio(caption)

        # Combine audio with video
        final_video_path = combine_audio_video(filepath, audio_filepath)

        return redirect(url_for('display_video', video_path=final_video_path))
    return render_template('upload.html', form=form)


@app.route('/display', methods=['GET'])
def display_video():
    video_path = request.args.get('video_path', None)
    if video_path:
        return render_template('display.html', video_path=video_path)
    return redirect(url_for('upload_file'))


@app.route('/clear', methods=['POST'])
def clear_and_redirect():
    clear_folder(app.config['UPLOAD_FOLDER'])
    clear_folder(app.config['PROCESSED_FOLDER'])
    return redirect(url_for('upload_file'))


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


if __name__ == '__main__':
    app.run(debug=True)
