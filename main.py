import os
import base64

from flask import Flask, request, render_template

from modules.utils import warmup_models
from modules.wrapper import get_videos

models = warmup_models('./models')

app = Flask(__name__, static_folder='static')
app.secret_key = 'asdafa'


@app.route('/', methods=['GET', 'POST'])
def get_main():
    if request.method == 'GET':
        return render_template('main.html')
    
    if request.method == 'POST':
        file = request.files['video_file']

        if not os.path.isdir('./tmp/'):
            os.mkdir('./tmp/')

        music_path = './tmp/music.mp3'
        if request.form.get('noAudio') != 'on':
            music = request.files['audio_file']
            music.save(music_path)

        file_path = './tmp/tmp_video.mp4'
        file.save(file_path)

        get_videos(file_path, models)
        with open('./tmp/shorts1.mp4', 'rb') as f:
            video_data = f.read()

        if os.path.exists(music_path):
            os.remove(music_path)

        base64_video = base64.b64encode(video_data)
        video_str = base64_video.decode('utf-8')

        return render_template('report.html', vids=[video_str for _ in range(2)])


if __name__ == '__main__':
    app.run(host='0.0.0.0')
