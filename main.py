import os
import base64

from flask import Flask, request, render_template


app = Flask(__name__, static_folder='static')
app.secret_key = 'asdafa'

@app.route('/', methods=['GET', 'POST'])
def get_main():
    if request.method == 'GET':
        return render_template('main.html')
    
    if request.method == 'POST':
        file = request.files['video_file']
        
        if request.form.get('noAudio') == 'on':
            print('on')
        else:
            print('no')
        
        if not os.path.isdir('./tmp/'):
            os.mkdir('./tmp/')

        file.save('./tmp/tmp_video.mp4')
        with open('./tmp/tmp_video.mp4', 'rb') as f:
            video_data = f.read()

        base64_video = base64.b64encode(video_data)
        video_str = base64_video.decode('utf-8')

        return render_template('report.html', vids=[video_str for _ in range(2)])

if __name__ == '__main__':
    app.run(host='0.0.0.0')
