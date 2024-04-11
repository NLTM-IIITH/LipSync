from flask import Flask, request, flash, render_template, jsonify, url_for, redirect
from tempfile import TemporaryDirectory
from subprocess import run
import os
import re
import time
from shutil import copy
from werkzeug import secure_filename
import subprocess
import random, string
import time

template = "python fa_batch_inference.py --checkpoint_path checkpoints/gen_258k_0.03_sl_0.1_disc.pth --face {} --audio {} --basename {}"
app = Flask(__name__)
app.secret_key = 'r@Im@Mdi|kOekZ)9Rh$]bJJ[[U)9er>Sdd(sea&mDk~QT(S8N}.[2wgnu~/24Pp'
app.config["APPLICATION_ROOT"] = '/lipsync/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

URL = 'http://bhaasha.iiit.ac.in'
PORT = 9001

# @app.route('/')
# def tts():
#     return render_template('demo_custom.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save("examples/"+secure_filename(f.filename))

      f1 = request.files['file1']
      f1.save("examples/"+secure_filename(f1.filename))
      random_string = ''.join([random.choice(string.ascii_letters) for _ in range(5)])
      basename = f.filename.split(".")[0] + "-synced-"+random_string
      command = template.format("examples/"+f.filename, "examples/"+f1.filename, basename)
      subprocess.call(command, shell=True)
      final_result_filename = "static/"+basename+".mp4"
      url = str(URL)+":"+str(PORT)+"/"+final_result_filename
      data = {"url": url}
      data = jsonify(data)
      return data






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
