from flask import Flask, render_template, request,Response
import object_detection

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        file.save(file.filename)
        return Response(object_detection.detect_objects(file.filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    # Command to run flask app
    app.run(host='192.168.1.12',debug=True)
