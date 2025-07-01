from flask import Flask, render_template, request, jsonify
import subprocess
import csv
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save-path', methods=['POST'])
def save_path():
    try:
        data = request.get_json()
        color = data.get('color')
        if not color:
            return jsonify({'status': 'error', 'message': '색상이 지정되지 않았습니다'}), 400

        # 📍 리팩토링된 코드 실행
        subprocess.run(['python3', 'detect_and_export_refactored.py', '--color', color], check=True)

        # 📍 생성된 JSON 파일 읽어서 응답
        json_path = f'data/{color}.json'
        if not os.path.exists(json_path):
            return jsonify({'status': 'error', 'message': 'JSON 파일 없음'}), 500

        with open(json_path) as f:
            data = json.load(f)

        return jsonify({'status': 'ok', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/get-path/<color>', methods=['GET'])
def get_path_by_color(color):
    try:
        csv_path = f'data/{color}.csv'
        if not os.path.exists(csv_path):
            return jsonify({'status': 'error', 'message': '경로 없음'}), 404

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]

        # 이미지도 있다고 가정하고 반환
        return jsonify({'status': 'ok', 'data': data, 'image': f'/static/{color}.jpg'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
    
@app.route('/get-hold-image/<color>', methods=['GET'])
def get_hold_image(color):
    try:
        image_path = f'static/{color}.jpg'
        if not os.path.exists(image_path):
            subprocess.run(['python3', 'extract_holds_by_color.py', '--color', color], check=True)

        if not os.path.exists(image_path):
            return jsonify({'status': 'error', 'message': '이미지를 생성하지 못했습니다'}), 500

        return jsonify({'status': 'ok', 'image_url': f'/static/{color}.jpg'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
