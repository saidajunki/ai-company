from flask import Flask, jsonify, request
import os
from flask import send_from_directory, redirect
from generate_rap import generate_rap

app = Flask(__name__)

@app.route('/api/generate_rap', methods=['GET', 'POST'])
def api_generate_rap():
    try:
        lyric = generate_rap()
        return jsonify({"lyric": lyric})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 開発用: 全IP・5000番portで起動
    app.run(host="0.0.0.0", port=5000, debug=True)

@app.route('/')
def root():
    return redirect('/static/index.html', code=302)
