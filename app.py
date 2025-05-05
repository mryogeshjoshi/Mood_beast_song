from flask import Flask, render_template, request, url_for
import joblib
import random
import os

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("mood_model.pkl")
le = joblib.load("label_encoder.pkl")

# Predefined song database
song_db = {
    'Happy': [
        {'name': 'Crush', 'img': 'happy1.jpg', 'file': 'happy1.mp3'},
        {'name': 'Dekha tenu pahli pahli bar ve', 'img': 'happy2.jpg', 'file': 'happy2.mp3'},
        {'name': 'My Queen', 'img': 'happy3.jpg', 'file': 'happy3.mp3'},
        {'name': 'Nazar', 'img': 'happy4.jpg', 'file': 'happy4.mp3'},
    ],
    'Sad': [
        {'name': 'Agar tum sath ho', 'img': 'sad1.jpg', 'file': 'sad1.mp3'},
        {'name': 'Sanam Teri Kasam 2', 'img': 'sad2.jpg', 'file': 'sad2.mp3'},
        {'name': 'Hamari Adhuri kahani', 'img': 'sad3.jpg', 'file': 'sad3.mp3'},
        {'name': 'Roi na je Yaad Meri Aayi ve', 'img': 'sad4.jpg', 'file': 'sad4.mp3'}
    ],
    'Motivational': [
        {'name': 'Razzi', 'img': 'mot1.jpg', 'file': 'mot1.mp3'},
        {'name': 'Badal pe Paon Hain', 'img': 'mot2.jpg', 'file': 'mot2.mp3'},
        {'name': 'Salaam India', 'img': 'mot3.jpg', 'file': 'mot3.mp3'},
        {'name': 'Ilahi', 'img': 'mot4.jpg', 'file': 'mot4.mp3'}      
    ],
    'Party': [
        {'name': 'Oh HO HO HO', 'img': 'party1.jpg', 'file': 'party1.mp3'},
        {'name': 'Party All Night', 'img': 'party2.jpg', 'file': 'party2.mp3'},
        {'name': 'Sauda Khara Khara', 'img': 'party3.jpg', 'file': 'party3.mp3'},
        {'name': 'Soni de Nakhre', 'img': 'party4.jpg', 'file': 'party4.mp3'}
    ]
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            energy = float(request.form['energy'])
            dance = float(request.form['danceability'])
            valence = float(request.form['valence'])

            prediction = model.predict([[energy, dance, valence]])[0]
            mood = le.inverse_transform([prediction])[0]  # Convert label index to label name

            songs = random.sample(song_db[mood], min(6, len(song_db[mood])))
            
            # Process songs to include proper URLs for images and audio files
            for song in songs:
                song['image_url'] = url_for('static', filename=f'images/{song["img"]}')
                song['audio_url'] = url_for('static', filename=f'audio/{song["file"]}')
                
            return render_template("index.html", mood=mood, songs=songs)
        except Exception as e:
            return f"Error occurred: {e}"

    return render_template("index.html", mood=None)

if __name__ == "__main__":
    app.run(debug=True)