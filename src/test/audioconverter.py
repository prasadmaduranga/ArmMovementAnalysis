import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])

# read a audio filelock

# import speech_recognition as sr
#
# # Initialize recognizer class (for recognizing the speech)
