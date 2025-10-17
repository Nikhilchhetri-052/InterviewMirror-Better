import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 150)  
engine.setProperty("volume", 1.0) 

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

text = "Hello! This is a text to speech test in Python."
engine.say(text)

engine.runAndWait()
