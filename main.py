import os
from datetime import datetime
import speech_recognition as sr
import pyttsx3

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Setup log folder and file
log_folder = "conversation_logs"
os.makedirs(log_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_folder, f"conversation_{timestamp}.txt")

# Setup voice engine
engine = pyttsx3.init()

# LangChain setup
template = """
Answer is below:

Here is the conversation history:
{context}

Question: {question}
Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Voice input function
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙 Speak now...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print("You (via voice):", query)
        return query
    except sr.UnknownValueError:
        print(" Could not understand audio.")
        return ""
    except sr.RequestError:
        print(" Could not request results; check your internet.")
        return ""

# Chat handler
def handle_conversation():
    context = ""
    print("You can type or enter 'talk' to speak. Type 'exit' to end.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            
            break
        elif user_input.lower() == 'talk':
            user_input = get_voice_input()
            if not user_input:
                continue  # Skip empty audio

        try:
            print("....Analysing....")
            result = chain.invoke({
                "context": context,
                "question": user_input
            })
            print("Bot:", result)

            # Speak the answer
            engine.say(result)
            engine.runAndWait()

            # Save
            context += f"\nUser: {user_input}\nBot: {result}"
            with open(log_file_path, "a", encoding="utf-8") as file:
                file.write(f"User: {user_input}\nBot: {result}\n\n")

        except Exception as e:
            print(" Error occurred:", e)

if __name__ == "__main__":
    handle_conversation()
