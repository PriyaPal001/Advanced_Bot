import os
from datetime import datetime
import speech_recognition as sr
import pyttsx3

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


log_folder = "conversation_logs"
os.makedirs(log_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_folder, f"conversation_{timestamp}.txt")


engine = pyttsx3.init()


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


def handle_conversation():
    context = ""
    print("You can type or enter 'talk' to speak. Type 'exit' to end.")

    while True:
        input_mode = "text"
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'talk':
            user_input = get_voice_input()
            input_mode = "voice"
            if not user_input:
                continue  

        try:
            print("....Analysing....")
            result = chain.invoke({
                "context": context,
                "question": user_input
            })
            print("Bot:", result)

            # if input_mode == "voice":
            #     engine.say(result)
            #     engine.runAndWait()

            
            context += f"\nUser: {user_input}\nBot: {result}"
            with open(log_file_path, "a", encoding="utf-8") as file:
                file.write(f"User: {user_input}\nBot: {result}\n\n")

        except Exception as e:
            print(" Error occurred:", e)

if __name__ == "__main__":
    handle_conversation()
