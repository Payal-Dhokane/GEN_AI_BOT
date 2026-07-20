from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown



from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

## Function to load OpenAI model and get respones
model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat(history=[])


def get_gemini_response(question):

    response = chat.send_message(question, stream=True)
    return response


##initialize our streamlit app

st.set_page_config(page_title="GEMINI CHATBOT DEMO")

st.header("Gemini Application")

input = st.text_input("Input: ", key="input")


submit = st.button("Ask the question")

## If ask button is clicked

if submit:

    response = get_gemini_response(input)
    st.subheader("The Response is")
    for chunk in response:
        print(st.write(chunk.text))
        print("_" * 80)

    st.write(chat.history)
    