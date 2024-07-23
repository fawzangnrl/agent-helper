import streamlit as st
from openai import OpenAI
import dotenv
import os
import PyPDF2
import random
import google.generativeai as genai
import anthropic

dotenv.load_dotenv()

# Define models
anthropic_models = ["claude-3-5-sonnet-20240620"]
google_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]
# Function to convert the messages format from OpenAI and Streamlit to Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))
            elif content["type"] == "pdf_file":
                gemini_message["parts"].append(genai.upload_file(content["pdf_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages


# Function to convert the messages format from OpenAI and Streamlit to Anthropic (the only difference is in the image messages)
def messages_to_anthropic(messages):
    anthropic_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            anthropic_message = anthropic_messages[-1]
        else:
            anthropic_message = {
                "role": message["role"] ,
                "content": [],
            }
        if message["content"][0]["type"] == "image_url":
            anthropic_message["content"].append(
                {
                    "type": "image",
                    "source":{   
                        "type": "base64",
                        "media_type": message["content"][0]["image_url"]["url"].split(";")[0].split(":")[1],
                        "data": message["content"][0]["image_url"]["url"].split(",")[1]
                        # f"data:{img_type};base64,{img}"
                    }
                }
            )
        else:
            anthropic_message["content"].append(message["content"][0])

        if prev_role != message["role"]:
            anthropic_messages.append(anthropic_message)

        prev_role = message["role"]
        
    return anthropic_messages
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""

    if model_type == "openai":
        client = OpenAI(api_key=api_key)
        for chunk in client.chat.completions.create(
            model=model_params["model"] if "model" in model_params else "gpt-4o",
            messages=st.session_state.messages,
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
            stream=True,
        ):
            chunk_text = chunk.choices[0].delta.content or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name = model_params["model"],
            generation_config={
                "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
            }
        )
        gemini_messages = messages_to_gemini(st.session_state.messages)

        for chunk in model.generate_content(
            contents=gemini_messages,
            stream=True,
        ):
            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model=model_params["model"] if "model" in model_params else "claude-3-5-sonnet-20240620",
            messages=messages_to_anthropic(st.session_state.messages),
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
        ) as stream:
            for text in stream.text_stream:
                response_message += text
                yield text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})

# Main function to run the Streamlit app
def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="The OmniChat",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Side Bar ---
    with st.sidebar:
        cols_keys = st.columns(2)
        with cols_keys[0]:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
        with cols_keys[1]:
            google_api_key = st.text_input("Google API Key", type="password")
        anthropic_api_key = st.text_input("Anthropic API Key", type="password")

    # --- Main Content ---
    if (openai_api_key == "" and google_api_key == "" and anthropic_api_key == ""):
        st.warning("Please introduce an API Key to continue...")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])

        # File upload for PDF
        pdf_uploaded = st.file_uploader("Upload a PDF file:", type=["pdf"], key="uploaded_file")

        pdf_text = ""
        if pdf_uploaded is not None:
            pdf_path = f"uploaded_{random.randint(100000, 999999)}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_uploaded.read())
            
            # Extract text from the PDF
            pdf_text = extract_text_from_pdf(pdf_path)
            # st.session_state.messages.append(
            #     {
            #         "role": "user",
            #         "content": [{
            #             "type": "text",
            #             "text": pdf_text,
            #         }]
            #     }
            # )
            st.success("PDF uploaded and text extracted successfully!")

    # Chat input
    if prompt := st.chat_input("Hi! Ask me anything based on the PDF..."):
        # Periksa apakah ada teks PDF yang diekstrak
        if not pdf_text.strip():
            st.warning("Silakan unggah PDF terlebih dahulu untuk mengajukan pertanyaan.")
        else:
            full_prompt = "PENTING!! : Jika pertanyaan User tidak sesuai dengan teks dibawah ini, tolong jawab 'Maaf, saya tidak memiliki informasi tersebut'" + pdf_text + "\n\nUser: Berdasarkan tesk tersebut, " + prompt

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": full_prompt,
                    }]
                }
            )

            # Display the new messages
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Memilih model berdasarkan kunci API yang digunakan
                response_text = ""
                if openai_api_key:
                    response = stream_llm_response(
                        model_params={"model": "gpt-4o", "temperature": 0.3},  # Contoh parameter
                        model_type="openai",
                        api_key=openai_api_key
                    )
                elif google_api_key:
                    response = stream_llm_response(
                        model_params={"model": "gemini-1.5-flash", "temperature": 0.3},  # Contoh parameter
                        model_type="google",
                        api_key=google_api_key
                    )
                elif anthropic_api_key:
                    response = stream_llm_response(
                        model_params={"model": "claude-3-5-sonnet-20240620", "temperature": 0.3},  # Contoh parameter
                        model_type="anthropic",
                        api_key=anthropic_api_key
                    )
                else:
                    st.warning("Tidak ada API Key yang valid.")
                    return

                # Mengumpulkan hasil dari generator
                for chunk in response:
                    response_text += chunk

                st.markdown(response_text)

if __name__ == "__main__":
    main()
