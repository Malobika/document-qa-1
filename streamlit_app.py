import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from io import BytesIO
import os
from dotenv import load_dotenv

from hw4 import run as hw4_run
import hw7 
import hw5 
import hw7_claude

import chromadb

load_dotenv()


def document_qa(page_name: str):
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    api_key_valid = False
    client = None

    st.title(f"📄 {page_name} – Document Summarizer")

    st.write(
        "Upload a document and select how you want it summarized. "
        "You can also switch between a fast, cheaper model (4o-mini) "
        "and an advanced model (4o)."
    )

    if not openai_api_key:
        st.error("No OpenAI API key found. Please add it to your .env file.", icon="🗝️")
    else:
        try:
            client = OpenAI(api_key=openai_api_key)
            # quick validation
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            api_key_valid = True
            st.success("✅ API key loaded from .env and is valid!")
        except Exception as e:
            st.error(f"❌ Invalid API key or API error: {str(e)}")

    if api_key_valid and client:
        upload_file = st.file_uploader(
            "Upload a document (.txt or .pdf)", type=("txt", "pdf")
        )

        if upload_file is None and "document_content" in st.session_state:
            del st.session_state["document_content"]
            del st.session_state["file_name"]
            st.info("File data is cleared from memory")

        if upload_file:
            st.success(f"File loaded: {upload_file.name}")
            st.session_state["file_name"] = upload_file.name
        else:
            st.info("No File uploaded")

        if upload_file:
            document = ""

            if (
                "document_content" in st.session_state
                and st.session_state.get("file_name") == upload_file.name
            ):
                document = st.session_state["document_content"]
                st.info("Using cached document")
            else:
                try:
                    file_extension = upload_file.name.split(".")[-1].lower()
                    if file_extension == "txt":
                        document = upload_file.read().decode()
                    elif file_extension == "pdf":
                        pdf_reader = PyPDF2.PdfReader(BytesIO(upload_file.read()))
                        for page in pdf_reader.pages:
                            document += (page.extract_text() or "") + "\n"
                    else:
                        st.error("Unsupported file type.")
                        st.stop()

                    st.session_state["document_content"] = document
                    st.session_state["file_name"] = upload_file.name
                    st.success("Document processed and cached.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.stop()

            if not document.strip():
                st.warning("The document appears to be empty or unreadable.")
                st.stop()

            
            st.sidebar.header("Summary Options")

            summary_type = st.sidebar.radio(
                "Choose summary format:",
                [
                    "Summarize the document in 100 words",
                    "Summarize the document in 2 connecting paragraphs",
                    "Summarize the document in 5 bullet points",
                ],
            )

            use_advanced = st.sidebar.checkbox("Use Advanced Model (4o)")
            model = "gpt-4o" if use_advanced else "gpt-4o-mini"

            
            st.subheader(f"Summary using {model}")
            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"Here is a document:\n\n{document}\n\n---\n\n{summary_type}",
                    }
                ]

                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
                st.write_stream(stream)

            except Exception as e:
                st.error(f"An error occurred while generating summary: {str(e)}")

def document_qa_lab3(page_name: str):
    st.title("My lab answering chatbot")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    api_key_valid = False
    client = None
    openai_model = st.sidebar.selectbox("which model",("mini","regular"))
    if openai_model =="mini":
        model_to_use ="gpt-4o-mini"
    else:
        model_to_use ="gpt-4o"


   
    if not openai_api_key:
        st.error("No OpenAI API key found. Please add it to your .env file.", icon="🗝️")
    else:
        try:
            client = OpenAI(api_key=openai_api_key)
            # quick validation
            client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            api_key_valid = True
            st.success("✅ API key loaded from .env and is valid!")
        except Exception as e:
            st.error(f"❌ Invalid API key or API error: {str(e)}")


    if api_key_valid and client:
        
        st.set_page_config(
            page_title= "My Title",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={ 'get Help': 'https://dbdmg.polito.it/',
                        'Report a bug': "https://dbdmg.polito.it",
                        'About': "# *Intro to datbases* course"

            }
        )

        #with st.chat_message("user"):
            #st.write("Hello ...")
        #with st.chat_message("assistant"):
            #st.write("Hello human")
        #promt =st.chat_input("say something")
        #if promt:
            #st.write(f"User has sent the promt:{promt}")
        #initialize chat history
        #if "messages" not in st.session_state:
            #st.session_state.messages =[]

        #display messages
        if "messages" not in st.session_state:
            st.session_state["messages"] =[{"role":"assistant","content":"How can I help you"}]
            #with st.chat_message(message["role"]):
                #st.markdown(message["content"])
        for message in st.session_state.messages:
            chat_msg =st.chat_message(message["role"])
            chat_msg.write(message["content"])

        

        
        #react to user input
        if promt := st.chat_input("What is up?"):
            # append latest user input
            st.session_state.messages.append({"role": "user", "content": promt})

            # keep only last 2 user messages + latest assistant message
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            if len(user_msgs) > 5:
                # find index of first user message to drop
                first_user_idx = next(i for i, m in enumerate(st.session_state.messages) if m["role"] == "user")
                # remove oldest user messages until only 2 remain
                while len([m for m in st.session_state.messages if m["role"] == "user"]) > 5:
                    st.session_state.messages.pop(first_user_idx)

            # display user message
            with st.chat_message("user"):
                st.markdown(promt)

            # stream assistant response
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "system", "content": "Always explain in very simple language so that a 10-year-old can understand."}] + st.session_state.messages,
                stream=True
            )
            with st.chat_message("assistant"):
                response = st.write_stream(stream)

            # save assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})


            # save assistant response
            more_q = "DO YOU WANT MORE INFO?"
            with st.chat_message("assistant"):
                st.write(more_q)
            st.session_state.messages.append({"role": "assistant", "content": more_q})

            # Switch mode to waiting for yes/no
            st.session_state["mode"] = "waiting_more_info"

                                


            #st.session_state.messages.append({"role":"user","content":promt})

            #response = f"Echo: {promt}"
            #with st.chat_message("assistant"):
                    #st.markdown(response)

            #st.session_state.messages.append({"role":"user","content":response})

def document_qa_lab4(page_name:str):
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    st.title("My lab answering chatbot")

    # Load API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("No OpenAI API key found. Please add it to your .env file.", icon="🗝️")
        return

    # Initialize OpenAI client once
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=openai_api_key)
    openai_client = st.session_state.openai_client

   
    
    chromaDB_path = "./Chroma_lab"
    chroma_client =chromadb.PersistentClient(chromaDB_path )
    if 'Lab4_vectorDB' not in st.session_state:
        collection = chroma_client.get_or_create_collection("Lab4Collection")
        pdf_path = "./files"

        # Get all PDF files in that folder
        pdf_files = [os.path.join(pdf_path, f) for f in os.listdir(pdf_path) if f.endswith(".pdf")]

        for pdf_file in pdf_files:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

            # ✅ Add each file separately, use just the filename
            add_to_collection(collection, text, os.path.basename(pdf_file))
            st.write(f"📄 Loaded {os.path.basename(pdf_file)} ({len(text)} characters)")

        # ✅ Save collection to session state
        st.session_state.Lab4_vectorDB = collection
        st.success("✅ Lab4 ChromaDB created and stored in session!")
    else:
        collection = st.session_state.Lab4_vectorDB

    # ---- Test query ----
    topic = st.sidebar.selectbox("Test Query", ("Generative AI", "Text Mining", "Data Science Overview"))
    response = openai_client.embeddings.create(input=topic, model="text-embedding-3-small")
    query_embedding = response.data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    st.subheader("Top 3 Retrieved Documents")
    for i, meta in enumerate(results["metadatas"][0]):
        st.write(f"{i+1}. {meta['filename']}")


def document_qa_lab5(page_num:str):
    import requests
    st.title("My Weather App")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("No OpenAI API key found. Please add it to your .env file.", icon="🗝️")
        return
    
    openweatherkey=os.getenv("OpenWeatherAPI")
    location="Syracuse, NY"
    st.write(get_current_weather(location, openweatherkey))

    st.title("👕 The What-To-Wear Weather Bot")

    city = st.text_input("Enter a city:", "Syracuse, NY")

    if st.button("Get Advice"):
        api_key = st.secrets["OPENWEATHER_API_KEY"]
        weather = get_current_weather(city, api_key)
        st.write("### 🌡 Current Weather", weather)

        advice = get_clothing_advice(weather)
        st.markdown("### 👗 Clothing & Picnic Advice")
        st.write(advice)
def get_clothing_advice(weather):
    prompt = f"""
    The current weather in {weather_data['location']} is:
    Temperature: {weather_data['temperature']}°C
    Feels like: {weather_data['feels_like']}°C
    Humidity: {weather_data['humidity']}%

    Suggest what someone should wear and say if it's a good day for a picnic.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful weather clothing advisor."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message["content"]
def get_current_weather(location:str,OpenWeatherAPI):
    import requests
    if "," in location:
        location = location.split(",")[0].strip()
    urlbase= "https://api.openweathermap.org/data/2.5/"
    urlweather= f"weather?q={location}&appid={OpenWeatherAPI}"
    url =urlbase+urlweather

    response = requests.get(url)
    data = response.json()

    # Extract temperatures & Convert Kelvin to Celsius
    temp = data['main']['temp'] - 273.15
    feels_like = data['main']['feels_like'] - 273.15
    temp_min = data['main']['temp_min'] - 273.15
    temp_max = data['main']['temp_max'] - 273.15
    humidity = data['main']['humidity']



    #Extract temperature, weather description, and other relevant data from the API response.
    return {"location":location,"temperature": round(temp,2),"feels_like": round(feels_like,2),"temp_min":round(temp_min,2),"temp_max":round(temp_max,2), "humidity": round(humidity, 2)}
def add_to_collection(collection, text, filename):
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    # ✅ store filename in metadata
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding],
        metadatas=[{"filename": filename}]
    )
def lab1():
    document_qa("Lab 1")

def lab2():
    document_qa("Lab 2")

def lab3():
    document_qa_lab3("Lab 3")
def lab4():
    document_qa_lab4("Lab 4")
def lab5():
    document_qa_lab5("Lab 5")

def hw5_run():
    hw5.run()

def hw7c_run():
    hw7_claude.run()

pg = st.navigation(
    {
        "Labs": [
            st.Page(lab2, title="Lab 2"),
            st.Page(lab1, title="Lab 1"),
            st.Page(lab3, title="Lab 3"),
            st.Page(lab4, title="Lab 4"),
            st.Page(lab5,title="Lab 5")
        ],
        "HWs": [
            st.Page(hw4_run, title="HW4"),
             st.Page(hw5_run, title="HW5"),
           
             st.Page(hw7c_run,title = "HW7_claude")
        ]
    },
)

pg.run()



