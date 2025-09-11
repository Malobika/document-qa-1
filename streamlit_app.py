import streamlit as st
from openai import OpenAI
import PyPDF2
from io import BytesIO
import os
from dotenv import load_dotenv

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

def lab1():
    document_qa("Lab 1")

def lab2():
    document_qa("Lab 2")

def lab3():
    document_qa("Lab 3")

pg = st.navigation(
    {
        "Labs": [
            st.Page(lab2, title="Lab 2"),
            st.Page(lab1, title="Lab 1"),
            st.Page(lab3,title="Lab 3")
        ]
    },
    
)

pg.run()


