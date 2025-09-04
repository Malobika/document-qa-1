import streamlit as st
from openai import OpenAI
import PyPDF2
from io import BytesIO

# ----------------- Shared Logic -----------------
def document_qa(page_name: str):
    st.title(f"📄 {page_name} – Document Q&A")

    st.write(
        "Upload a document and ask a question – GPT will answer! "
        "You need to provide an OpenAI API key from "
        "[here](https://platform.openai.com/account/api-keys)."
    )

    # API key input
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    api_key_valid = False
    client = None

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    else:
        try:
            client = OpenAI(api_key=openai_api_key)
            # simple validation
            client.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            api_key_valid = True
            st.success("API key is valid!")
        except Exception as e:
            st.error(f"Invalid API key or API error: {str(e)}")

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

        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Is this course hard?",
            disabled=not upload_file,
        )

        if upload_file and question:
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

            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document}\n\n---\n\n {question}",
                }
            ]

            try:
                for model in ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"]:
                    st.subheader(model)
                    stream = client.chat.completions.create(
                        model=model, messages=messages, stream=True
                    )
                    st.write_stream(stream)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# ----------------- Page Functions -----------------
def lab1():
    document_qa("Lab 1")

def lab2():
    document_qa("Lab 2")


# ----------------- Navigation -----------------
pg = st.navigation(
    {
        "Labs": [
            st.Page(lab2, title="Lab 2"),
            st.Page(lab1, title="Lab 1"),
        ]
    }
)

pg.run()
