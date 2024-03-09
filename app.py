from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi as yta

import speech_recognition as sr

genai.configure(api_key='AIzaSyDURpJOTQXExKr1qdZ1SD7aDS4lnyBkveg')

app = Flask(__name__)
CORS(app)

chain = None  # Initialize outside of any route

def youtube_url_to_text(youtube_url):
    vid_id = youtube_url.split('watch?v=')[1]
    data=yta.get_transcript(vid_id)

    transcript = ''
    for value in data:
        for key, val in value.items():
            if key=='text':
                transcript += val
                
    l = transcript.splitlines()
    final = "".join(l)
    return final

def get_pdf_text(pdf_path):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        return str(e)  # Handle PDF processing errors

    return text

def get_chain(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyDURpJOTQXExKr1qdZ1SD7aDS4lnyBkveg')
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key='AIzaSyDURpJOTQXExKr1qdZ1SD7aDS4lnyBkveg')
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Initialize the chain variable once when the server starts

@app.route('/upload', methods=['POST'])
def upload_file():
    print('Upload file called')
    try:
        if 'file' not in request.files:
            return jsonify({"error": "File not provided"}), 400

        uploaded_file = request.files['file']

        if uploaded_file:
            # Ensure the 'uploads' directory exists
            os.makedirs('uploads', exist_ok=True)

            # Save the file to the 'uploads' directory
            uploaded_file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(uploaded_file_path)
            print('Processing text')
            # Process the PDF content
            pdf_text = get_pdf_text(uploaded_file_path)
            global chain 
            print('Processing chain')
            chain = get_chain(pdf_text)
            # No need to reassign the chain variable here

            return jsonify({"message": "File Uploaded Successfully"})
        else:
            return jsonify({"error": "File not provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/message', methods=['POST', 'OPTIONS'])
def get_message():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight request handled successfully'})
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.get_json()

    if 'message' not in data:
        return jsonify({'error': 'data not found'})

    message = data['message']

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyDURpJOTQXExKr1qdZ1SD7aDS4lnyBkveg')

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(message)

    response = chain.invoke(
        {"input_documents": docs, "question": message},
        return_only_outputs=True
    )
    
    

    return jsonify({'response': response['output_text']})


@app.route('/videourl', methods=['POST','OPTIONS'])
def video_url():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight request handled successfully'})
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.get_json()

    if 'message' not in data:
        return jsonify({'error': 'data not found'})

    message = data['message']
    
    text = youtube_url_to_text(message)
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"You are an AI assistant that will generate neat notes, based on the corpus of text given. Generate with neat subheadings, multiple newlines, and neat list points, in markdown format. The text is : {text}"

    response = model.generate_content(prompt)
    
    print(response.text)
    
    return jsonify({'response' : response.text})
    

if __name__ == "__main__":
    app.run()
    # app.run(debug=True)
