from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)


template = """
You are a language tutor. Help the user learn the target language by correcting grammar, providing translations, and engaging in simple conversations.

Here is the conversation history: {context}

Language being learned: {language}

Question: {question}

Answer:
"""


model = OllamaLLM(model="llama3")  # Ensure "llama3" is the correct model name
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Flask route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html exists in the templates folder

# Flask route for handling conversation
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        context = data.get('context', '')
        language = data.get('language', '')
        question = data.get('question', '')

        # Validate input
        if not language or not question:
            return jsonify({'error': 'Language and question are required.'}), 400

        # Invoke the model with the provided context, language, and question
        result = chain.invoke({"context": context, "language": language, "question": question})
        
        # Debugging: print the result to check its structure
        print(result)

        return jsonify({'response': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
