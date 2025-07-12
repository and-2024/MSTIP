# 🤖 NLP Chatbot using Flask

A simple and interactive **rule-based chatbot** powered by **Natural Language Processing (NLP)** and **TensorFlow**. This chatbot can understand basic intents like greetings, farewells, and help queries, and respond meaningfully. It also includes a clean web interface using Flask. Great for learning how to build intelligent bots using Python!

---

## 🌟 Demo

> 💬 Type messages like:  
> `"Hello"` → _"Hey! How can I help you?"_  
> `"Bye"` → _"Goodbye! Take care."_  
> `"Thanks"` → _"You're welcome!"_

Launch it locally or run from your terminal. Web and CLI both supported ✅

---

## 🛠️ Features

- 🤖 Rule-based chatbot with intent classification
- 🧠 Trained using TensorFlow & NLP preprocessing
- 🌐 Flask-powered web interface
- 💻 Terminal-based chat mode (CLI)
- 🪄 Beginner-friendly and customizable

---

## 🚀 Getting Started

Follow these simple steps to run the chatbot locally:

### 🔁 Clone the Repository

```bash
git clone https://github.com/and-2024/MSTIP.git
cd MSTIP


⚙️ Set Up Virtual Environment (Windows)

py -3.10 -m venv chatbot-env
chatbot-env\Scripts\activate
📦 Install Dependencies

pip install -r requirements.txt
🏋️‍♀️ Train the Chatbot Model (Optional)

python train_chatbot.py
🧪 Test via Command Line

python chat.py
🌐 Run Flask Web App

python app.py
🖥️ Open in your browser: http://127.0.0.1:5000

📂 Project Structure

MSTIP/
├── chatbot-env/            # Virtual environment (excluded)
├── templates/              # HTML templates for Flask
├── static/                 # CSS or JS files (if any)
├── intents.json            # Predefined intents & responses
├── train_chatbot.py        # Model training script
├── app.py                  # Flask web app
├── chat.py                 # Terminal chatbot
├── requirements.txt        # Python packages
└── README.md               # Project overview (this file)
