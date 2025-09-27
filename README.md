# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.


🎥 Shortloom AI

AI-powered tool for YouTube Shorts, Reels, and TikTok creators.
It helps generate Titles, Descriptions, and Hashtags automatically for uploaded videos.

📖 Table of Contents

Introduction

Problems We Solve

Features

Demo Workflow

Getting Started

Requirements

Installation

Running the Project

AI Models Used

Project Structure

Contributing

Future Scope

License

🚀 Introduction

Shortloom AI is an open-source project designed to help content creators save time.
Instead of spending hours thinking about a catchy title or searching for trending hashtags, Shortloom AI generates them automatically from the video/audio you upload.

Think of it as Canva for video metadata 🎨 – simple, fast, and AI-powered.

❌ Problems We Solve

Before Shortloom AI:

Creators struggled to come up with engaging titles.

Hashtags were random and not trending.

Descriptions were either too short or irrelevant.

Manual keyword research took a lot of time.

With Shortloom AI:

Upload video → AI listens & analyzes.

Automatically generates Title, Hashtags, and Description.

Faster publishing.

Higher reach & engagement.

⭐ Features

✅ Upload video and get AI-generated Title, Description & Hashtags.
✅ Works offline using open-source AI models (Whisper, Transformers).
✅ Simple UI for beginners.
✅ Manual option if AI fails to detect audio.
✅ 100% free & open-source.

🎬 Demo Workflow

User uploads a short video 🎥

AI listens to audio → extracts keywords, speech, or background music 🎶

Generates:

Title

Description

Hashtags (#Shorts, #Trending, #YourTopic)

User copies & uses for YouTube Shorts, Reels, TikTok 🚀

⚙️ Getting Started
Requirements

Python 3.9+

VS Code or Google Colab

Installed libraries (see below)

Installation

Clone the repo:

git clone https://github.com/your-username/shortloom-ai.git
cd shortloom-ai


Create virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


requirements.txt example:

transformers
torch
accelerate
sentencepiece
moviepy
yt-dlp

Running the Project

Run the app locally:

python app.py


or in Jupyter Notebook/Colab:

!python app.py

🧠 AI Models Used

Whisper (OpenAI, offline version via HuggingFace) → Extracts audio & speech from video.

Transformers (BART, T5, or DistilGPT2) → Summarizes into Title & Description.

Keyword Extractors (YAKE, KeyBERT) → Creates relevant Hashtags.

📂 Project Structure
shortloom-ai/
│── app.py              # Main entry point
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── /static             # CSS, JS, Icons
│── /templates          # HTML frontend
│── /models             # Downloaded AI models
│── /uploads            # User uploaded videos

🤝 Contributing

We welcome contributions ❤️

Fork the repository

Create a new branch (feature/new-idea)

Commit changes (git commit -m "Added new feature")

Push to branch

Open a Pull Request

🔮 Future Scope

Add trending hashtag detection from live sources.

Add AI-powered thumbnail generator.

Multi-language support.

Mobile app version.

📜 License (MIT)

This project uses the MIT License.

What it means:

✅ You can use this code for personal or commercial projects.

✅ You can modify the code to fit your needs.

✅ You can distribute it freely.

❌ You cannot hold the authors liable for issues or damages.

In short → MIT gives you maximum freedom to use and grow the project while protecting the original creators from legal problem
