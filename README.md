🏗️ AI-Driven Pipeline Integrity & Risk Analyzer

Project Overview

This project provides a multimodal approach to offshore asset integrity. It combines Computer Vision to identify physical corrosion on pipelines and Natural Language Processing (NLP) to analyze the sentiment and urgency of technician maintenance logs.

The goal is to bridge the gap between visual inspection and textual reporting, ensuring that high-risk integrity issues are prioritized for maintenance in the oil and gas sector.

🚀 Key Features
Corrosion Classification: 
A Deep Learning model (ResNet-50) that classifies pipeline images into three states: Healthy, Mild Corrosion, and Severe Structural Damage.

NLP Urgency Detection: 
A sentiment analysis pipeline that processes maintenance logs to detect "fear" or "urgency" in report text, mapping it to critical safety risks.

Interactive Dashboard: 
A live Streamlit application providing real-time diagnostics and actionable maintenance recommendations.

Enterprise DevOps: 
Implementation of Git LFS for managing large-scale machine learning model weights (.pth files).

🛠️ Tech Stack
Language: Python

Deep Learning: PyTorch, Torchvision

NLP: Hugging Face Transformers

Web Framework: Streamlit

📂 Repository Structure
Plaintext
├── app.py              # Main Streamlit application logic
├── corrosion_model.pth # Trained ResNet-50 model weights (Managed via LFS)
├── requirements.txt    # Python library dependencies
├── packages.txt        # System-level Linux dependencies
├── dataset.csv         # Metadata for model training/evaluation
└── README.md           # Project documentation

💡 Lessons Learned

Multimodal Integration: Successfully combined Computer Vision and NLP to solve a singular industrial problem from two different data angles.

Cloud Deployment: Overcame Linux-level dependency conflicts (libgl1, libglib2.0-0) to ensure a stable production environment on Streamlit Cloud.

Large File Management: Leveraged Git LFS to maintain a clean version history while hosting 90MB+ machine learning weights.

Deployment: Streamlit Cloud & Git LFS

