# 🏥 Medical Image Analysis & Report Generation (AI-Powered)

## 📌 Overview
This project is an AI-powered medical diagnostic system that integrates **Computer Vision (CV)** and **Natural Language Processing (NLP)** to analyze medical images (X-rays, MRIs, CT scans) and generate **diagnostic reports**.  

The workflow includes:
- Training CV models and exporting `.pt` weights.  
- Building a Retrieval-Augmented Generation (RAG) pipeline using **Pinecone** for medical knowledge retrieval.  
- A **Flask backend** for serving predictions and report generation.  
- A **React frontend** for doctors/patients to interact with the system.  

---

## 🚀 Features
- **YOLOv8 & CNN-based models** for disease detection.  
- **NLP-powered diagnostic report generation** with RAG.  
- **Pinecone vector database** integration for retrieval.  
- **JPG + DICOM medical imaging support.**  
- **Full-stack application**: Flask backend + React frontend.  
- **PDF report generation with recommended tests.**

---

## 📂 Project Structure
```
📦 Project Root
│── 📁 CV/                     # Model creation (training notebooks)
│   ├── model1.ipynb
│   ├── model2.ipynb
│   └── ...
│   └── (Outputs: *.pt files → move to /website/server/)
│
│── 📁 NLP/                    # NLP & RAG setup
│   ├── RAG_Pinecodeloading_Sanjeevani.ipynb          # Run this after Pinecone setup
│
│── 📁 website/
│   ├── 📁 client/              # React frontend
│   │   ├── package.json
│   │   └── src/
│   │
│   └── 📁 server/              # Flask backend
│       ├── app.py
│       ├── requirements.txt
│       └── *.pt (Model weights go here)
│
│── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DataGurus/Sanjeevani_AI.git
cd Sanjeevani_AI
```

---

### 2️⃣ Computer Vision Models (Training)
1. Navigate to `CV/` folder.  
2. Run each `.ipynb` file (Jupyter/Colab).  
3. Collect the generated `.pt` files.  
4. Move them into `/website/server/`.

---

### 3️⃣ NLP & Pinecone Setup
1. Navigate to `NLP/`.  
2. Extract `datasets.zip`.  
3. Setup your **Pinecone API key** and environment variables.  
4. Run `RAG_Pinecodeloading_Sanjeevani.ipynb` to build embeddings and upload them to Pinecone.  

✅ Now your **medical RAG database** is ready.

---

### 4️⃣ Backend Setup (Flask)
```bash
cd website/server
pip install -r requirements.txt
python app.py
```

---

### 5️⃣ Frontend Setup (React)
```bash
cd website/client
npm install
npm start
```

---

## 🖥️ Usage Flow
1. **Upload medical image** (`.jpg` / `.dcm`).  
2. **AI CV models** analyze and classify diseases.  
3. **NLP RAG system** generates diagnostic reports + test recommendations.  
4. Download final **PDF report**.  

---

## 🔮 Future Enhancements
- Add support for more diseases (multi-label classification).  
- Multi-language reports.  
- Integration with **FHIR/HL7 medical standards**.  
- Cloud deployment (AWS/GCP/Azure).  

---

## 👨‍💻 Contributors
- [Prasanna Patwardhan](https://github.com/prasannapp100)
- [Yash Kulkarni](https://github.com/YashKulkarni7996)
- [Piyush Deshmukh](https://github.com/PiyushDeshmukh-apperentice)
- [Rahul Dewani](https://github.com/Rahul-Dewani)
- [Yugandhar Chawale](https://github.com/yugandhar)

---

## 📬 Contact
For queries, reach out at:  
📩 **team.datagurus@gmail.com**
