# Medical Image Analysis & NLP Report Generation

<h2>📌 Overview</h1>
This project is an AI-powered medical diagnostic tool that analyzes medical images (X-rays, MRIs, CT scans) to detect diseases using deep learning models. It also generates diagnostic reports using NLP techniques. The system supports both JPG and DICOM file formats.

<h2>🚀 Features</h2>

- YOLOv8n for object detection in medical images.
- CNN-based classification for disease identification.
- DICOM file support for medical imaging.
- OpenCV preprocessing & postprocessing.
- NLP-powered report generation.
- Flask API for backend processing.
- React-based frontend for user-friendly interactions.
- PDF report generation with test recommendations.

 <h2>⚙️ Execution flow</h2>

 <h3>Execution Flow</h3>
1. Running Python files : Training and storing weights of the model.
2. Running code for generation  of Hybrid Database for NLP (Retrieval Augmented Generation) engine.
3. Loading the Med-Embed model from HuggingFace : Coverting database into embeddings efficient for retrieval.
4. Upload vector embeddings in Pinecone Vector Database.
5. Run app.py
6. Run npm : Frontend Code files


 <h2>Configuration Information</h2>

 <h3>1️⃣ Clone the Repository</h3>

```python
git clone https://github.com/DataGurus/Sanjeevani_AI.git
cd Sanjeevani_AI
```

 <h3>2️⃣ Set Up Virtual Environment (Optional but Recommended)</h3>

```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

 <h3>3️⃣ Install Backend Dependencies</h3>

```bash
pip install -r requirements.txt
```

 <h3>4️⃣ Install Frontend Dependencies</h3>
Navigate to the frontend directory:

```bash
cd frontend
npm install
```

<h2> 🛠 Installation Instructions</h2>

 <h3>🔹 Backend Setup (Flask)</h3>

1. Ensure `Python 3.8+` is installed.
2. Run the Flask server:

```python
   python app.py
```

 <h3>🔹 Frontend Setup (React)</h3>

1. Ensure `Node.js 16+` is installed.
2. Start the React development server:
```bash
   npm start
```
 <h3>🚀 Operating Instructions</h3>

1. Upload a Medical Image
   - Choose between `.jpg` or `.dcm` format.
2. AI Model Processing
   - Object detection via YOLOv8.
   - Disease classification using CNN.
3. Report Generation
   - Extracted medical insights using NLP.
   - PDF download option.

 <h2>📂 Directory Structure</h2>

```
 Directory Structure Project Root 📦 
|-- 📄 index 
|-- 📄 package.json # Frontend dependencies 
|-- 📄 package-lock.json # Dependency lock file 
|-- 📄 README.md # Project documentation 
|-- 📄 tsconfig.json # TypeScript configuration 
|-- 📄 .gitignore # Git ignore file 
|-- 📄 styles # CSS styles
|-- 📁 public # Static files for React
|   |-- impact-3.jpeg 
|   |-- impact-3.jpg 
|   |-- index.jpg 
|   |-- kidney.jpg 
|   |-- liver.jpg 
|   |-- Brain.jpg 
|   |-- Eye.jpg 
|   |-- logo.png 
|   |-- logo-name.png 
|   |-- logo192.png 
|   |-- logo512.png 
|   `-- favicon.ico 
|-- 📁 assets # Image and media assets 
`-- 📁 src # React app source code 
    |-- 📄 MainPage.css # CSS for the main page 
    |-- 📄 MainPage.tsx # Main page component 
    |-- 📄 setupTests.ts # Testing setup 
    |-- 📄 SignInSide.tsx # Sign-in page component 
    |-- 📄 App.tsx # Main application file 
    |-- 📄 index.css # Global styles 
    |-- 📄 index.tsx # Application entry point 
    |-- 📄 logo.html # Logo file 
    |-- 📁 components # React components folder 
    |   |-- CustomIcons.tsx 📄 
    |   |-- CustomIcons.css 🎨 
    |   |-- Dashboard.tsx 📄 
    |   |-- Dashboard.css 🎨
    |   |-- ForgotPassword.tsx 📄
    |   |-- Navbar.tsx 📄
    |   |-- Navbar.css 🎨
    |   |-- Profile.tsx 📄
    |   |-- rofile.css 🎨
    |   |-- Forum.tsx 📄
    |   |-- Forum.css 🎨
    |   |-- GenerateReport.tsx 📄
    |   |-- GenerateReport.css 🎨
    |   |-- Records.tsx 📄
    |   |-- Records.css 🎨
    |   |-- Report.tsx 📄 
    |   |-- Report.css 🎨
    |   |-- SignInCard.tsx 📄
    |   |-- SignInCard.css 🎨
    |   `-- ToastifyStyles.tsx 📄
    |-- 📁theme
    |   |-- AppTheme.tsx 📄
    |   |-- ColorModeIconDropdown.tsx 📄
    |   |-- ColorModeSelect.tsx 📄
    |   `-- themePrimitives.ts 📜
    `-- 📁customizations
        |-- feedback.tsx 📄
        |-- inputs.tsx 📄
        |-- navigation.tsx 📄
        |-- surfaces.ts 📜
        `-- dataDisplay.tsx 📄
📁 Python files 
|-- 📁 Computer Vision  
|   |-- 📁 Model-weights
|   |   |-- yolo.pt
|   |   `-- cnn_weights.h5
|   |-- 📁 Datasets
|   |   `-- dataset.csv
|   |-- 📄 liver.py
|   |-- 📄 brain.py
|   |-- 📄 eyes.py
|   |-- 📄 lungs.py 
|   `-- 📄 kidney.py
`-- 📁 Natural Language Processing
    |-- 📄 rag_mdb.py
    `-- 📦 dataset.zip
```

 <h2>🔥 Future Enhancements</h2>
 
- 🏥 Integrate additional AI models for more disease classification.
- 📊 Add data visualization for medical trends.
- 🌍 Multi-language support for medical reports.

 <h2>🤝 Contributors</h2>

- Prasanna Patwardhan
- Yash Kulkarni
- Piyush Deshmukh
- Rahul Dewani
- Yugandhar Chawale

<h2> 📧 Contact</h2>
For queries, reach out at:
📩 team.datagurus@gmail.com
