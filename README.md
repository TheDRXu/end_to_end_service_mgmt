# End-to-End Service Management System

This project is a prototype **end-to-end service booking and technician assignment system**.  
It integrates **AWS (DynamoDB, S3)**, **OpenAI**, and a simple **frontend UI** for service booking.

---

## 📂 Project Structure

```bash
SERVICE_MANAGEMENT_SYSTEM/
│── .env                # Environment variables (NOT in git)
│── .gitignore
│── main.py             # Entry point
│── README.md           # Project documentation
│
├── docs/               # Documentation & notes
│   └── connect_to_dynamodb.txt
│
├── figure/             # Images & diagrams
│   ├── booking_pipeline.png
│   └── solution.png
│
├── front-end/          # Frontend files
│   └── index.html
│
├── src/                # Python source code
│   └── graph.py
│
├── test-case/          # Test datasets & notebooks
│   ├── insert_technicians.ipynb
│   └── testcase.txt
│
└── __pycache__/        # Auto-generated cache files
```

---

## 🚀 Setup

### 1. Clone repository
```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add `.env` file
```env
DDB_TABLE=end_to_end_service_mgmt
OPENAI_API_KEY=sk-xxxxxxx
AWS_ACCESS_KEY_ID=xxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxx
AWS_SESSION_TOKEN=xxxxxxx
AWS_DEFAULT_REGION=ap-southeast-2
```

⚠️ Never commit `.env` to GitHub.

---

## ▶️ Run the Project

Start backend:
```bash
python main.py
```

API runs on:
```
http://localhost:8000
```

Open the frontend:
```
front-end/index.html
```

---

## 🛠 Features

- **AI-powered booking** – users can book via natural language.
- **Smart technician matching** – skills, location, workload, availability.
- **AWS backend** – DynamoDB for bookings, S3 for storage.
- **Frontend UI** – simple static booking page.

---

## 📊 Figures

Booking pipeline:

![Booking Pipeline](figure/booking_pipeline.png)

Solution overview:

![Solution](figure/solution.png)

---

## 📖 Documentation

- DynamoDB connection notes: `docs/connect_to_dynamodb.txt`
