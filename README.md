# End-to-End Service Management System

This project is a prototype **end-to-end service booking and technician assignment system**.  
It integrates **AWS (DynamoDB, S3)**, **OpenAI**, and a simple **frontend UI** for service booking.

---

## 📂 Project Structure
# End-to-End Service Management System

This project is a prototype **end-to-end service booking and technician assignment system**.  
It integrates **AWS (DynamoDB, S3)**, **OpenAI**, and a simple **frontend UI** for service booking.

---

## 📂 Project Structure

```bash
project-root/
│── .env                # Environment variables (NOT in git)
│── main.py             # Entry point
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation
│
├── src/                # Python source code
│   ├── graph.py        # Booking workflow graph
│   ├── database/       # Database scripts
│   │   └── insert_technicians.py
│   └── utils/          # Helper functions
│
├── notebooks/          # Jupyter notebooks
│   ├── graphmaker.ipynb
│   └── insert_technicians.ipynb
│
├── data/               # Input/test data
│   └── testcase.txt
│
├── docs/               # Documentation & reports
│   ├── connect_to_dynamodb.txt
│   └── output.pdf
│
├── figures/            # Images & diagrams
│   ├── booking_pipeline.png
│   └── solution.png
│
└── web/                # Frontend
    └── index.html
---

## 🚀 Setup

### 1. Clone repository
```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject