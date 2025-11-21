
# 🚗 Vehicle Insurance Prediction System (End-to-End MLOps Project)

> **An industry-grade, production-ready Machine Learning system that predicts vehicle insurance outcomes using a complete MLOps pipeline.**
>
> This project demonstrates how to build, train, validate, deploy, and monitor a real-world ML application with scalable architecture, CI/CD automation, and cloud integration.

#### Web App Link: *[Vehicle Insurance App](http://18.234.35.23:5000/)*

## 📌 Project Highlights

✅ Full end-to-end ML lifecycle implementation

✅ Dynamic data ingestion from MongoDB Atlas

✅ Modular pipeline architecture (Ingestion → Validation → Transformation → Training → Evaluation → Deployment)

✅ CI/CD with Docker, GitHub Actions & AWS EC2

✅ Model storage & versioning using AWS S3

✅ FastAPI-based prediction web application

✅ Production-ready logging & exception handling

This project replicates how ML systems are actually built and deployed in real companies.

---

## 🧠 System Architecture Overview

```text

                    ┌──────────────────────────┐
                    │        USER / DATASET   │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   MongoDB Atlas │
                         └────────┬─────────┘
                                  │
                    Data Fetch & Transformation
                                  │
                                  ▼
┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ Data Ingestion│ → │ Data Validation│ → │ Data Transformation│
└───────────────┘   └─────────────────┘   └──────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Model Training │
                         └────────┬─────────┘
                                  │
                                  ▼
                 ┌─────────────────────────┐
                 │ Model Evaluation       │
                 │ (Compare with old model)│
                 └────────┬────────────────┘
                          │
                          ▼
              ┌────────────────────────────┐
              │ Model stored in AWS S3    │
              │ (Model Registry)          │
              └────────┬──────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ FastAPI Prediction Web App │
         └────────┬────────────────────┘
                  ▼
         GitHub Actions CI/CD Pipeline
                  ▼
         Docker Image → AWS EC2 Deployment
```

---

## ⚙️ Project Workflow (Simplified for Recruiters)

### 1️⃣ Project Initialization

* Auto-generate project structure using `template.py`
* Configure packaging using `setup.py` and `pyproject.toml`
* Install dependencies using virtual environment

### 2️⃣ Database Setup (MongoDB Atlas)

* Cloud database to store vehicle insurance dataset
* Python notebook pushes dataset to MongoDB
* Secure connection through environment variables

### 3️⃣ Data Pipeline

Each module is isolated and production-ready:

| Component           | Description                                         |
| ------------------- | --------------------------------------------------- |
| Data Ingestion      | Pulls data from MongoDB and converts to DataFrame   |
| Data Validation     | Ensures schema correctness using config.schema.yaml |
| Data Transformation | Feature engineering + preprocessing                 |
| Model Trainer       | Trains ML model                                     |
| Model Evaluation    | Compares new vs old model                           |
| Model Pusher        | Stores selected model in AWS S3                     |

---

## ☁️ Cloud & Deployment Flow

### CI/CD Pipeline

* GitHub Actions automatically builds image & pushes to AWS ECR
* AWS EC2 (self-hosted runner) pulls image and runs container
* Application exposed via public IP and port

### Deployment Stack

| Layer            | Technology             |
| ---------------- | ---------------------- |
| Backend          | FastAPI                |
| Containerization | Docker                 |
| CI/CD            | GitHub Actions         |
| Hosting          | AWS EC2                |
| Storage          | AWS S3 + MongoDB Atlas |

---

## 📡 Web Application

Access the application at:

```
http://<EC2_PUBLIC_IP>:5000
```

Routes:

---



| Endpoint      | Purpose                |
| ------------- | ---------------------- |
| `/`         | Home page              |
| `/predict`  | Insurance prediction   |
| `/training` | Trigger model training |

---

## 🔐 Key Technical Features

### 🔹 Logging & Exception Handling

* Custom logger and exception module
* All pipeline steps tracked for debugging

### 🔹 Model Versioning

* S3-based model registry
* Comparison logic for best model selection

### 🔹 Config Driven Architecture

* Constants & schema files control entire pipeline

### 🔹 Fully Automated Deployment

* From code commit to EC2 deployment

---

## 🧪 Local Setup Instructions

```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```

Set environment variables:

For Bash:

```bash
export MONGODB_URL="<your connection string>"
```

For PowerShell:

```powershell
$env:MONGODB_URL="<your connection string>"
```

---

## 🐳 Docker Execution

```bash
docker build -t vehicle-insurance .
docker run -p 5080:5000 vehicle-insurance
```

---

## 📈 What This Project Demonstrates

✅ MLOps workflow understanding

✅ Scalable architecture

✅ Cloud-native ML deployment

✅ Industrial-standard best practices

✅ Model lifecycle automation

This project showcases how to convert a raw Machine Learning concept into a production-ready, automated real-world system.

---

## 👨‍💻 Ideal For

* Data Scientists
* ML Engineers
* MLOps Engineers
* AI Engineers

Perfect portfolio project to demonstrate real-world deployment skills.

---

## 💼 Recruiter-Friendly Summary

> This project demonstrates enterprise-level MLOps skills by building a complete vehicle insurance prediction system using MongoDB, FastAPI, Docker, AWS, and CI/CD automation. It showcases the full lifecycle from data ingestion to live deployment with cloud-based scalability and maintainability.

---

## 📬 Contact

For questions or collaboration, reach out via GitHub or LinkedIn.

---

⭐ If you find this project helpful, consider giving it a star!
