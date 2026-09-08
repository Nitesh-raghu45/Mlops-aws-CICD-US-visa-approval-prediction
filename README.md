# Mlops-aws-CICD-US-visa-approval-prediction

An end-to-end machine learning system that predicts whether a US work visa (H-1B / EB-type) application will be **certified or denied**, based on employer and applicant attributes. The project covers the full MLOps lifecycle: data ingestion from MongoDB, validation, transformation, model training and evaluation, model deployment to AWS S3, a FastAPI web app for serving predictions, and CI/CD to AWS via Docker, ECR, and EC2.

## Problem Statement

Immigration case management processes and reviews thousands of visa applications every year. Manually reviewing each case is time-consuming and inconsistent. This project builds a classification model that predicts the likely outcome (`Certified` / `Denied`) of a visa application based on features such as employer size, education, prior job experience, wage, and region of employment — helping streamline the review process.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.8 |
| Data storage | MongoDB |
| ML / Data | pandas, numpy, scikit-learn, XGBoost, CatBoost, imbalanced-learn |
| Data drift monitoring | Evidently |
| Model registry / artifact storage | AWS S3 |
| Serving | FastAPI, Uvicorn, Jinja2 |
| Containerization | Docker |
| CI/CD | GitHub Actions, AWS ECR, AWS EC2 (self-hosted runner) |

## Project Architecture

```
MongoDB (raw data)
      │
      ▼
Data Ingestion  ──►  Data Validation  ──►  Data Transformation
      │                                          │
      │                                          ▼
      │                                    Model Trainer
      │                                          │
      │                                          ▼
      │                                  Model Evaluation
      │                                          │
      │                              (accept if better than
      │                               previously deployed model)
      │                                          │
      │                                          ▼
      │                                    Model Pusher ──► AWS S3
      │
      ▼
FastAPI App ──► loads best model from S3 ──► serves predictions (web form + API)
```

Each stage is implemented as an independent, config-driven **component** under `us_visa/components/`, orchestrated by `us_visa/pipline/training_pipeline.py` (training) and `us_visa/pipline/prediction_pipeline.py` (inference).

## Project Structure

```
├── us_visa/
│   ├── components/          # Pipeline stages: ingestion, validation, transformation,
│   │                         # model trainer, model evaluation, model pusher
│   ├── configuration/       # MongoDB and AWS (S3) client connections
│   ├── cloud_storage/       # S3 read/write helper (SimpleStorageService)
│   ├── data_access/         # MongoDB → pandas DataFrame access layer
│   ├── entity/              # Config and artifact dataclasses passed between stages
│   ├── pipline/             # Training pipeline and prediction pipeline orchestration
│   ├── constants/           # Project-wide constants
│   ├── exception/           # Custom exception handling
│   ├── logger/               # Custom logging setup
│   └── utils/                # Shared utility functions (YAML I/O, object save/load, etc.)
├── notebook/                 # EDA, feature engineering, MongoDB and data-drift demo notebooks
├── config/
│   ├── schema.yaml           # Expected dataset schema (columns, dtypes)
│   └── model.yaml            # Model candidates and hyperparameter search grids
├── templates/usvisa.html     # Web UI form for single predictions
├── app.py                    # FastAPI application entry point
├── template.py                # Script to scaffold the project's folder/file structure
├── Dockerfile                 # Container image definition
├── .github_/workflows/aws.yaml # CI/CD pipeline (build → ECR → deploy to EC2)
├── requirements.txt
└── setup.py
```

## Dataset

`notebook/Visadataset.csv` — visa case records with features including:

- `continent`, `education_of_employee`, `has_job_experience`, `requires_job_training`
- `no_of_employees`, `yr_of_estab`, `region_of_employment`
- `prevailing_wage`, `unit_of_wage`, `full_time_position`
- Target: `case_status` (Certified / Denied)

The schema is formally defined in `config/schema.yaml` and validated automatically during the **Data Validation** stage.

## Pipeline Stages

1. **Data Ingestion** — pulls the raw collection from MongoDB, exports it to a feature store CSV, and splits it into train/test sets.
2. **Data Validation** — checks the ingested data against `config/schema.yaml` for missing columns, type mismatches, and dataset drift.
3. **Data Transformation** — applies preprocessing (encoding, scaling, imbalance handling) and produces model-ready train/test arrays.
4. **Model Trainer** — trains and tunes candidate models defined in `config/model.yaml` (e.g., KNN, and others via grid search), selecting the best performer.
5. **Model Evaluation** — compares the newly trained model against the currently deployed model (if any) in S3; only accepts it if it improves on the baseline.
6. **Model Pusher** — if accepted, pushes the new model artifact to the configured AWS S3 bucket for serving.

## Local Setup

### Prerequisites
- Python 3.8+
- A MongoDB connection string (Atlas or self-hosted)
- AWS account with an S3 bucket and IAM credentials

### 1. Clone and create an environment
```bash
git clone <repo-url>
cd Mlops-aws-CICD-US-visa-approval-prediction
conda create -n visa python=3.8 -y
conda activate visa
pip install -r requirements.txt
```

### 2. Configure environment variables
```bash
export MONGODB_URL="<your-mongodb-connection-string>"
export AWS_ACCESS_KEY_ID="<your-aws-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-aws-secret-key>"
export AWS_DEFAULT_REGION="<your-aws-region>"
```
(On Windows, use `set` instead of `export`.)

### 3. Run the training pipeline
```python
from us_visa.pipline.training_pipeline import TrainPipeline

pipeline = TrainPipeline()
pipeline.run_pipeline()
```

### 4. Run the web app
```bash
python app.py
```
Then visit `http://localhost:8080` to submit case details through the web form (`templates/usvisa.html`) and get a prediction, or to trigger training via the app's `/train` route.

## Docker

Build and run the app in a container:
```bash
docker build -t us-visa-app .
docker run -p 8080:8080 \
  -e MONGODB_URL="<mongodb-url>" \
  -e AWS_ACCESS_KEY_ID="<key>" \
  -e AWS_SECRET_ACCESS_KEY="<secret>" \
  -e AWS_DEFAULT_REGION="<region>" \
  us-visa-app
```

## CI/CD (AWS)

The GitHub Actions workflow (`.github_/workflows/aws.yaml`) automates deployment on every push to `main`:

1. **Continuous Integration** — checks out the code, authenticates to AWS, builds the Docker image, and pushes it to **Amazon ECR**.
2. **Continuous Deployment** — a self-hosted runner (an EC2 instance) pulls the latest image from ECR and runs it, exposing the app on port `8080`.

### Required GitHub Secrets
| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM access key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key |
| `AWS_DEFAULT_REGION` | AWS region (e.g. `us-east-1`) |
| `ECR_REPO` | Target ECR repository name |
| `MONGODB_URL` | MongoDB connection string used at container runtime |

### High-level AWS setup
1. Create an IAM user with `AmazonEC2ContainerRegistryFullAccess` and `AmazonEC2FullAccess`.
2. Create an ECR repository to host the Docker image.
3. Launch an EC2 instance (Ubuntu), install Docker on it, and register it as a **self-hosted GitHub Actions runner**.
4. Add the secrets above to the GitHub repo, then push to `main` to trigger the pipeline.

## Data Drift Monitoring

`notebook/data_drift_demo_evidently.ipynb` demonstrates using **Evidently** to detect drift between reference (training) and current (production) data distributions — useful for deciding when the model needs retraining.

## Author

Nitesh Raghuwanshi
