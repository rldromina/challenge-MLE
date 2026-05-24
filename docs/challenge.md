# Challenge Documentation

## Part I: Model Selection

Six models were evaluated, combining two algorithms (XGBoost and Logistic Regression: LR), with and without class balancing, and with all features vs. the top 10 most important features.

Since the dataset is imbalanced (aprox 82% no delay, aprox 18% delay), models trained without class balancing achieve aprox 81% accuracy but a recall of 0.00 for the delay class, meaning they never predict a delay. This makes them useless for the actual problem.

Models trained with class balancing drop to 60% accuracy but achieve a recall of 0.60 for the delay class, which is the metric that matters here.

We can see that XGBoost and LR both with balancing produce identical results. This suggests the problem is linearly separable with the selected features, meaning a linear model is sufficient.

**Chosen model: LR with class balancing and top 10 features.**

LR was chosen over XGBoost because both models perform equally, and LR is simpler, faster to train and easier to interpret. If the problem becomes more complex or non-linear features are added in the future, XGBoost would be the natural next step.

## Part I: Implementation

A `DelayModel` class was created to predict flight delays. The class uses the 10 most important features identified during exploration and
defines a threshold of 15 minutes to label a flight as delayed.

The implementation has three steps:

In `preprocess`, the raw data is prepared for training or serving. One-hot encoding is applied to the `OPERA`, `TIPOVUELO` and `MES`
columns, and the result is filtered to keep only the top 10 features. When a target column is provided, the delay label is computed from the
difference between scheduled and actual times.

In `fit`, the model is trained using Logistic Regression with class balancing, since the dataset is heavily imbalanced toward non-delayed
flights.

In `predict`, the trained model returns a list of integers (0 or 1) indicating whether each flight is expected to be delayed.

## Part II: API

A REST API was built using FastAPI to expose the delay prediction model.

Two Pydantic models were defined to validate the incoming request: `Flight`, which represents a single flight with its airline (`OPERA`), flight type (`TIPOVUELO`) and month (`MES`); and `PredictRequest`, which wraps a list of flights.

The validation rules are: `TIPOVUELO` only accepts `"N"` or `"I"`, and `MES` must be between 1 and 12. Any invalid input returns a 400 error.

The `POST /predict` endpoint receives the request, preprocesses the flights using the `DelayModel` and returns the predictions as a list of integers.

## Part III: Deployment

A Docker image was created to containerize the API and run it in the cloud.  The image was built for linux/amd64 and pushed to Google Container Registry (GCR).  The API was then deployed on Google Cloud Run, which allows it to scale  automatically based on traffic.

The deployed API is available at: https://challenge-mle-282015016889.us-central1.run.app

A stress test was run against the deployed API with 100 concurrent users  over 60 seconds, resulting in 7200 requests with 0 failures and an average response time of 226ms.

## Part IV: CI/CD

Two GitHub Actions workflows were configured to automate testing and deployment.

The CI workflow runs on every push and pull request to `main`. It installs the dependencies and runs both the model and API tests automatically.

The CD workflow runs on every push to `main`. It builds the Docker image for linux, pushes it to Google Container Registry and deploys the updated version to Cloud Run. Authentication with GCP is handled through Workload Identity Federation, which allows GitHub Actions to authenticate securely without storing credentials.