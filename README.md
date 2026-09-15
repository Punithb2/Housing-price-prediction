# 🏡 Housing Price Prediction

A multiple linear regression model built **from scratch with NumPy** (no scikit-learn) to predict house prices, served through a small **Streamlit** web app.

The model is trained in a Jupyter notebook using hand-written z-score normalization, a mean squared error cost function, and batch gradient descent. The trained parameters are saved to a pickle file, which the Streamlit app loads to give instant price estimates.

## Project Structure

```
Housing_price_prediction/
├── Housing.csv                       # Dataset (545 houses, 13 columns)
├── Housing_price_prediction.ipynb    # Data prep, training, evaluation, model export
├── house_model.pkl                   # Trained model parameters (weights, bias, mu, sigma)
├── app.py                            # Streamlit web app for predictions
└── requirements.txt                  # Python dependencies
```

## Dataset

`Housing.csv` contains 545 houses with the following columns:

| Column | Type | Description |
|---|---|---|
| `price` | numeric | Sale price (target) |
| `area` | numeric | Area in square feet |
| `bedrooms` | numeric | Number of bedrooms |
| `bathrooms` | numeric | Number of bathrooms |
| `stories` | numeric | Number of stories |
| `mainroad` | yes/no | Connected to a main road |
| `guestroom` | yes/no | Has a guest room |
| `basement` | yes/no | Has a basement |
| `hotwaterheating` | yes/no | Has hot water heating |
| `airconditioning` | yes/no | Has air conditioning |
| `parking` | numeric | Number of parking spots |
| `prefarea` | yes/no | Located in a preferred area |
| `furnishingstatus` | category | `furnished`, `semi-furnished`, or `unfurnished` |

## How the Model Works

### 1. Preprocessing
- Yes/no columns are mapped to `1`/`0`.
- `furnishingstatus` is one-hot encoded (`drop_first=True` to avoid the dummy variable trap).

### 2. Feature Selection
The features were ranked by their correlation with `price`, and the top six were kept:

| Feature | Correlation with price |
|---|---|
| `area` | 0.536 |
| `bathrooms` | 0.518 |
| `airconditioning` | 0.453 |
| `stories` | 0.421 |
| `parking` | 0.384 |
| `bedrooms` | 0.366 |

The target `price` is divided by `100,000` so the cost stays in a manageable range during training.

### 3. Feature Scaling
Each feature is z-score normalized using the training mean (`mu`) and standard deviation (`sigma`):

```
x_scaled = (x - mu) / sigma
```

### 4. Training
The model is `f(x) = w · x + b`, trained with batch gradient descent:

- **Cost:** `J(w, b) = (1 / 2m) · Σ (f(x) − y)²`
- **Learning rate (alpha):** `0.01`
- **Iterations:** `5000`
- **Initial parameters:** all zeros

The cost falls from about **1283.87** to **68.66** and levels off by around 1,000 iterations. The notebook plots the learning curve to show this.

### 5. Export
The weights, bias, `mu`, and `sigma` are saved together in `house_model.pkl`:

```python
model_data = {'weights': final_w, 'bias': final_b, 'mu': mu_X, 'sigma': sigma_X}
```

## Getting Started

### Prerequisites
- Python 3 (the project was developed with Python 3.14)

### Installation

```bash
git clone <your-repo-url>
cd Housing_price_prediction

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501). Enter:

- Area (square feet)
- Number of bathrooms
- Number of bedrooms
- Number of stories
- Parking spots
- Air conditioning (Yes/No)

Click **Predict Price** to get the estimated price.

### Retrain the Model

```bash
jupyter notebook Housing_price_prediction.ipynb
```

Run all cells. The last cell overwrites `house_model.pkl` with the new parameters, and the app picks them up the next time it starts.

## Example Prediction

From the notebook: a 6,000 sq ft house with 3 bathrooms, air conditioning, 1 story, no parking, and 2 bedrooms:

```
Features: [6000    3    1    1    0    2]
Predicted Price: 6,772,061.24
```

## Important: Feature Order

The model expects features in **exactly** this order, both in the notebook and in `app.py`:

```
['area', 'bathrooms', 'airconditioning', 'stories', 'parking', 'bedrooms']
```

If you change the selected features or their order in the notebook, update the feature array in `app.py` to match.

## Tech Stack

- **NumPy**: model math (normalization, cost, gradients, gradient descent)
- **pandas**: loading and preprocessing the data
- **Matplotlib**: learning curve plot
- **Jupyter**: training notebook
- **Streamlit**: web interface

## Possible Improvements

- Add a train/test split and report metrics such as RMSE or R²
- Use the remaining features (`prefarea`, `mainroad`, `guestroom`, and others)
- Compare the results with scikit-learn's `LinearRegression`
- Trim the Jupyter-only packages from `requirements.txt` for deployment
