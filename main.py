import sqlite3

from config import settings
from data import SQLRepository
from fastapi import FastAPI
from model import GarchModel, DD_EMWA
from pydantic import BaseModel


# Task 8.4.14, `FitIn` class
class FitIn(BaseModel):
    ticker: str
    use_new_data: bool
    n_observations: int
    volatility_type: str
    p: int
    q: int


# Task 8.4.14, `FitOut` class
class FitOut(FitIn):
    success: bool
    message : str


# Task 8.4.18, `PredictIn` class
class PredictIn(BaseModel):
    ticker: str
    n_days: int
    volatility_type: str
    model_name: str

# Task 8.4.18, `PredictOut` class
class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str


# Task 8.4.15
def build_model(ticker, use_new_data, model_name):

    # Create DB connection
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)

    # Create `SQLRepository`
    repo = SQLRepository(connection=connection)
    
    if model_name == 'garch':
        model = GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)
    
    if model_name == 'dd-emwa':
        model = DD_EMWA(ticker=ticker, use_new_data=use_new_data, repo=repo)
    # Return model
    return model


# Task 8.4.9
app = FastAPI()


# Task 8.4.11
# `"/hello" path with 200 status code
@app.get("/hello", status_code=200)
def hello():
    """Return dictionary with greeting message."""
    return {"message": "Hello World!"}



# Task 8.4.16, `"/fit" path, 200 status code
@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request: FitIn):

    """Fit model, return confirmation message.

    Parameters
    ----------
    request : FitIn

    Returns
    ------
    dict
        Must conform to `FitOut` class
    """
    # Create `response` dictionary from `request`
    response = request.dict()
    
    try:
        # Build model with `build_model` function
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data, model_name='garch')
        model.wrangle_data(n_observations=request.n_observations, volatility_type=request.volatility_type)
        model.fit(p=request.p, q=request.q)

        file_name = model.dump()

        #response
        response['success'] = True
        response['message'] = f"Trained and saved in '{file_name}'. Metrics: AIC {model.aic}, BIC {model.bic}."


    except Exception as e:
        response['success'] = False
        response['message'] = str(e)

    return response


# Task 8.4.19 `"/predict" path, 200 status code
@app.post("/predict", status_code=200, response_model=PredictOut)
def get_prediction(request: PredictIn):

    response = request.dict()
    
    if request.model_name == 'garch':
        try:
            model = build_model(ticker=request.ticker, use_new_data=False, model_name='garch')
            model.load(volatility_type=request.volatility_type)
            prediction = model.predict_volatility(horizon=request.n_days)

            #response
            response['success'] = True
            response['forecast'] = prediction
            response['message'] = ""

        except Exception as e:
            response['success'] = False
            response['forecast'] = {}
            response['message'] = str(e)
            
    elif request.model_name == 'dd-emwa':
        try:
            model = build_model(ticker=request.ticker, use_new_data=False, model_name='dd-emwa')
            model.wrangle_data(n_observations=request.n_days, volatility_type= request.volatility_type)
            prediction = model.predict_volatility(horizon=request.n_days)
            
            #response
            response['success'] = True
            response['forecast'] = prediction
            response['message'] = ""

        except Exception as e:
            response['success'] = False
            response['forecast'] = {}
            response['message'] = str(e)

    return response
