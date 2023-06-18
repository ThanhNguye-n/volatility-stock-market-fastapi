import os
from glob import glob
import numpy as np
import joblib
import pandas as pd
from arch import arch_model
from config import settings
from data import AlphaVantageAPI, SQLRepository
import scipy.stats
from datetime import datetime
from pathlib import Path

class DD_EMWA:
    """Class for training DD_EMWA model and generating predictions.

    Atttributes
    -----------
    ticker : str
        Ticker symbol of the equity whose volatility will be predicted.
    repo : SQLRepository
        The repository where the training data will be stored.
    use_new_data : bool
        Whether to download new data from the AlphaVantage API to train
        the model or to use the existing data stored in the repository.
    model_directory : str
        Path for directory where trained models will be stored.

    Methods
    -------
    wrangle_data
        Generate equity returns from data in database.
    predict
        Generate volatilty forecast from trained model.
    """

    def __init__(self, ticker, repo, use_new_data):
    
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = settings.model_directory

    def wrangle_data(self, n_observations, volatility_type):

        """Extract data from database (or get from AlphaVantage), transform it
        for training model, and attach it to `self.data`.

        Parameters
        ----------
        n_observations : int
            Number of observations to retrieve from database
            
        volatility_type : str
            There are 2 type of volatility: 'log_return' and 'pct_change'
        Returns
        -------
        None
        """
        # Add new data to database if required
        if self.use_new_data:
            api = AlphaVantageAPI()
            new_data = api.get_daily(ticker=self.ticker)
            self.repo.insert_table(table_name=self.ticker, records=new_data, if_exists='replace')

        # Pull data from SQL database
        df = self.repo.read_table(table_name=self.ticker, limit=n_observations+1)

        # Clean data, attach to class as `data` attribute
        df.sort_index(ascending=True, inplace=True)
        
        if volatility_type=='log_return':
            df['return'] = np.log(df['close'] / df['close'].shift(1))*100
        if volatility_type=='pct_change':
            df['return'] = df['close'].pct_change()*100

        self.data = df['return'].dropna()
        
    def __rho_cal(self, X):
        """
        Calculates the sample sign correlation using the Pearson correlation coefficient.
        
        """
        rho_hat =  scipy.stats.pearsonr(X-np.mean(X), np.sign(X-  np.mean(X)))
        return rho_hat[0]

    def DD_volatility(self, y, cut_t, alpha):
        """Calculates the forecasted volatility and root mean square error (RMSE) 
        using the Double-Exponential (Holt) Smoothing method.

        Parameters
        ----------
        y : dataframe
            A time series data representing the observed volatility.

        cut_t : int
            An integer value indicating the number of initial observations to use in 
            calculating the initial smoothed statistic.
            
        alpha: list
            A list of values representing the smoothing factor to use for the volatility forecast.

        Returns
        -------
        vol_forecast : This is the forecasted volatility, which value returned depends on 
                       the minimum value of the MSE_alpha.
                       
        RMSE: It represents the square root of the minimum MSE_alpha value.
        """
        t = len(y)
        
        # calculate sample sign correlation
        rho = self.__rho_cal(y)
        
        # calculate observed volatility
        vol = abs(y-np.mean(y))/rho
        MSE_alpha = np.zeros(len(alpha))
        sn = np.zeros(len(alpha)) # volatility
        
        for a in range(len(alpha)):
            # initial smoothed statistic 
            s = np.mean(vol[0:cut_t]) 
            error = np.zeros(t)
            for i in range(t):
                error[i] = vol[i] - s
                s = alpha[a]*vol[i]+(1-alpha[a])*s
            
            # forecast error sum of squares (FESS)
            MSE_alpha[a] = np.mean((error[(len(error)-cut_t):(len(error))])**2) 
            sn[a] = s
         
        #get the minimum value of the MSE_alpha 
        vol_forecast = sn[[i for i, j in enumerate(MSE_alpha) if j == min(MSE_alpha)]]
        
        RMSE = np.sqrt(min(MSE_alpha))
        return vol_forecast[0].astype(float), RMSE

    def predict_volatility(self, horizon, cut_t=5, alpha = [0.9, 0.95, 0.99]):

        """Predict volatility using `self.model`

        Parameters
        ----------
        horizon : int
            Horizon of forecast, by default 5.
            
        cut_t : int
            An integer value indicating the number of initial observations to use in 
            calculating the initial smoothed statistic.
            
        alpha: list
            A list of values representing the smoothing factor to use for the volatility forecast.
            
        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        
        y_train = self.data.values
        prediction = []
        start_index = self.data.index[-1]
        
        # Generate variance forecast
        for day in range(horizon):
            vol_forecast, _ = self.DD_volatility(y_train[-7:], cut_t, alpha) #get last 7 days
            y_train = np.append(y_train, vol_forecast)
            prediction.append(vol_forecast)
        
        # Calculate forecast start date
        start_index = pd.to_datetime(start_index) + pd.DateOffset(days=1)

        # Create the date range for the index
        prediction_dates = pd.bdate_range(start=start_index, periods=horizon, freq='D')
        
        # Create prediction index labels, ISO 8601 format
        prediction_index = [d.isoformat() for d in prediction_dates]

        # Create the Series using prediction as the data and index_dates as the index
        prediction_series = pd.Series(prediction, index=prediction_index)
        
        return prediction_series.to_dict()
            
        
        
class GarchModel:
    """Class for training GARCH model and generating predictions.

    Atttributes
    -----------
    ticker : str
        Ticker symbol of the equity whose volatility will be predicted.
    repo : SQLRepository
        The repository where the training data will be stored.
    use_new_data : bool
        Whether to download new data from the AlphaVantage API to train
        the model or to use the existing data stored in the repository.
    model_directory : str
        Path for directory where trained models will be stored.

    Methods
    -------
    wrangle_data
        Generate equity returns from data in database.
    fit
        Fit model to training data.
    predict
        Generate volatilty forecast from trained model.
    dump
        Save trained model to file.
    load
        Load trained model from file.
    """

    def __init__(self, ticker, repo, use_new_data):
    
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = settings.model_directory

    def wrangle_data(self, n_observations, volatility_type):

        """Extract data from database (or get from AlphaVantage), transform it
        for training model, and attach it to `self.data`.

        Parameters
        ----------
        n_observations : int
            Number of observations to retrieve from database
            
        volatility_type : str
            There are 2 type of volatility: 'log_return' and 'pct_change'
        Returns
        -------
        None
        """
        # Add new data to database if required
        if self.use_new_data:
            api = AlphaVantageAPI()
            new_data = api.get_daily(ticker=self.ticker)
            self.repo.insert_table(table_name=self.ticker, records=new_data, if_exists='replace')

        # Pull data from SQL database
        df = self.repo.read_table(table_name=self.ticker, limit=n_observations+1)

        # Clean data, attach to class as `data` attribute
        df.sort_index(ascending=True, inplace=True)
        
        if volatility_type=='log_return':
            df['return'] = np.log(df['close'] / df['close'].shift(1))*100
        if volatility_type=='pct_change':
            df['return'] = df['close'].pct_change()*100

        self.data = df['return'].dropna()
        self.vol_type = volatility_type

    def fit(self, p=1 , q=1):

        """Create model, fit to `self.data`, and attach to `self.model` attribute.
        For assignment, also assigns adds metrics to `self.aic` and `self.bic`.

        Parameters
        ----------
        p : int
            Lag order of the symmetric innovation

        q : ind
            Lag order of lagged volatility

        Returns
        -------
        None
        """
        # Train Model, attach to `self.model`
        self.model = arch_model(self.data, p=p, q=q, rescale=False).fit(disp=0)
        self.aic = self.model.aic
        self.bic = self.model.bic
        
    def __clean_prediction(self, prediction):

        """Reformat model prediction to JSON.

        Parameters
        ----------
        prediction : pd.DataFrame
            Variance from a `ARCHModelForecast`

        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        # Calculate forecast start date
        start = prediction.index[0] + pd.DateOffset(days=1)

        # Create date range
        prediction_dates = pd.bdate_range(start=start, periods=prediction.shape[1])

        # Create prediction index labels, ISO 8601 format
        prediction_index = [d.isoformat() for d in prediction_dates]

        # Extract predictions from DataFrame, get square root
        data = prediction.values.flatten()**0.5

        # Combine `data` and `prediction_index` into Series
        prediction_formatted = pd.Series(data, index=prediction_index)

        # Return Series as dictionary
        return prediction_formatted.to_dict()
    
    def predict_volatility(self, horizon):

        """Predict volatility using `self.model`

        Parameters
        ----------
        horizon : int
            Horizon of forecast, by default 5.

        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        # Generate variance forecast from `self.model`
        prediction = self.model.forecast(horizon=horizon, reindex=False).variance

        # Format prediction with `self.__clean_predction`
        prediction_formatted = self.__clean_prediction(prediction)

        # Return `prediction_formatted`
        return prediction_formatted

    def dump(self):

        """Save model to `self.model_directory` with timestamp.

        Returns
        -------
        str
            filepath where model was saved.
        """
        
        # Create timestamp in ISO format
        timestamp = pd.Timestamp.now().isoformat()
        timestamp = timestamp[0:10]

        # Create filepath, including `self.model_directory`
        os.makedirs(self.model_directory, exist_ok=True)
        filename = f'{timestamp}_{self.ticker}_{self.vol_type}.pkl'
        filepath = os.path.join(f'{self.model_directory}/', filename)

        # Save `self.model`
        joblib.dump(self.model, filepath)
        # Return filepath
        return filepath


    def load(self, volatility_type):

        """Load most recent model in `self.model_directory` for `self.ticker`,
        attach to `self.model` attribute.

        """
        # Create pattern for glob search
        pattern = os.path.join(f'{self.model_directory}/', f"*{self.ticker}_{volatility_type}.pkl")

        # Try to find path of latest model
        try: 
            model_path = sorted(glob(pattern))[-1]
        except IndexError:
            raise Exception(f"No model trained for '{self.ticker}'")

        # Load model
        self.model = joblib.load(model_path)
        

