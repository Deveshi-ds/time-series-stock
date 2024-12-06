# Stock Price Prediction Using LSTM Networks  

## Overview  
This project implements a predictive model using Long Short-Term Memory (LSTM) networks to forecast stock price movements. By capturing long-term dependencies in sequential data, the model provides accurate and actionable insights for investors. Integrated with a Power BI dashboard that updates daily, it delivers an intuitive and interactive interface for analyzing stock trends and making informed investment decisions.  

## Features  
- **Time Series Analysis:** Leverages LSTM networks to identify patterns and trends in historical stock market data.  
- **Data Normalization:** Uses MinMaxScaler to standardize features for improved model performance.  
- **Real-Time Updates:** Connects to a Power BI dashboard with daily updates for seamless visualization of predictions and trends.  
- **Interactive Visualizations:** Offers stakeholders a user-friendly interface to analyze historical data, predicted prices, and trends.  

## Objectives  
- Develop a robust predictive model for stock price forecasting using sequential data.  
- Enable users to visualize and interpret predictions effectively through Power BI dashboards.  
- Provide real-time updates for actionable insights in volatile financial markets.  

## Technology Stack  
- **Programming Language:** Python  
- **Libraries:** TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn  
- **Tools:** Power BI, MinMaxScaler  
- **Frameworks:** LSTM networks  

## Dataset  
The project utilizes historical stock price data, including features such as:  
- Historical closing prices  
- Lagged closing prices over a defined lookback period  

## How to Run  
1. **Clone the Repository:**  
   ```bash  
   git clone <repository-url>  
   cd <repository-directory>  
   ```  

2. **Install Dependencies:**  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Train the Model:**  
   ```bash  
   python train_lstm.py  
   ```  

4. **Visualize Results:**  
   - Export predictions to Power BI for daily updates and interactive dashboards.  

## Results  
- Accurate stock price predictions with interactive trend analysis for informed investment decisions.  
- Daily updated Power BI dashboards for real-time insights into market movements.  

## Future Scope  
- Extend the model to include additional features like trading volume and market sentiment analysis.  
- Automate end-to-end integration with live stock data sources for real-time prediction updates.  

## Contributing  
Contributions are welcome! Feel free to submit a pull request or open an issue for suggestions and improvements.  

