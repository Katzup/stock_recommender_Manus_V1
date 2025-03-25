import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import requests
import traceback
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Stock Analysis & Trading Backtester",
    page_icon="📈",
    layout="wide"
)


# Alpha Vantage API key setup
def get_alpha_vantage_api_key():
    return st.secrets.get("ALPHA_VANTAGE_API_KEY", "WEML0SKXI7OC592D")


# Fetch data from Alpha Vantage
def fetch_alpha_vantage_data(ticker):
    api_key = get_alpha_vantage_api_key()

    try:
        # Try to get daily adjusted data
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}'
        r = requests.get(url)
        data = r.json()

        if "Error Message" in data:
            st.error(f"Alpha Vantage API Error: {data['Error Message']}")
            return None

        if "Time Series (Daily)" not in data:
            st.error(f"No data found for {ticker} in Alpha Vantage API")
            return None

        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame(time_series).T

        # Rename columns and convert types
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume',
            '7. dividend amount': 'Dividend',
            '8. split coefficient': 'Split Coefficient'
        })

        # Convert strings to appropriate types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split Coefficient']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)

        # Sort by date
        df = df.sort_index()

        return df

    except Exception as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return None


# Recommendation system
def generate_recommendation(ticker, data_source='yahoo'):
    try:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)  # Increased for SMA 200

        if data_source == 'yahoo':
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            data = fetch_alpha_vantage_data(ticker)

        if data is None or len(data) == 0:
            return {"recommendation": "No Data", "reason": "Insufficient historical data", "score": 0}

        # Get the latest stock info
        if data_source == 'yahoo':
            try:
                stock_info = yf.Ticker(ticker).info
                current_price = float(data['Close'].iloc[-1])
                company_name = stock_info.get('shortName', ticker)
            except:
                current_price = float(data['Close'].iloc[-1])
                company_name = ticker
        else:
            stock_info = {}
            current_price = float(data['Close'].iloc[-1])
            company_name = ticker

        # ----- CALCULATING TECHNICAL INDICATORS -----
        # Calculate indicators without relying on intermediate DataFrames

        # Moving Averages
        sma50 = None
        sma200 = None
        ema20 = None

        if len(data) >= 50:
            sma50 = float(data['Close'].rolling(window=50).mean().iloc[-1])

        if len(data) >= 200:
            sma200 = float(data['Close'].rolling(window=200).mean().iloc[-1])

        if len(data) >= 20:
            ema20 = float(data['Close'].ewm(span=20, adjust=False).mean().iloc[-1])

        # RSI calculation
        rsi = None
        if len(data) >= 15:  # Need at least 15 days for a 14-day RSI
            delta = data['Close'].diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            down = abs(down)

            avg_gain = up.rolling(window=14).mean()
            avg_loss = down.rolling(window=14).mean()

            # Make sure we don't divide by zero
            last_avg_loss = float(avg_loss.iloc[-1])
            if last_avg_loss == 0:
                last_avg_loss = 0.001

            last_avg_gain = float(avg_gain.iloc[-1])
            rs = last_avg_gain / last_avg_loss
            rsi = 100 - (100 / (1 + rs))

        # MACD calculation
        macd = None
        macd_signal = None

        if len(data) >= 26:
            ema12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data['Close'].ewm(span=26, adjust=False).mean()
            macd_series = ema12 - ema26
            macd = float(macd_series.iloc[-1])

            if len(data) >= 35:  # Need 26 + 9 days
                signal_series = macd_series.ewm(span=9, adjust=False).mean()
                macd_signal = float(signal_series.iloc[-1])

                # Store previous values for crossover detection
                prev_macd = float(macd_series.iloc[-2]) if len(macd_series) >= 2 else None
                prev_signal = float(signal_series.iloc[-2]) if len(signal_series) >= 2 else None

        # Bollinger Bands
        bb_middle = None
        bb_upper = None
        bb_lower = None
        bb_width = None

        if len(data) >= 20:
            bb_middle = float(data['Close'].rolling(window=20).mean().iloc[-1])
            bb_stddev = float(data['Close'].rolling(window=20).std().iloc[-1])
            bb_upper = bb_middle + (bb_stddev * 2)
            bb_lower = bb_middle - (bb_stddev * 2)

            if bb_middle != 0:
                bb_width = (bb_upper - bb_lower) / bb_middle

            # Store previous BB values for width comparison
            if len(data) >= 41:  # Need data for 20-day window plus 21 days ago
                prev_bb_middle = float(data['Close'].rolling(window=20).mean().iloc[-21])
                prev_bb_stddev = float(data['Close'].rolling(window=20).std().iloc[-21])
                prev_bb_upper = prev_bb_middle + (prev_bb_stddev * 2)
                prev_bb_lower = prev_bb_middle - (prev_bb_stddev * 2)

                if prev_bb_middle != 0:
                    prev_bb_width = (prev_bb_upper - prev_bb_lower) / prev_bb_middle
                else:
                    prev_bb_width = None

        # Volume indicators
        volume_ratio = None

        if len(data) >= 20:
            volume_ema20 = float(data['Volume'].ewm(span=20, adjust=False).mean().iloc[-1])
            if volume_ema20 > 0:
                volume_ratio = float(data['Volume'].iloc[-1]) / volume_ema20

        # Get sentiment analysis for the stock
        sentiment_score = 0
        sentiment_signals = []
        has_sentiment = False

        # MARKET-BASED SENTIMENT (works for both data sources)
        # 1. Price momentum as sentiment
        if len(data) >= 6:  # Make sure we have enough data for 5-day momentum
            price_5d_ago = float(data['Close'].iloc[-6])
            current_close = float(data['Close'].iloc[-1])
            recent_momentum = ((current_close / price_5d_ago) - 1) * 100

            if recent_momentum > 3:
                sentiment_score += 1
                sentiment_signals.append(f"Strong positive price momentum: +{recent_momentum:.1f}% (bullish)")
                has_sentiment = True
            elif recent_momentum < -3:
                sentiment_score -= 1
                sentiment_signals.append(f"Strong negative price momentum: {recent_momentum:.1f}% (bearish)")
                has_sentiment = True

        # 2. Volume analysis as sentiment
        if volume_ratio is not None and volume_ratio > 2.0:
            # Significant volume spike
            if len(data) >= 2:
                prev_close = float(data['Close'].iloc[-2])
                curr_close = float(data['Close'].iloc[-1])
                price_change = curr_close - prev_close

                if price_change > 0:
                    # Up day on high volume
                    sentiment_score += 0.5
                    sentiment_signals.append(f"High volume on price increase: {volume_ratio:.1f}x avg (bullish)")
                    has_sentiment = True
                elif price_change < 0:
                    # Down day on high volume
                    sentiment_score -= 0.5
                    sentiment_signals.append(f"High volume on price decrease: {volume_ratio:.1f}x avg (bearish)")
                    has_sentiment = True

        # 3. Longer-term trend analysis
        if len(data) >= 20:
            # Calculate trend strength
            price_20d_ago = float(data['Close'].iloc[-20])
            current_close = float(data['Close'].iloc[-1])
            trend_strength = ((current_close / price_20d_ago) - 1) * 100

            if trend_strength > 10:  # Strong uptrend
                sentiment_score += 1
                sentiment_signals.append(f"Strong positive trend: +{trend_strength:.1f}% over 20 days (bullish)")
                has_sentiment = True
            elif trend_strength < -10:  # Strong downtrend
                sentiment_score -= 1
                sentiment_signals.append(f"Strong negative trend: {trend_strength:.1f}% over 20 days (bearish)")
                has_sentiment = True

        # YAHOO-SPECIFIC SENTIMENT
        if data_source == 'yahoo':
            # Try to get recommendation data from Yahoo Finance info
            try:
                if isinstance(stock_info, dict):
                    # Check recommendation fields
                    if 'recommendationKey' in stock_info and stock_info['recommendationKey'] is not None:
                        rec_value = stock_info['recommendationKey']

                        # Text-based recommendation
                        if isinstance(rec_value, str):
                            if rec_value.lower() in ['buy', 'strong_buy']:
                                sentiment_score += 1.5
                                sentiment_signals.append(f"Analyst consensus: {rec_value} (bullish)")
                                has_sentiment = True
                            elif rec_value.lower() in ['sell', 'strong_sell', 'underperform']:
                                sentiment_score -= 1.5
                                sentiment_signals.append(f"Analyst consensus: {rec_value} (bearish)")
                                has_sentiment = True
                            elif rec_value.lower() in ['hold', 'neutral']:
                                sentiment_signals.append(f"Analyst consensus: {rec_value} (neutral)")
                                has_sentiment = True

                    if 'recommendationMean' in stock_info and stock_info['recommendationMean'] is not None:
                        rec_mean = stock_info['recommendationMean']

                        # Numeric recommendation (1=Strong Buy to 5=Strong Sell)
                        if isinstance(rec_mean, (int, float)) and 1 <= rec_mean <= 5:
                            if rec_mean < 2.5:
                                sentiment_score += (2.5 - rec_mean)
                                sentiment_signals.append(f"Analyst rating: {rec_mean:.1f}/5.0 (bullish)")
                                has_sentiment = True
                            elif rec_mean > 3.5:
                                sentiment_score -= (rec_mean - 3.5)
                                sentiment_signals.append(f"Analyst rating: {rec_mean:.1f}/5.0 (bearish)")
                                has_sentiment = True
                            else:
                                sentiment_signals.append(f"Analyst rating: {rec_mean:.1f}/5.0 (neutral)")
                                has_sentiment = True
            except Exception as e:
                print(f"Error retrieving Yahoo recommendation data: {e}")



        # ----- ANALYSIS SYSTEM -----
        # Initialize score
        score = 0
        signals = []

        # 1. Trend Analysis
        # Check if price is above major moving averages (bullish)
        if sma50 is not None:
            if current_price > sma50:
                score += 1
                signals.append("Price above 50-day SMA (bullish)")
            else:
                score -= 1
                signals.append("Price below 50-day SMA (bearish)")

        if sma200 is not None:
            if current_price > sma200:
                score += 1.5  # Stronger weight for longer-term trend
                signals.append("Price above 200-day SMA (strongly bullish)")
            else:
                score -= 1.5
                signals.append("Price below 200-day SMA (strongly bearish)")
        else:
            # If SMA200 is not available, use SMA50 with less weight
            if sma50 is not None and current_price > sma50:
                score += 0.75
                signals.append("Price above 50-day SMA (bullish trend)")
            elif sma50 is not None:
                score -= 0.75
                signals.append("Price below 50-day SMA (bearish trend)")

        # 2. Moving Average Crossovers
        # Only check for Golden Cross or Death Cross if we have SMA200
        if sma200 is not None and len(data) >= 2:
            # Get yesterday's values
            prev_sma50 = float(data['Close'].rolling(window=50).mean().iloc[-2])
            prev_sma200 = float(data['Close'].rolling(window=200).mean().iloc[-2])

            # Golden Cross (50 SMA crosses above 200 SMA)
            if (sma50 > sma200) and (prev_sma50 <= prev_sma200):
                score += 2
                signals.append("Golden Cross detected (strongly bullish)")

            # Death Cross (50 SMA crosses below 200 SMA)
            if (sma50 < sma200) and (prev_sma50 >= prev_sma200):
                score -= 2
                signals.append("Death Cross detected (strongly bearish)")

        # 3. Momentum Indicators - RSI
        if rsi is not None:
            # Oversold conditions (RSI below 30)
            if rsi < 30:
                score += 1
                signals.append(f"RSI at {rsi:.1f}: oversold (bullish)")
            # Overbought conditions (RSI above 70)
            elif rsi > 70:
                score -= 1
                signals.append(f"RSI at {rsi:.1f}: overbought (bearish)")
            # Neutral RSI but trending in bullish zone (40-60)
            elif 40 <= rsi <= 60:
                # In bullish trend, this is healthy
                if sma50 is not None and current_price > sma50:
                    score += 0.5
                    signals.append(f"RSI at {rsi:.1f}: healthy in bullish trend")

        # 4. MACD Analysis
        if macd is not None and macd_signal is not None and len(data) >= 2:
            # Get previous values
            prev_macd = float(data['Close'].ewm(span=12, adjust=False).mean().iloc[-2]) - float(
                data['Close'].ewm(span=26, adjust=False).mean().iloc[-2])
            prev_signal = float(
                (data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()).ewm(
                    span=9, adjust=False).mean().iloc[-2])

            # MACD crossing above signal line (bullish)
            if (macd > macd_signal) and (prev_macd <= prev_signal):
                score += 1.5
                signals.append("MACD crossed above signal line (bullish)")

            # MACD crossing below signal line (bearish)
            if (macd < macd_signal) and (prev_macd >= prev_signal):
                score -= 1.5
                signals.append("MACD crossed below signal line (bearish)")

        # 5. Bollinger Bands
        if bb_lower is not None and bb_upper is not None:
            # Price near lower band (potential bounce)
            if current_price < (bb_lower * 1.02):  # Within 2% of lower band
                score += 1
                signals.append("Price near lower Bollinger Band (potential bounce)")

            # Price near upper band and RSI high (potential reversal)
            if current_price > (bb_upper * 0.98) and rsi is not None and rsi > 70:
                score -= 1
                signals.append("Price near upper Bollinger Band with high RSI (potential reversal)")

        # 6. Volume Analysis
        if volume_ratio is not None and volume_ratio > 1.5:
            # Higher than average volume (confirms moves)
            if len(data) >= 2:
                prev_close = float(data['Close'].iloc[-2])
                curr_close = float(data['Close'].iloc[-1])

                # High volume on up day (bullish)
                if curr_close > prev_close:
                    score += 1
                    signals.append(f"High volume on up day (bullish confirmation): {volume_ratio:.1f}x avg")
                # High volume on down day (bearish)
                elif curr_close < prev_close:
                    score -= 1
                    signals.append(f"High volume on down day (bearish confirmation): {volume_ratio:.1f}x avg")

        # 7. Recent Performance (short-term momentum)
        # Calculate 5-day return
        if len(data) >= 6:
            price_5d_ago = float(data['Close'].iloc[-6])
            five_day_return = ((current_price / price_5d_ago) - 1) * 100

            if five_day_return > 5:  # 5% gain in a week is significant
                score += 1
                signals.append(f"Strong 5-day momentum: +{five_day_return:.1f}%")
            elif five_day_return < -5:  # 5% loss in a week is concerning
                score -= 1
                signals.append(f"Weak 5-day momentum: {five_day_return:.1f}%")

        # 8. Volatility Assessment using Bollinger Band width
        if bb_width is not None and len(data) >= 41:
            # Calculate previous BB width
            prev_20d = data.iloc[-41:-21]
            if len(prev_20d) >= 20:
                prev_bb_middle = float(prev_20d['Close'].mean())
                prev_bb_stddev = float(prev_20d['Close'].std())
                prev_bb_upper = prev_bb_middle + (prev_bb_stddev * 2)
                prev_bb_lower = prev_bb_middle - (prev_bb_stddev * 2)

                if prev_bb_middle != 0:
                    prev_bb_width = (prev_bb_upper - prev_bb_lower) / prev_bb_middle

                    # Contracting volatility after high volatility can indicate consolidation before next move
                    if bb_width < prev_bb_width * 0.7:  # 30% reduction in volatility
                        if sma50 is not None and current_price > sma50:  # In uptrend
                            score += 0.5
                            signals.append("Decreasing volatility in uptrend (potential continuation)")

        # 9. Add sentiment analysis if we have it
        if sentiment_score != 0:
            score += sentiment_score
            signals.extend(sentiment_signals)

        # ----- DETERMINE RECOMMENDATION -----
        # Generate recommendation based on score
        if score >= 3:
            recommendation = "Strong Buy"
            reason = "Multiple strong bullish indicators present"
        elif score >= 1:
            recommendation = "Buy"
            reason = "Moderately bullish signals with positive momentum"
        elif score >= -1:
            recommendation = "Hold"
            reason = "Mixed signals with no clear direction"
        elif score >= -3:
            recommendation = "Sell"
            reason = "Moderately bearish signals with negative momentum"
        else:
            recommendation = "Strong Sell"
            reason = "Multiple strong bearish indicators present"

        # Clean indicators for the response
        clean_indicators = {
            "SMA50": sma50 if sma50 is not None else 0,
            "SMA200": sma200 if sma200 is not None else 0,
            "RSI": rsi if rsi is not None else 0,
            "MACD": macd if macd is not None else 0,
            "MACD_Signal": macd_signal if macd_signal is not None else 0,
            "BB_Upper": bb_upper if bb_upper is not None else 0,
            "BB_Lower": bb_lower if bb_lower is not None else 0,
            "BB_Width": bb_width if bb_width is not None else 0,
            "Volume_Ratio": volume_ratio if volume_ratio is not None else 0,
            "Sentiment": sentiment_score
        }

        # Return detailed results
        return {
            "ticker": ticker,
            "company": company_name,
            "price": current_price,
            "recommendation": recommendation,
            "score": float(score),
            "reason": reason,
            "signals": signals,
            "indicators": clean_indicators,
            "has_sentiment": has_sentiment,
            "data_source": data_source
        }

    except Exception as e:
        traceback.print_exc()
        return {"recommendation": "Error", "reason": f"Error generating recommendation: {str(e)}", "score": 0}


def stock_analysis(ticker, data_source='yahoo'):
    st.subheader(f"Analysis for {ticker}")

    try:
        # Generate recommendation
        with st.spinner(f"Analyzing {ticker}..."):
            recommendation = generate_recommendation(ticker, data_source)

        if recommendation.get("recommendation") == "Error":
            st.error(recommendation.get("reason", "Error generating recommendation"))
            return

        # Display recommendation
        rec_color = {
            "Strong Buy": "green",
            "Buy": "lightgreen",
            "Hold": "gray",
            "Sell": "orange",
            "Strong Sell": "red",
            "No Data": "gray",
            "Error": "red"
        }

        color = rec_color.get(recommendation["recommendation"], "gray")

        # Create columns for layout
        col1, col2 = st.columns([1, 1])

        # Display recommendation and price
        with col1:
            st.markdown(f"### Recommendation: <span style='color:{color}'>{recommendation['recommendation']}</span>",
                        unsafe_allow_html=True)
            st.write(f"**Reason:** {recommendation['reason']}")

        with col2:
            st.metric("Current Price", f"${recommendation.get('price', 0):.2f}")
            st.metric("Analysis Score", f"{recommendation.get('score', 0):.1f}",
                      delta="Bullish" if recommendation.get('score', 0) > 0 else "Bearish" if recommendation.get(
                          'score', 0) < 0 else "Neutral")

            # Show data source
            source_name = "Yahoo Finance" if recommendation.get("data_source") == "yahoo" else "Alpha Vantage"
            st.write(f"Data source: **{source_name}**")

        # Display signal details
        if "signals" in recommendation and recommendation["signals"]:
            # Find sentiment signals (those that contain specific keywords)
            sentiment_signals = []
            technical_signals = []

            sentiment_keywords = [
                "sentiment", "analyst", "consensus", "rating",
                "momentum: +", "momentum:", "trend:", "volume on price"
            ]

            for signal in recommendation["signals"]:
                is_sentiment = False
                for keyword in sentiment_keywords:
                    if keyword in signal.lower():
                        is_sentiment = True
                        break

                if is_sentiment:
                    sentiment_signals.append(signal)
                else:
                    technical_signals.append(signal)

            # Show technical signals
            st.subheader("Technical Signals")

            # Process the technical signals
            bullish_signals = [s for s in technical_signals if "bullish" in s.lower()]
            bearish_signals = [s for s in technical_signals if "bearish" in s.lower()]
            neutral_signals = [s for s in technical_signals if
                               "bullish" not in s.lower() and "bearish" not in s.lower()]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Bullish Signals")
                if bullish_signals:
                    for signal in bullish_signals:
                        st.markdown(f"✅ {signal}")
                else:
                    st.write("No bullish signals detected")

                st.markdown("#### Neutral Signals")
                if neutral_signals:
                    for signal in neutral_signals:
                        st.markdown(f"➖ {signal}")

            with col2:
                st.markdown("#### Bearish Signals")
                if bearish_signals:
                    for signal in bearish_signals:
                        st.markdown(f"❌ {signal}")
                else:
                    st.write("No bearish signals detected")

            # Show sentiment signals separately if any
            if sentiment_signals:
                st.subheader("Market Sentiment Analysis")

                bull_sentiment = [s for s in sentiment_signals if "bullish" in s.lower()]
                bear_sentiment = [s for s in sentiment_signals if "bearish" in s.lower()]
                neutral_sentiment = [s for s in sentiment_signals if
                                     "bullish" not in s.lower() and "bearish" not in s.lower()]

                for signal in bull_sentiment:
                    st.markdown(f"✅ {signal}")

                for signal in neutral_sentiment:
                    st.markdown(f"➖ {signal}")

                for signal in bear_sentiment:
                    st.markdown(f"❌ {signal}")

        # Display indicator values
        if "indicators" in recommendation:
            st.subheader("Technical Indicators")

            ind = recommendation["indicators"]

            # Create 3 columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("RSI (14)", f"{ind.get('RSI', 0):.1f}",
                          delta="Oversold" if ind.get('RSI', 0) < 30 else "Overbought" if ind.get('RSI',
                                                                                                  0) > 70 else "Neutral")

                # Only show valid SMA50
                if ind.get('SMA50', 0) > 0:
                    st.metric("SMA 50", f"${ind.get('SMA50', 0):.2f}",
                              delta=f"{(recommendation.get('price', 0) / ind.get('SMA50', 1) - 1) * 100:.1f}%")
                else:
                    st.metric("SMA 50", "Not Available")

                st.metric("Volume Ratio", f"{ind.get('Volume_Ratio', 0):.2f}x",
                          delta="Above Average" if ind.get('Volume_Ratio', 0) > 1 else "Below Average")

            with col2:
                macd_val = ind.get('MACD', 0)
                macd_signal = ind.get('MACD_Signal', 0)
                macd_delta = macd_val - macd_signal

                st.metric("MACD", f"{macd_val:.3f}",
                          delta=f"{macd_delta:.3f}" if macd_delta != 0 else None)

                # Only show valid SMA200
                if ind.get('SMA200', 0) > 0:
                    st.metric("SMA 200", f"${ind.get('SMA200', 0):.2f}",
                              delta=f"{(recommendation.get('price', 0) / ind.get('SMA200', 1) - 1) * 100:.1f}%")
                else:
                    st.metric("SMA 200", "Not Available")

                # Add sentiment if available
                if ind.get('Sentiment', 0) != 0:
                    sent_val = ind.get('Sentiment', 0)
                    sentiment_text = (
                        "Bullish" if sent_val > 0.5 else
                        "Slightly Bullish" if sent_val > 0 else
                        "Bearish" if sent_val < -0.5 else
                        "Slightly Bearish" if sent_val < 0 else
                        "Neutral"
                    )
                    st.metric("Sentiment Score", f"{sent_val:.2f}", delta=sentiment_text)

            with col3:
                price = recommendation.get('price', 0)
                upper_band = ind.get('BB_Upper', 0)
                lower_band = ind.get('BB_Lower', 0)

                # Calculate percentage distance to bands if valid
                if price > 0 and upper_band > 0:
                    pct_to_upper = ((upper_band / price) - 1) * 100
                    st.metric("BB Upper", f"${upper_band:.2f}", delta=f"{pct_to_upper:.1f}% away")
                else:
                    st.metric("BB Upper", "Not Available")

                if price > 0 and lower_band > 0:
                    pct_to_lower = ((price / lower_band) - 1) * 100
                    st.metric("BB Lower", f"${lower_band:.2f}", delta=f"{pct_to_lower:.1f}% away")
                else:
                    st.metric("BB Lower", "Not Available")

                st.metric("BB Width", f"{ind.get('BB_Width', 0):.3f}")

    except Exception as e:
        st.error(f"Error analyzing {ticker}: {e}")
        st.code(traceback.format_exc())

# Backtesting function with ML
def backtest_strategy(ticker, data_source, lookback_days=365):
    val_accuracy = 0.0
    ml_used = False
    features=None
    pipeline=None
    try:
        # Get historical data for a longer period (3 years for ML training)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 750)  # Extra 750 days for training

        st.info(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")

        if data_source == 'yahoo':
            historical_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            historical_data = fetch_alpha_vantage_data(ticker)

        if historical_data is None or historical_data.empty:
            st.error(f"No historical data available for {ticker}")
            return (None, None)

        # Initialize results DataFrame
        results = pd.DataFrame(index=historical_data.index)
        results['Open'] = historical_data['Open']
        results['High'] = historical_data['High']
        results['Low'] = historical_data['Low']
        results['Close'] = historical_data['Close']
        results['Volume'] = historical_data['Volume']
        results['Daily_Returns'] = historical_data['Close'].pct_change().fillna(0)

        # ----- FEATURE ENGINEERING -----
        # Basic features
        results['EMA10'] = historical_data['Close'].ewm(span=10, adjust=False).mean()
        results['EMA20'] = historical_data['Close'].ewm(span=20, adjust=False).mean()
        results['EMA50'] = historical_data['Close'].ewm(span=50, adjust=False).mean()
        results['EMA200'] = historical_data['Close'].ewm(span=200, adjust=False).mean()

        # Price relative to moving averages (normalized)
        results['Price_to_EMA10'] = results['Close'] / results['EMA10'] - 1
        results['Price_to_EMA20'] = results['Close'] / results['EMA20'] - 1
        results['Price_to_EMA50'] = results['Close'] / results['EMA50'] - 1
        results['Price_to_EMA200'] = results['Close'] / results['EMA200'] - 1

        # Moving average crossovers
        results['EMA10_cross_EMA20'] = np.where(results['EMA10'] > results['EMA20'], 1, -1)
        results['EMA20_cross_EMA50'] = np.where(results['EMA20'] > results['EMA50'], 1, -1)
        results['EMA50_cross_EMA200'] = np.where(results['EMA50'] > results['EMA200'], 1, -1)

        # RSI (Relative Strength Index)
        delta = results['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.001)  # Avoid division by zero
        results['RSI'] = 100 - (100 / (1 + rs))

        # Volatility
        results['Volatility_20d'] = results['Daily_Returns'].rolling(window=20).std()

        # Returns over different periods
        for period in [5, 10, 20, 60]:
            results[f'Return_{period}d'] = results['Close'].pct_change(periods=period)

        # Volume indicators
        results['Volume_EMA20'] = results['Volume'].ewm(span=20, adjust=False).mean()
        results['Volume_Ratio'] = results['Volume'] / results['Volume_EMA20']

        # Bollinger Bands
        results['BB_Middle'] = results['Close'].rolling(window=20).mean()
        results['BB_StdDev'] = results['Close'].rolling(window=20).std()
        results['BB_Upper'] = results['BB_Middle'] + (results['BB_StdDev'] * 2)
        results['BB_Lower'] = results['BB_Middle'] - (results['BB_StdDev'] * 2)
        # BB position (where is price within the bands, normalized 0-1)
        bb_width = results['BB_Upper'] - results['BB_Lower']
        results['BB_Position'] = (results['Close'] - results['BB_Lower']) / (bb_width.replace(0, 0.001))

        # MACD
        results['MACD'] = results['EMA10'] - results['EMA20']
        results['MACD_Signal'] = results['MACD'].ewm(span=9, adjust=False).mean()
        results['MACD_Hist'] = results['MACD'] - results['MACD_Signal']

        # Momentum indicators
        results['ROC_10'] = results['Close'].pct_change(10) * 100  # Rate of Change

        # Target variable: Will price be higher in 5 days? (1=yes, 0=no)
        results['Target'] = (results['Close'].shift(-5) > results['Close']).astype(int)

        # Drop NaN values for ML
        ml_data = results.dropna()

        # Check if we have enough data for ML
        if len(ml_data) > 300:  # Need at least 300 days of data
            # Select features for ML
            features = [
                'Price_to_EMA10', 'Price_to_EMA20', 'Price_to_EMA50', 'Price_to_EMA200',
                'EMA10_cross_EMA20', 'EMA20_cross_EMA50', 'EMA50_cross_EMA200',
                'RSI', 'Volatility_20d', 'Return_5d', 'Return_10d', 'Return_20d', 'Return_60d',
                'Volume_Ratio', 'BB_Position', 'MACD_Hist', 'ROC_10'
            ]
        try:
            # Evaluate on validation set
            val_accuracy = pipeline.score(X_val, y_val)
            st.info(f"ML model validation accuracy: {val_accuracy:.2f}")
        except:
            val_accuracy = 0.0

            X = ml_data[features]
            y = ml_data['Target']

            # Split data: training (70%), validation (30%)
            # We'll use time-based split since this is time series data
            split_idx = int(len(ml_data) * 0.7)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            # Create a pipeline with preprocessing and model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    n_estimators=200,  # More trees for better performance
                    max_depth=8,  # Deeper trees to capture complex patterns
                    min_samples_split=10,
                    random_state=42,
                    class_weight='balanced'  # Handle any class imbalance
                ))
            ])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Evaluate on validation set
            val_accuracy = pipeline.score(X_val, y_val)
            st.info(f"ML model validation accuracy: {val_accuracy:.2f}")

            # Generate predictions for the entire dataset
            ml_data['ML_Probability'] = pipeline.predict_proba(ml_data[features])[:, 1]

            # Generate signals based on ML predictions
            ml_data['Position'] = 0.0
            ml_data['Signal'] = 0.0

            # Higher threshold for buying to increase confidence
            buy_threshold = 0.65
            # Lower threshold for selling to lock in profits
            sell_threshold = 0.35

            # Generate signals
            for i in range(1, len(ml_data)):
                prev_position = ml_data['Position'].iloc[i - 1]

                # Buy signal
                if ml_data['ML_Probability'].iloc[i] > buy_threshold and prev_position == 0:
                    ml_data.iloc[i, ml_data.columns.get_loc('Position')] = 1.0
                    ml_data.iloc[i, ml_data.columns.get_loc('Signal')] = 1.0
                    print(
                        f"ML BUY at {ml_data.index[i]}: Prob={ml_data['ML_Probability'].iloc[i]:.2f}, Price={ml_data['Close'].iloc[i]:.2f}")

                # Sell signal
                elif ml_data['ML_Probability'].iloc[i] < sell_threshold and prev_position > 0:
                    ml_data.iloc[i, ml_data.columns.get_loc('Position')] = 0.0
                    ml_data.iloc[i, ml_data.columns.get_loc('Signal')] = -1.0
                    print(
                        f"ML SELL at {ml_data.index[i]}: Prob={ml_data['ML_Probability'].iloc[i]:.2f}, Price={ml_data['Close'].iloc[i]:.2f}")

                # Trailing stop logic (exit if we've lost more than 5% from peak)
                elif prev_position > 0:
                    # Find the highest price since entry
                    entry_signals = ml_data['Signal'][:i]
                    if sum(entry_signals == 1) > 0:  # If we have any buy signals
                        last_entry_idx = entry_signals[entry_signals == 1].index[-1]
                        highest_price_since_entry = ml_data.loc[last_entry_idx:ml_data.index[i], 'Close'].max()
                        current_price = ml_data['Close'].iloc[i]

                        # Check for trailing stop
                        if current_price < highest_price_since_entry * 0.95:  # 5% trailing stop
                            ml_data.iloc[i, ml_data.columns.get_loc('Position')] = 0.0
                            ml_data.iloc[i, ml_data.columns.get_loc('Signal')] = -1.0
                            print(
                                f"ML TRAILING STOP at {ml_data.index[i]}: Price={current_price:.2f}, High={highest_price_since_entry:.2f}")
                        else:
                            ml_data.iloc[i, ml_data.columns.get_loc('Position')] = prev_position
                            ml_data.iloc[i, ml_data.columns.get_loc('Signal')] = 0.0
                    else:
                        ml_data.iloc[i, ml_data.columns.get_loc('Position')] = prev_position
                        ml_data.iloc[i, ml_data.columns.get_loc('Signal')] = 0.0
                else:
                    ml_data.iloc[i, ml_data.columns.get_loc('Position')] = prev_position
                    ml_data.iloc[i, ml_data.columns.get_loc('Signal')] = 0.0

            # Calculate returns
            ml_data['Strategy_Returns'] = ml_data['Position'].shift(1) * ml_data['Daily_Returns']
            ml_data['Strategy_Returns'] = ml_data['Strategy_Returns'].fillna(0)

            # Calculate equity curves
            initial_capital = 10000
            ml_data['Strategy_Equity'] = initial_capital * (1 + ml_data['Strategy_Returns']).cumprod()
            ml_data['Buy_Hold_Equity'] = initial_capital * (1 + ml_data['Daily_Returns']).cumprod()

            # Calculate drawdowns
            ml_data['Strategy_Peak'] = ml_data['Strategy_Equity'].cummax()
            ml_data['Buy_Hold_Peak'] = ml_data['Buy_Hold_Equity'].cummax()
            ml_data['Strategy_Drawdown'] = (ml_data['Strategy_Equity'] / ml_data['Strategy_Peak'] - 1) * 100
            ml_data['Buy_Hold_Drawdown'] = (ml_data['Buy_Hold_Equity'] / ml_data['Buy_Hold_Peak'] - 1) * 100

            # Copy ML results back to the main results dataframe
            common_indices = results.index.intersection(ml_data.index)

            # Initialize columns in results
            results['ML_Probability'] = None
            results['Position'] = 0.0
            results['Signal'] = 0.0
            results['Strategy_Returns'] = 0.0
            results['Strategy_Equity'] = None
            results['Buy_Hold_Equity'] = None
            results['Strategy_Peak'] = None
            results['Buy_Hold_Peak'] = None
            results['Strategy_Drawdown'] = None
            results['Buy_Hold_Drawdown'] = None

            # Copy values from ml_data to results
            results.loc[common_indices, 'ML_Probability'] = ml_data.loc[common_indices, 'ML_Probability']
            results.loc[common_indices, 'Position'] = ml_data.loc[common_indices, 'Position']
            results.loc[common_indices, 'Signal'] = ml_data.loc[common_indices, 'Signal']
            results.loc[common_indices, 'Strategy_Returns'] = ml_data.loc[common_indices, 'Strategy_Returns']
            results.loc[common_indices, 'Strategy_Equity'] = ml_data.loc[common_indices, 'Strategy_Equity']
            results.loc[common_indices, 'Buy_Hold_Equity'] = ml_data.loc[common_indices, 'Buy_Hold_Equity']
            results.loc[common_indices, 'Strategy_Peak'] = ml_data.loc[common_indices, 'Strategy_Peak']
            results.loc[common_indices, 'Buy_Hold_Peak'] = ml_data.loc[common_indices, 'Buy_Hold_Peak']
            results.loc[common_indices, 'Strategy_Drawdown'] = ml_data.loc[common_indices, 'Strategy_Drawdown']
            results.loc[common_indices, 'Buy_Hold_Drawdown'] = ml_data.loc[common_indices, 'Buy_Hold_Drawdown']

            # Fill any NaN values in equity curves
            results['Strategy_Equity'] = results['Strategy_Equity'].fillna(method='ffill').fillna(initial_capital)
            results['Buy_Hold_Equity'] = results['Buy_Hold_Equity'].fillna(method='ffill').fillna(initial_capital)
            results['Strategy_Peak'] = results['Strategy_Peak'].fillna(method='ffill').fillna(initial_capital)
            results['Buy_Hold_Peak'] = results['Buy_Hold_Peak'].fillna(method='ffill').fillna(initial_capital)
            results['Strategy_Drawdown'] = results['Strategy_Drawdown'].fillna(0)
            results['Buy_Hold_Drawdown'] = results['Buy_Hold_Drawdown'].fillna(0)

            ml_used = True

        else:
            # Not enough data for ML, fall back to traditional strategy
            st.warning(f"Not enough data for ML model. Using traditional strategy instead.")

            # Initialize columns
            results['Position'] = 0.0
            results['Signal'] = 0.0

            # Simple EMA crossover strategy with improved parameters for tech stocks
            fast_ema = 'EMA20'
            slow_ema = 'EMA50'

            # Generate signals
            for i in range(1, len(results)):
                if pd.isna(results[fast_ema].iloc[i]) or pd.isna(results[slow_ema].iloc[i]):
                    continue

                prev_position = results['Position'].iloc[i - 1]

                # Buy signal: fast EMA crosses above slow EMA
                if (results[fast_ema].iloc[i] > results[slow_ema].iloc[i] and
                        results[fast_ema].iloc[i - 1] <= results[slow_ema].iloc[i - 1]):
                    results.iloc[i, results.columns.get_loc('Position')] = 1.0
                    results.iloc[i, results.columns.get_loc('Signal')] = 1.0
                    print(f"BUY at {results.index[i]}: Price={results['Close'].iloc[i]:.2f}")

                # Sell signal: fast EMA crosses below slow EMA
                elif (results[fast_ema].iloc[i] < results[slow_ema].iloc[i] and
                      results[fast_ema].iloc[i - 1] >= results[slow_ema].iloc[i - 1] and
                      prev_position > 0):
                    results.iloc[i, results.columns.get_loc('Position')] = 0.0
                    results.iloc[i, results.columns.get_loc('Signal')] = -1.0
                    print(f"SELL at {results.index[i]}: Price={results['Close'].iloc[i]:.2f}")

                # Hold position
                else:
                    results.iloc[i, results.columns.get_loc('Position')] = prev_position
                    results.iloc[i, results.columns.get_loc('Signal')] = 0.0

            # Calculate returns
            results['Strategy_Returns'] = results['Position'].shift(1) * results['Daily_Returns']
            results['Strategy_Returns'] = results['Strategy_Returns'].fillna(0)

            # Calculate equity curves
            initial_capital = 10000
            results['Strategy_Equity'] = initial_capital * (1 + results['Strategy_Returns']).cumprod()
            results['Buy_Hold_Equity'] = initial_capital * (1 + results['Daily_Returns']).cumprod()

            # Calculate drawdowns
            results['Strategy_Peak'] = results['Strategy_Equity'].cummax()
            results['Buy_Hold_Peak'] = results['Buy_Hold_Equity'].cummax()
            results['Strategy_Drawdown'] = (results['Strategy_Equity'] / results['Strategy_Peak'] - 1) * 100
            results['Buy_Hold_Drawdown'] = (results['Buy_Hold_Equity'] / results['Buy_Hold_Peak'] - 1) * 100

            ml_used = False

        # Build trade log
        buy_signals = results[results['Signal'] == 1]
        sell_signals = results[results['Signal'] == -1]

        # Add final exit if needed
        if len(buy_signals) > len(sell_signals):
            # Add final exit at last date
            last_row_idx = results.index[-1]
            results.loc[last_row_idx, 'Signal'] = -1.0
            sell_signals = results[results['Signal'] == -1]

        # Create trade log
        trade_log = []

        entry_dates = buy_signals.index.tolist()
        exit_dates = sell_signals.index.tolist()

        if entry_dates and exit_dates:
            entry_dates.sort()
            exit_dates.sort()

            entry_idx = 0
            exit_idx = 0

            while entry_idx < len(entry_dates) and exit_idx < len(exit_dates):
                entry_date = entry_dates[entry_idx]
                exit_date = exit_dates[exit_idx]

                if exit_date > entry_date:
                    entry_price = results.loc[entry_date, 'Close']
                    exit_price = results.loc[exit_date, 'Close']

                    trade_return = ((exit_price / entry_price) - 1) * 100

                    trade_log.append({
                        'Entry Date': entry_date,
                        'Exit Date': exit_date,
                        'Entry Price': float(entry_price),
                        'Exit Price': float(exit_price),
                        'Duration (days)': (exit_date - entry_date).days,
                        'Return (%)': float(trade_return),
                        'Exit Reason': 'ML Signal' if ml_used else 'Signal'
                    })

                    entry_idx += 1
                    exit_idx += 1
                else:
                    exit_idx += 1

        trade_log_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

        # Filter results to lookback period
        backtest_period = min(lookback_days, len(results))
        backtest_results = results.iloc[-backtest_period:]

        # Calculate metrics
        if trade_log:
            # Win/loss statistics
            returns = [t['Return (%)'] for t in trade_log]
            winning_trades = sum(1 for r in returns if r > 0)
            losing_trades = sum(1 for r in returns if r <= 0)

            win_rate = winning_trades / len(returns) if returns else 0

            win_returns = [r for r in returns if r > 0]
            loss_returns = [r for r in returns if r <= 0]

            avg_win = (sum(win_returns) / winning_trades) / 100 if winning_trades else 0
            avg_loss = (sum(loss_returns) / losing_trades) / 100 if losing_trades else 0

            win_sum = sum(win_returns)
            loss_sum = abs(sum(loss_returns)) if loss_returns else 0

            profit_factor = win_sum / loss_sum if loss_sum > 0 else float('inf')

            # Get first and last valid equity values
            start_equity = backtest_results['Strategy_Equity'].iloc[0]
            end_equity = backtest_results['Strategy_Equity'].iloc[-1]

            # Calculate strategy return
            strategy_return = ((end_equity / start_equity) - 1) * 100

            # Buy & Hold return
            bh_start = backtest_results['Buy_Hold_Equity'].iloc[0]
            bh_end = backtest_results['Buy_Hold_Equity'].iloc[-1]
            buy_hold_return = ((bh_end / bh_start) - 1) * 100

            # Calculate annual return
            days = (backtest_results.index[-1] - backtest_results.index[0]).days
            annual_return = ((1 + strategy_return / 100) ** (365 / days) - 1) * 100 if days > 0 else 0

            # Calculate Sharpe ratio
            sharpe_ratio = (backtest_results['Strategy_Returns'].mean() / backtest_results[
                'Strategy_Returns'].std()) * np.sqrt(252) if backtest_results['Strategy_Returns'].std() > 0 else 0

            # Calculate maximum drawdown
            max_drawdown = backtest_results['Strategy_Drawdown'].min()

            metrics = {
                'Total Return': float(strategy_return),
                'Annual Return': float(annual_return),
                'Sharpe Ratio': float(sharpe_ratio),
                'Max Drawdown': float(max_drawdown),
                'Win Rate': float(win_rate),
                'Profit Factor': float(profit_factor),
                'Avg Win': float(avg_win),
                'Avg Loss': float(avg_loss),
                'Number of Trades': len(trade_log),
                'Buy & Hold Return': float(buy_hold_return),
                'trade_log': trade_log_df,
                'ML_Used': ml_used,
                'ML_Validation_Accuracy': val_accuracy
            }
        else:
            # If no trades, use simplified metrics
            start_equity = backtest_results['Strategy_Equity'].iloc[0]
            end_equity = backtest_results['Strategy_Equity'].iloc[-1]

            strategy_return = ((end_equity / start_equity) - 1) * 100

            bh_start = backtest_results['Buy_Hold_Equity'].iloc[0]
            bh_end = backtest_results['Buy_Hold_Equity'].iloc[-1]
            buy_hold_return = ((bh_end / bh_start) - 1) * 100

            metrics = {
                'Total Return': float(strategy_return),
                'Annual Return': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Profit Factor': 0.0,
                'Avg Win': 0.0,
                'Avg Loss': 0.0,
                'Number of Trades': 0,
                'Buy & Hold Return': float(buy_hold_return),
                'trade_log': pd.DataFrame(),
                'ML_Used': ml_used,
                'ML_Validation_Accuracy': val_accuracy
            }

        # Add feature importance if ML was used
        if ml_used and len(ml_data) > 300:
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': pipeline.named_steps['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            metrics['feature_importance'] = feature_importance

        return (backtest_results, metrics)

    except Exception as e:
        traceback.print_exc()
        st.error(f"Error in backtesting for {ticker}: {e}")
        return (None, None)


def show_backtest_results(ticker, data_source, lookback_days=365):
    """
    Display enhanced backtest results in the Streamlit app
    """
    st.subheader(f"Backtest Results for {ticker}")

    try:
        with st.spinner(f"Backtesting strategy for {ticker}..."):
            result = backtest_strategy(ticker, data_source, lookback_days)

        if result is None or result[0] is None or result[1] is None:
            st.error(f"Backtesting failed for {ticker}. No results or metrics available.")
            return

        results, metrics = result

        # Check if ML was used
        ml_used = metrics.get('ML_Used', False)
        if ml_used:
            st.success("✅ Machine Learning model was used for this backtest")
        else:
            st.info("ℹ️ Traditional strategy was used (not enough data for ML)")

        # Display the backtest performance
        st.subheader(f"Backtest Performance: {ticker}")

        # Replace NaN values with 0 for display
        for key in metrics:
            if key not in ['trade_log', 'feature_importance'] and isinstance(metrics[key], (int, float)) and np.isnan(
                    metrics[key]):
                metrics[key] = 0.0

        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)

        # First row of metrics
        with col1:
            st.metric("Total Return", f"{metrics['Total Return']:.2f}%")
        with col2:
            st.metric("Annual Return", f"{metrics['Annual Return']:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}%")

        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Win Rate", f"{metrics['Win Rate'] * 100:.1f}%")
        with col2:
            st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
        with col3:
            st.metric("Avg Win", f"{metrics['Avg Win'] * 100:.2f}%")
        with col4:
            st.metric("Avg Loss", f"{metrics['Avg Loss'] * 100:.2f}%")

        # Compare to Buy & Hold
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Strategy Return", f"{metrics['Total Return']:.2f}%")
        with col2:
            performance_delta = metrics['Total Return'] - metrics['Buy & Hold Return']
            st.metric("Buy & Hold Return", f"{metrics['Buy & Hold Return']:.2f}%",
                      delta=f"{performance_delta:.2f}%")

        # Plot the equity curve
        st.subheader("Equity Curves")

        # Check if we have valid equity data
        has_valid_equity = not results['Strategy_Equity'].isna().all() and not results['Buy_Hold_Equity'].isna().all()

        if has_valid_equity:
            # Create a new DataFrame with percentage returns
            equity_analysis = pd.DataFrame(index=results.index)

            # Calculate percentage returns (normalized to 100)
            initial_strategy = results['Strategy_Equity'].iloc[0]
            initial_bah = results['Buy_Hold_Equity'].iloc[0]

            equity_analysis['Strategy'] = (results['Strategy_Equity'] / initial_strategy) * 100
            equity_analysis['Buy_&_Hold'] = (results['Buy_Hold_Equity'] / initial_bah) * 100

            # Calculate relative performance (how much the strategy is outperforming/underperforming)
            equity_analysis['Relative_Performance'] = equity_analysis['Strategy'] - equity_analysis['Buy_&_Hold']

            # Create the figure with two subplots (main chart and relative performance)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.05,
                                row_heights=[0.7, 0.3],
                                subplot_titles=("Percentage Returns (Starting at 100%)",
                                                "Strategy Outperformance vs Buy & Hold"))

            # Add the main equity curves
            fig.add_trace(
                go.Scatter(
                    x=equity_analysis.index,
                    y=equity_analysis['Strategy'],
                    mode='lines',
                    name='Strategy',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=equity_analysis.index,
                    y=equity_analysis['Buy_&_Hold'],
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', width=2, dash='dash')
                ),
                row=1, col=1
            )

            # Add the relative performance
            fig.add_trace(
                go.Scatter(
                    x=equity_analysis.index,
                    y=equity_analysis['Relative_Performance'],
                    mode='lines',
                    name='Outperformance',
                    line=dict(color='green', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.1)'
                ),
                row=2, col=1
            )

            # Add a horizontal line at 0 for the relative performance
            fig.add_shape(
                type="line",
                x0=equity_analysis.index[0],
                y0=0,
                x1=equity_analysis.index[-1],
                y1=0,
                line=dict(color="black", width=1, dash="dash"),
                row=2, col=1
            )

            # Format the figure
            fig.update_layout(
                title=f"{ticker} Strategy vs. Buy & Hold Performance",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=700,
                margin=dict(t=80),
                hovermode="x unified"
            )

            # Update y-axis labels
            fig.update_yaxes(title_text="Percentage Return (%)", row=1, col=1)
            fig.update_yaxes(title_text="Difference (%)", row=2, col=1)

            # Add strategic annotations with performance metrics at key points
            performance_points = {}

            # Find points where strategy outperforms or underperforms significantly
            for i in range(1, len(equity_analysis)):
                if i % (len(equity_analysis) // 10) == 0:  # Add annotations at roughly 10 points
                    date = equity_analysis.index[i]
                    strategy_return = equity_analysis['Strategy'][i] - 100
                    bah_return = equity_analysis['Buy_&_Hold'][i] - 100
                    diff = strategy_return - bah_return

                    performance_points[date] = {
                        'strategy': strategy_return,
                        'bah': bah_return,
                        'diff': diff
                    }

            # Add range slider and buttons
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display performance table at key points
            st.subheader("Performance at Key Points")

            if performance_points:
                perf_df = pd.DataFrame([
                    {
                        'Date': date.strftime('%Y-%m-%d'),
                        'Strategy (%)': f"{data['strategy']:.1f}%",
                        'Buy & Hold (%)': f"{data['bah']:.1f}%",
                        'Difference (%)': f"{data['diff']:.1f}%"
                    }
                    for date, data in performance_points.items()
                ])

                st.dataframe(perf_df)
        else:
            st.warning("Insufficient data to plot equity curves.")

        # Show drawdown chart if available
        if has_valid_equity and not results['Strategy_Drawdown'].isna().all():
            st.subheader("Drawdown Analysis")

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=results.index,
                y=results['Strategy_Drawdown'],
                mode='lines',
                name='Strategy Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ))
            fig_dd.add_trace(go.Scatter(
                x=results.index,
                y=results['Buy_Hold_Drawdown'],
                mode='lines',
                name='Buy & Hold Drawdown',
                line=dict(color='orange', width=2, dash='dash')
            ))

            # Format the drawdown figure
            fig_dd.update_layout(
                title="Drawdown Comparison",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                yaxis=dict(tickformat=".1f"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400
            )

            st.plotly_chart(fig_dd, use_container_width=True)

        # Display feature importance if available
        if 'feature_importance' in metrics and not metrics['feature_importance'].empty:
            st.subheader("ML Feature Importance")

            # Plot feature importance
            fig_imp = px.bar(
                metrics['feature_importance'].head(10),  # Top 10 features
                y='Feature',
                x='Importance',
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale='Blues'
            )

            fig_imp.update_layout(height=400)
            st.plotly_chart(fig_imp, use_container_width=True)

            # Show insights based on feature importance
            top_feature = metrics['feature_importance'].iloc[0]['Feature']
            st.info(f"🔍 Insight: The model relies most heavily on '{top_feature}' when making predictions.")

            if 'RSI' in metrics['feature_importance']['Feature'].values:
                st.info(
                    "📊 Insight: RSI (Relative Strength Index) is a significant factor, suggesting momentum is important for this stock.")

            if any('EMA' in f for f in metrics['feature_importance']['Feature'].values):
                st.info(
                    "📈 Insight: Moving averages play an important role in predicting price movement for this stock.")

        # Display trade log if available
        if 'trade_log' in metrics and not metrics['trade_log'].empty:
            st.subheader("Trade Analysis")

            # Trade statistics
            trade_count = len(metrics['trade_log'])
            avg_duration = metrics['trade_log']['Duration (days)'].mean()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Trades", trade_count)
            with col2:
                st.metric("Win Rate", f"{metrics['Win Rate'] * 100:.1f}%")
            with col3:
                st.metric("Avg Duration", f"{avg_duration:.1f} days")

            # Plot trade returns distribution
            trade_returns = metrics['trade_log']['Return (%)']

            fig_dist = px.histogram(
                trade_returns,
                nbins=20,
                title="Distribution of Trade Returns",
                labels={'value': 'Return (%)'},
                color_discrete_sequence=['blue']
            )

            # Add vertical line at zero
            fig_dist.add_shape(
                type="line",
                x0=0, y0=0, x1=0, y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash")
            )

            st.plotly_chart(fig_dist, use_container_width=True)

            # Show trade log
            st.subheader("Trade Log")
            st.dataframe(metrics['trade_log'])
        else:
            st.info("No trades were executed during the backtest period.")

            # Check for open positions
            if 'trade_log' in metrics and metrics['trade_log'].empty:
                # Check if the last signal in the results is a buy (position=1) with no matching sell
                last_position = results['Position'].iloc[-1]

                if last_position > 0:
                    st.subheader("Currently Open Position")

                    # Find when this position was opened
                    # First, get all signals
                    signals = results[results['Signal'] != 0]

                    # Find the last buy signal
                    last_buy = signals[signals['Signal'] > 0].iloc[-1]
                    last_buy_date = last_buy.name
                    last_buy_price = results.loc[last_buy_date, 'Close']

                    # Current values
                    current_date = results.index[-1]
                    current_price = results.loc[current_date, 'Close']

                    # Calculate metrics
                    days_open = (current_date - last_buy_date).days
                    profit_pct = ((current_price / last_buy_price) - 1) * 100

                    # Create a dataframe for the open position
                    open_position = pd.DataFrame({
                        'Entry Date': [last_buy_date.strftime('%Y-%m-%d')],
                        'Days Open': [days_open],
                        'Entry Price': [f"${last_buy_price:.2f}"],
                        'Current Price': [f"${current_price:.2f}"],
                        'Current P/L': [f"{profit_pct:.2f}%"],
                    })

                    # Display with warning
                    st.warning(
                        "⚠️ This strategy has an open position that hasn't been closed yet. The backtest metrics don't include the unrealized P/L of this position.")
                    st.table(open_position)

    except Exception as e:
        st.error(f"Error displaying backtest results: {e}")
        st.code(traceback.format_exc())

# Helper function to determine recommendation strength
def get_recommendation_strength(score):
    if score >= 3:
        return "Very Bullish"
    elif score >= 1:
        return "Bullish"
    elif score >= -1:
        return "Neutral"
    elif score >= -3:
        return "Bearish"
    else:
        return "Very Bearish"


# Main app UI
def main():
    st.title("Stock Analysis & Trading Strategy Backtester")

    # Sidebar setup
    st.sidebar.title("Settings")

    # Select data source
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["yahoo", "alpha_vantage"]
    )

    # User inputs for single ticker or multiple tickers
    ticker_input_mode = st.sidebar.radio(
        "Ticker Input Mode",
        ["Single Ticker", "Multiple Tickers"]
    )

    if ticker_input_mode == "Single Ticker":
        ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
        tickers = [ticker] if ticker else []
    else:
        tickers_input = st.sidebar.text_area("Enter Stock Tickers (comma or newline separated)",
                                             "AAPL, MSFT, GOOGL, AMZN")
        # Parse tickers from input (accept commas or newlines as separators)
        tickers = [t.strip().upper() for t in re.split(r'[,\n]', tickers_input) if t.strip()]

    # Dashboard navigation
    tab1, tab2 = st.tabs(["Analysis & Recommendation", "Backtest Strategy"])

    with tab1:
        if tickers:
            # For multiple tickers, create a summary table first
            if len(tickers) > 1:
                st.subheader("Multi-Stock Analysis Summary")

                # Create a table to display all recommendations
                summary_data = []

                with st.spinner("Analyzing multiple stocks..."):
                    for ticker in tickers:
                        if ticker:  # Skip empty tickers
                            recommendation = generate_recommendation(ticker, data_source)
                            if recommendation["recommendation"] != "Error" and recommendation[
                                "recommendation"] != "No Data":
                                # Run a quick backtest to get ML accuracy
                                try:
                                    backtest_result = backtest_strategy(ticker, data_source,
                                                                        lookback_days=180)  # Use shorter period for speed

                                    # Get ML accuracy if available
                                    ml_accuracy = 0.0
                                    ml_used = False
                                    if backtest_result is not None and backtest_result[1] is not None:
                                        metrics = backtest_result[1]
                                        ml_accuracy = metrics.get('ML_Validation_Accuracy', 0.0)
                                        ml_used = metrics.get('ML_Used', False)
                                except Exception as e:
                                    # Handle the error gracefully
                                    st.warning(f"Error backtesting {ticker}: {str(e)}")
                                    ml_accuracy = 0.0
                                    ml_used = False

                                # Add to summary
                                summary_data.append({
                                    "Ticker": ticker,
                                    "Price": f"${recommendation['price']:.2f}",
                                    "Recommendation": recommendation["recommendation"],
                                    "Score": f"{recommendation['score']:.1f}",
                                    "Strength": get_recommendation_strength(recommendation["score"]),
                                    "ML Accuracy": f"{ml_accuracy:.2f}" if ml_used else "N/A"
                                })

                # Display the summary table
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)

                    # Apply color formatting
                    def color_recommendations(val):
                        if "Strong Buy" in val:
                            return 'background-color: #90EE90'  # Light green
                        elif "Buy" in val:
                            return 'background-color: #C1FFC1'  # Lighter green
                        elif "Hold" in val:
                            return 'background-color: #F0F0F0'  # Light gray
                        elif "Sell" in val:
                            return 'background-color: #FFB6C1'  # Light pink
                        elif "Strong Sell" in val:
                            return 'background-color: #FF6B6B'  # Darker pink
                        return ''

                    def color_ml_accuracy(val):
                        if val == "N/A":
                            return 'background-color: #F0F0F0'  # Light gray
                        try:
                            acc = float(val)
                            if acc >= 0.60:
                                return 'background-color: #90EE90'  # Light green - excellent
                            elif acc >= 0.55:
                                return 'background-color: #C1FFC1'  # Lighter green - good
                            elif acc >= 0.53:
                                return 'background-color: #FFFACD'  # Light yellow - acceptable
                            elif acc >= 0.50:
                                return 'background-color: #FFB6C1'  # Light pink - poor
                            else:
                                return 'background-color: #FF6B6B'  # Darker pink - very poor
                        except:
                            return ''

                    # Apply styling to both recommendation and ML accuracy columns
                    styled_df = summary_df.style.applymap(color_recommendations, subset=['Recommendation']) \
                        .applymap(color_ml_accuracy, subset=['ML Accuracy'])

                    st.dataframe(styled_df)

                    # Allow user to select a ticker for detailed analysis
                    selected_ticker = st.selectbox("Select a ticker for detailed analysis:", sorted(tickers))
                    if selected_ticker:
                        stock_analysis(selected_ticker, data_source)
                else:
                    st.warning("No valid analysis results found for the tickers provided.")

            # For single ticker, just show detailed analysis
            elif len(tickers) == 1:
                stock_analysis(tickers[0], data_source)

    with tab2:
        # For backtest tab
        lookback_days = st.sidebar.slider("Lookback Period (days)", 30, 365 * 3, 365)

        if ticker_input_mode == "Single Ticker" and tickers:
            show_backtest_results(tickers[0], data_source, lookback_days)
        elif ticker_input_mode == "Multiple Tickers" and tickers:
            selected_ticker = st.selectbox("Select a ticker to backtest:", sorted(tickers))
            if selected_ticker:
                show_backtest_results(selected_ticker, data_source, lookback_days)


# Run the app
if __name__ == "__main__":
    main()