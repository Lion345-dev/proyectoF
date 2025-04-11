import streamlit as st

# Configuración de la página - DEBE SER EL PRIMER COMANDO DE STREAMLIT
st.set_page_config(page_title="Análisis Técnico", layout="wide")

# Resto de imports
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Cargar variables de entorno
load_dotenv()

# Obtener API Key
GEMINI_API_KEY = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception as e:
        st.error("No se encontró la API Key de Gemini")

# Configurar API Key si está disponible
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        st.success("API de Gemini configurada correctamente")
    except Exception as e:
        st.error(f"Error al configurar Gemini: {str(e)}")

# Código de verificación (eliminar en producción)
if st.checkbox("Debug API Key"):
    try:
        if "GEMINI_API_KEY" in st.secrets:
            st.success("API Key encontrada en secretos de Streamlit")
    except AttributeError:
        st.error("API Key no encontrada en secretos de Streamlit")

st.title("📈 Análisis Técnico Avanzado")
st.markdown("""
Herramienta profesional de análisis técnico con datos en tiempo real de Yahoo Finance.
Visualiza indicadores clave, patrones de velas y tendencias del mercado.
""")

# Inicializar estado de sesión
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'search_input' not in st.session_state:
    st.session_state.search_input = ''

# Función para obtener tickers del S&P 500
@st.cache_data(ttl=86400)  # Cache por 24 horas
def get_sp500_tickers():
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-')
        tickers = df['Symbol'].tolist()
        additional_tickers = ['BTC-USD', 'ETH-USD', 'GC=F', 'CL=F', 'EURUSD=X']
        return sorted(tickers + additional_tickers)
    except Exception as e:
        st.error(f"Error al obtener tickers: {str(e)}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JNJ', 'JPM', 'V', 'BTC-USD']

# Función unificada para cargar datos
@st.cache_data(ttl=3600, show_spinner="Obteniendo datos de mercado...")
def load_ticker_data(ticker, period, interval):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            time.sleep(attempt * 2)
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'symbol' not in info:
                raise ValueError(f"No se encontró información para {ticker}")
            
            # Descargar datos según período e intervalo
            data = stock.history(period=period, interval=interval, prepost=True)
            
            if data.empty:
                raise ValueError("No se pudieron obtener datos")
            
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            return data
        
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error al cargar {ticker} tras {max_retries} intentos: {str(e)}")
                return None
            continue
    return None

# Sidebar con configuración
with st.sidebar:
    st.header("Configuración")
    
    ticker_list = get_sp500_tickers()
    try:
        default_index = ticker_list.index('AAPL') if 'AAPL' in ticker_list else 0
    except ValueError:
        default_index = 0
        
    ticker = st.selectbox(
        "Seleccionar activo:",
        options=ticker_list,
        index=default_index,
        help="Escribe para buscar un ticker específico"
    )
    
    # Campo de búsqueda manual
    search_input = st.text_input("Buscar ticker (ej: AAPL, BTC-USD)", value=st.session_state.search_input)
    if search_input and search_input != st.session_state.search_input:
        st.session_state.search_input = search_input
        if search_input.upper() in ticker_list:
            ticker = search_input.upper()
    
    period = st.selectbox(
        "Período histórico:",
        options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=2
    )
    
    interval = st.selectbox(
        "Intervalo:",
        options=['1m', '5m', '15m', '30m', '60m', '1d'],
        index=5
    )
    
    st.markdown("---")
    st.markdown("**Indicadores técnicos**")
    show_ma = st.checkbox("Media Móvil", True)
    show_bb = st.checkbox("Bollinger Bands")
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_volume = st.checkbox("Volumen", True)
    
    if show_ma:
        ma_window = st.slider("Período Media Móvil:", 5, 200, 50)
    if show_bb:
        bb_window = st.slider("Período Bollinger Bands:", 5, 60, 20)
    if show_rsi:
        rsi_window = st.slider("Período RSI:", 5, 30, 14)

# Validar compatibilidad de intervalo y período
if period == '1d' and interval in ['30m', '60m']:
    st.warning("Intervalos de 30m o 60m no son compatibles con período de 1 día. Selecciona un intervalo menor.")
    interval = '15m'
elif period in ['1d', '5d'] and interval == '1d':
    st.warning("Intervalo diario no es compatible con períodos cortos. Selecciona un intervalo intradiario.")
    interval = '15m'

# Cargar datos
data = load_ticker_data(ticker, period, interval)
if data is None or data.empty:
    st.error(f"No se pudieron cargar datos para {ticker}")
else:
    st.success(f"Datos cargados exitosamente para {ticker}")
    
    # Actualizar historial
    if ticker not in st.session_state.history:
        st.session_state.history.append(ticker)

    # Información básica del activo
    col1, col2, col3 = st.columns(3)
    
    try:
        last_close = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else last_close
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100 if prev_close != 0 else 0

        with col1:
            st.metric(
                label=f"Precio {ticker}", 
                value=f"${last_close:,.2f}", 
                delta=f"{change:+,.2f} ({pct_change:+.2f}%)"
            )
    except Exception as e:
        st.error(f"Error al procesar métricas de precio: {str(e)}")
        with col1:
            st.metric(label=f"Precio {ticker}", value="N/A", delta="N/A")
    
    try:
        avg_volume = data['Volume'].mean()
        last_volume = data['Volume'].iloc[-1]
        volume_change = ((last_volume - avg_volume) / avg_volume * 100) if avg_volume != 0 else 0
        
        with col2:
            st.metric(
                label="Volumen", 
                value=f"{int(last_volume):,}", 
                delta=f"{volume_change:+.1f}% vs promedio"
            )
    except Exception as e:
        st.error(f"Error al procesar métricas de volumen: {str(e)}")
        with col2:
            st.metric(label="Volumen", value="N/A", delta="N/A")
    
    try:
        week_high = data['High'].max()
        week_low = data['Low'].min()
        
        with col3:
            st.metric(
                label=f"Rango del período", 
                value=f"${week_low:,.2f} - ${week_high:,.2f}"
            )
    except Exception as e:
        st.error(f"Error al calcular rango de precios: {str(e)}")
        with col3:
            st.metric(label="Rango", value="N/A")

    # Gráfico principal de velas
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, 
        open=data['Open'], 
        high=data['High'], 
        low=data['Low'], 
        close=data['Close'], 
        name=ticker
    ))
    
    fig.update_layout(
        title=f"{ticker} - Gráfico de Velas",
        yaxis_title="Precio (USD)",
        xaxis_title="Fecha",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de rendimientos y riesgo
    st.subheader("Rendimientos y Riesgo")
    
    # Obtener datos diarios para cálculos
    daily_data = data.resample('D').last().dropna() if interval != '1d' else data
    
    if len(daily_data) < 2:
        st.warning("No hay suficientes datos para calcular rendimientos y riesgo (se requiere al menos 2 días).")
    else:
        try:
            # Calcular CAGR
            initial_price = daily_data['Close'].iloc[0]
            final_price = daily_data['Close'].iloc[-1]
            num_days = (daily_data.index[-1] - daily_data.index[0]).days
            num_years = num_days / 365.25 if num_days > 0 else 1
            cagr = ((final_price / initial_price) ** (1 / num_years) - 1) * 100 if num_years > 0 else 0
            
            # Calcular rendimientos acumulados
            cumulative_return = ((final_price - initial_price) / initial_price) * 100
            
            # Calcular rendimientos diarios
            daily_simple_returns = daily_data['Close'].pct_change().dropna() * 100
            daily_log_returns = np.log(daily_data['Close'] / daily_data['Close'].shift(1)).dropna() * 100
            
            # Calcular volatilidad anualizada
            vol_simple = np.std(daily_simple_returns) * np.sqrt(252) if len(daily_simple_returns) > 0 else 0
            vol_log = np.std(daily_log_returns) * np.sqrt(252) if len(daily_log_returns) > 0 else 0
            
            # Desviación estándar diaria
            std_daily_simple = np.std(daily_simple_returns) if len(daily_simple_returns) > 0 else 0
            std_daily_log = np.std(daily_log_returns) if len(daily_log_returns) > 0 else 0
            
            # Crear DataFrame para la tabla
            table_data = {
                'Período': ['Diario', 'Semanal', 'Mensual'],
                'CAGR (%)': [cagr, cagr, cagr],  # CAGR es anual, se repite para consistencia
                'Volatilidad Simple (%)': [vol_simple, vol_simple * np.sqrt(5/252), vol_simple * np.sqrt(21/252)],
                'Volatilidad Logarítmica (%)': [vol_log, vol_log * np.sqrt(5/252), vol_log * np.sqrt(21/252)],
                'Rendimiento Acumulado (%)': [cumulative_return, cumulative_return, cumulative_return],
                'Desv. Estándar Diaria (%)': [std_daily_simple, std_daily_simple, std_daily_simple]
            }
            df_table = pd.DataFrame(table_data)
            
            # Filtro interactivo
            period_filter = st.selectbox("Filtrar por período:", options=['Todos', 'Diario', 'Semanal', 'Mensual'])
            if period_filter != 'Todos':
                df_table = df_table[df_table['Período'] == period_filter]
            
            # Mostrar tabla
            st.dataframe(
                df_table.style.format("{:.2f}", subset=['CAGR (%)', 'Volatilidad Simple (%)', 'Volatilidad Logarítmica (%)', 'Rendimiento Acumulado (%)', 'Desv. Estándar Diaria (%)']),
                use_container_width=True
            )
            
            # Comparar volatilidad con rango
            price_range_pct = ((week_high - week_low) / week_low * 100) if week_low != 0 else 0
            if vol_simple > price_range_pct:
                st.warning(f"La volatilidad simple ({vol_simple:.2f}%) es alta comparada con el rango del período ({price_range_pct:.2f}%).")
            
            # Explicación en contenedor expandible
            with st.expander("📚 Explicación de Rendimientos y Riesgo", expanded=False):
                st.markdown("""
                **Rendimiento Anualizado (CAGR):**  
                El rendimiento anualizado se calculó usando la fórmula de CAGR, que mide el crecimiento compuesto anual del activo. Representa el rendimiento promedio que habría generado la inversión si creciera a una tasa constante durante el período seleccionado. Un CAGR positivo indica crecimiento, mientras que un valor negativo señala una pérdida.

                **Volatilidad Anualizada (Riesgo):**  
                La volatilidad simple ({:.2f}%) representa la desviación estándar de los rendimientos diarios simples, anualizada multiplicando por √252 (días hábiles en un año). La volatilidad logarítmica ({:.2f}%) usa rendimientos logarítmicos, que son más precisos para series temporales financieras. Estos valores indican el riesgo histórico del activo: una volatilidad más alta implica mayores fluctuaciones en el precio.

                **Rendimiento Acumulado y Desviación Estándar Diaria:**  
                El rendimiento acumulado muestra el cambio porcentual total durante el período. La desviación estándar diaria mide la variabilidad promedio de los rendimientos diarios, ofreciendo una visión de la estabilidad del activo a corto plazo.
                """.format(vol_simple, vol_log), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error al calcular rendimientos y riesgo: {str(e)}")

    # Indicadores adicionales
    tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Análisis"])
    
    with tab1:
        if show_rsi:
            st.subheader("Índice de Fuerza Relativa (RSI)")
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
            avg_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(rsi_fig, use_container_width=True)
    
    with tab2:
        if show_macd:
            st.subheader("MACD")
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp12 - exp26
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['Histogram'] = data['MACD'] - data['Signal']
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
            macd_fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], name='Señal'))
            macd_fig.add_trace(go.Bar(x=data.index, y=data['Histogram'], name='Histograma'))
            macd_fig.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(macd_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Análisis Técnico Completo")
        st.write("Resumen de señales técnicas aquí...")  # Simplificado por brevedad

    # Sección adicional con información de la empresa
    stock = yf.Ticker(ticker)
    info = stock.info
    
    if info and 'symbol' in info:
        st.markdown(f"""
            <div class="stock-card">
                <h2 style="color: #E5E9F0; margin:0;">{info.get('longName', ticker)}</h2>
                <p style="color: #D8DEE9; margin:0;">{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📌 Descripción de la Empresa", expanded=True):
            if 'longBusinessSummary' in info and info['longBusinessSummary']:
                prompt = f"Traduce al español y resume en 3 párrafos máximo: {info['longBusinessSummary']}"
                try:
                    if GEMINI_API_KEY:
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(prompt)
                        st.write(response.text)
                    else:
                        st.write(info['longBusinessSummary'])
                except Exception as e:
                    st.warning(f"Error al procesar con Gemini: {str(e)}")
                    st.write(info['longBusinessSummary'])
            else:
                st.warning("Descripción no disponible")
        
        st.markdown("<h3 style='color: #E5E9F0;'>📊 Métricas Clave</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precio Actual", f"${info.get('currentPrice', 'N/A'):,.2f}")
            st.metric("Mín. 52 Sem", f"${info.get('fiftyTwoWeekLow', 'N/A'):,.2f}")
        with col2:
            st.metric("Capitalización", f"${info.get('marketCap', 'N/A'):,.0f}")
            st.metric("Ratio P/E", f"{info.get('trailingPE', 'N/A'):,.1f}")
        with col3:
            st.metric("Máx. 52 Sem", f"${info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}")
            st.metric("Beta", f"{info.get('beta', 'N/A'):,.2f}")
        with col4:
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
            st.metric("Vol. Promedio", f"{info.get('averageVolume', 'N/A'):,}")
        
        st.markdown("<h3 style='color: #E5E9F0;'>📰 Noticias Relevantes</h3>", unsafe_allow_html=True)
        if GEMINI_API_KEY:
            try:
                # Determinar el rango de fechas según el período
                end_date = datetime.now()
                if period == '1d':
                    start_date = end_date - timedelta(days=1)
                elif period == '5d':
                    start_date = end_date - timedelta(days=5)
                elif period == '1mo':
                    start_date = end_date - timedelta(days=30)
                elif period == '3mo':
                    start_date = end_date - timedelta(days=90)
                elif period == '6mo':
                    start_date = end_date - timedelta(days=180)
                elif period == '1y':
                    start_date = end_date - timedelta(days=365)
                elif period == '2y':
                    start_date = end_date - timedelta(days=365*2)
                else:
                    start_date = end_date - timedelta(days=365*5)
                
                prompt = f"""
                Busca y resume en español las noticias más relevantes sobre {info.get('longName', ticker)} 
                entre {start_date.strftime('%Y-%m-%d')} y {end_date.strftime('%Y-%m-%d')}. 
                Incluye máximo 5 noticias, cada una con:
                - Título
                - Fecha (formato DD/MM/YYYY)
                - Resumen breve (2-3 frases)
                - Fuente (si está disponible)
                Ordena por relevancia e impacto en el precio de la acción.
                """
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.warning(f"Error al obtener noticias con Gemini: {str(e)}")
                # Fallback a noticias de yfinance
                news = stock.news
                for item in news[:5]:
                    publish_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%d/%m/%Y %H:%M') if 'providerPublishTime' in item else 'N/A'
                    st.markdown(f"""
                        <div class="news-card">
                            <h4 style="color: #E5E9F0;">{item.get('title', 'Título no disponible')}</h4>
                            <p style="color: #D8DEE9;">{item.get('publisher', 'Fuente desconocida')} - {publish_time}</p>
                            <a href="{item.get('link', '#')}" target="_blank" style="color: #5E81AC;">Leer más →</a>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            news = stock.news
            for item in news[:5]:
                publish_time = datetime.fromtimestamp(item['providerPublishTime']).strftime('%d/%m/%Y %H:%M') if 'providerPublishTime' in item else 'N/A'
                st.markdown(f"""
                    <div class="news-card">
                        <h4 style="color: #E5E9F0;">{item.get('title', 'Título no disponible')}</h4>
                        <p style="color: #D8DEE9;">{item.get('publisher', 'Fuente desconocida')} - {publish_time}</p>
                        <a href="{item.get('link', '#')}" target="_blank" style="color: #5E81AC;">Leer más →</a>
                    </div>
                    """, unsafe_allow_html=True)
        
        if st.button("⭐ Agregar a Favoritos"):
            if ticker not in st.session_state.favorites:
                st.session_state.favorites.append(ticker)
                st.success(f"{ticker} agregado a favoritos!")
            else:
                st.warning(f"{ticker} ya está en tus favoritos")

# Nota legal
st.markdown("---")
st.caption("""
📊 **Nota Legal:** Los datos son proporcionados por Yahoo Finance.  
💡 **Propósito Educativo:** Este análisis no constituye asesoramiento financiero.  
""")