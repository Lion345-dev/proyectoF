# 📈 Análisis Técnico Avanzado

## Descripción
Una aplicación web desarrollada con Streamlit que proporciona análisis técnico en tiempo real de acciones y criptomonedas, utilizando datos de Yahoo Finance y capacidades de IA con Gemini.

## Características Principales
- 📊 Visualización de gráficos de velas (candlesticks)
- 📈 Indicadores técnicos:
  - Media Móvil (MA)
  - Bandas de Bollinger (BB)
  - Índice de Fuerza Relativa (RSI)
  - MACD (Moving Average Convergence Divergence)
- 📱 Métricas en tiempo real
- 🔍 Búsqueda de símbolos bursátiles
- 💼 Información detallada de empresas
- 🤖 Resúmenes automáticos usando IA (Gemini)
- 📰 Noticias recientes
- ⭐ Sistema de favoritos

## Requisitos
```bash
streamlit
python-dotenv
google-generativeai
pandas
numpy
yfinance
plotly
```

## Instalación
1. Clonar el repositorio
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```
3. Crear archivo `.env` en la raíz del proyecto:
```
GEMINI_API_KEY=tu_api_key_aquí
```

## Uso
Para ejecutar la aplicación:
```bash
streamlit run proyectoF.py
```

## Configuración
- La aplicación busca la API key de Gemini en:
  1. Archivo `.env`
  2. Secretos de Streamlit
- Los datos se actualizan automáticamente cada hora
- Caché de lista de tickers S&P 500 por 24 horas

## Características Detalladas
### Análisis Técnico
- Gráficos de velas interactivos
- Múltiples timeframes disponibles
- Indicadores técnicos personalizables
- Análisis de volumen

### Información Fundamental
- Métricas clave de la empresa
- Resúmenes generados por IA
- Noticias recientes
- Datos financieros importantes

### Interfaz de Usuario
- Diseño responsive
- Modo oscuro
- Navegación intuitiva
- Sistema de favoritos y historial

## Notas Legales
- Datos proporcionados por Yahoo Finance
- Solo para propósitos educativos
- No constituye asesoramiento financiero

## Contribuir
Siéntete libre de:
- Reportar bugs
- Sugerir nuevas características
- Enviar pull requests

## Licencia
Este proyecto es para uso educativo y no comercial.