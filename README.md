# 🛒 Woolies Order Analyser

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Woolies Order Analyser is a powerful Streamlit application that helps you gain insights into your Woolworths shopping habits. By analyzing your order history, it provides comprehensive visualizations and statistics about your spending patterns, frequently purchased items, and more.

## 🌟 Features

- 📊 Interactive data visualizations
- 💰 Spending trend analysis
- 🗓️ Monthly spend heatmap
- 🏷️ Category-wise expense breakdown
- 🔝 Top items by cost and frequency
- 🧾 Detailed order breakdowns
- 🔒 Secure handling of user data

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- poetry (Python package manager)
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/wipash/woolies-order-analyser.git
   cd woolies-order-analyser
   ```

2. Install the required dependencies:
   ```
   poetry install
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

To start the Woolies Order Analyser, run:

```
poetry run streamlit run src\woolies_order_analyser\app.py
```

Navigate to the URL provided in the terminal (usually `http://localhost:8501`) to access the application.

## 🔒 Security Note

This application requires your Woolworths website cookie to access your order history. Always keep your cookie private and do not share it with others. The app processes your data locally and does not store or transmit your personal information.

## 📊 How to Use

1. Obtain your Woolworths website cookie:
   - Log in to the Woolworths website
   - Open your browser's developer tools (usually F12)
   - Go to the Network tab and refresh the page
   - Find a request to the Woolworths API and copy the value of the `Cookie` header

2. Paste your cookie into the app when prompted.

3. Select the orders you want to analyze.

4. Click "Proceed" to start the analysis.

5. Explore the various charts and insights provided by the app.

## 🛠️ Technology Stack

- [Python](https://www.python.org/) - Core programming language
- [Streamlit](https://streamlit.io/) - Web application framework
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [Plotly](https://plotly.com/) - Interactive data visualization
- [OpenAI API](https://openai.com/) - PDF data extraction
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Woolworths](https://www.woolworths.co.nz/) for providing the order data
- [OpenAI](https://openai.com/) for their powerful GPT models used in data extraction

---

Made with anxiety about how much I'm spending on groceries by Sean McGrath
