# Fundamental Analyst AI Agent

**MSc Coursework: IFTE0001 - AI Agents in Asset Management**  
**Track A: Fundamental Analyst Agent**

An AI-powered financial analysis agent that performs comprehensive fundamental analysis on publicly traded companies and generates professional investment memoranda.

---

## Overview

This agent simulates the role of a buy-side fundamental analyst by:

1. **Collecting Data**: Ingesting 5 years of financial statements from Yahoo Finance
2. **Processing Data**: Standardizing and validating financial data
3. **Analyzing Profitability**: EBIT variance bridge with volume/margin decomposition
4. **Analyzing Cash Flow**: Net Income to Free Cash Flow bridge
5. **Assessing Earnings Quality**: Accruals analysis and red flag detection
6. **Analyzing Working Capital**: DSO, DIO, DPO, and Cash Conversion Cycle
7. **Computing Ratios**: Profitability, leverage, growth, and efficiency ratios
8. **Valuation**: Multiples-based relative valuation
9. **Generating Memo**: LLM-powered (or rule-based) investment recommendation

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd fundamental_analyst_agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Analyze Apple (default)
python run_demo.py

# Analyze any ticker
python run_demo.py MSFT

# Specify output directory
python run_demo.py GOOGL --output ./my_reports

# Quiet mode (less verbose)
python run_demo.py TSLA --quiet
```

### Output Files

After running, the agent generates four output files:

| Format | File | Description |
|--------|------|-------------|
| JSON | `{TICKER}_analysis_*.json` | Machine-readable structured data |
| Markdown | `{TICKER}_investment_memo_*.md` | Investment memo document |
| HTML | `{TICKER}_report_*.html` | Interactive web report |
| PDF | `{TICKER}_report_*.pdf` | Printable professional document |

---

## Project Structure

```
fundamental_analyst_agent/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── run_demo.py               # Demo script (MAIN ENTRY POINT)
│
├── src/                      # Source modules
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration and constants
│   ├── data_collector.py     # Step 1: Data ingestion
│   ├── data_processor.py     # Step 2: Data standardization
│   ├── profitability_analyzer.py  # Step 3: Profitability bridge
│   ├── cash_flow_analyzer.py # Step 4: Cash flow bridge
│   ├── earnings_quality_analyzer.py  # Step 5: Earnings quality
│   ├── working_capital_analyzer.py   # Step 6: Working capital
│   ├── ratio_calculator.py   # Step 7: Financial ratios
│   ├── valuation.py          # Step 8: Valuation multiples
│   ├── memo_generator.py     # Step 9: Investment memo
│   └── agent.py              # Step 10: Main orchestration
│
└── outputs/                  # Generated reports
    └── {TICKER}/
        ├── {TICKER}_analysis_*.json
        ├── {TICKER}_investment_memo_*.md
        ├── {TICKER}_report_*.html
        └── {TICKER}_report_*.pdf
```

---

## Module Descriptions

### Step 0: config.py
Centralized configuration including:
- Field name mappings for Yahoo Finance data
- Validation thresholds
- Earnings quality assessment thresholds
- Valuation parameters

### Step 1: data_collector.py
Data ingestion from Yahoo Finance:
- Income Statement (5 years annual)
- Balance Sheet (5 years annual)
- Cash Flow Statement (5 years annual)
- Company metadata (price, beta, shares outstanding)

### Step 2: data_processor.py
Data standardization:
- Field name normalization
- Missing value handling
- Derived field calculation (EBITDA, Gross Profit)
- Accounting identity validation
- Audit trail maintenance

### Step 3: profitability_analyzer.py
EBIT Variance Bridge:
```
EBIT Change = Volume Effect + GM Rate Effect + OpEx Rate Effect
```
- Mathematically guaranteed reconciliation
- Margin trend analysis
- Primary driver identification

### Step 4: cash_flow_analyzer.py
Cash Flow Bridge:
```
Net Income
  + Non-cash items (D&A, Stock Comp)
  + Working capital changes
= Operating Cash Flow
  - CapEx
= Free Cash Flow
```
- Cash conversion rate
- FCF margin analysis
- Red flag detection

### Step 5: earnings_quality_analyzer.py
Earnings quality assessment:
- Accruals ratio analysis
- Growth divergence detection (AR vs Revenue)
- Red flag identification
- Quality scoring (0-100)

### Step 6: working_capital_analyzer.py
Working capital efficiency:
- DSO, DIO, DPO calculations
- Cash Conversion Cycle
- Trend analysis
- Alert generation

### Step 7: ratio_calculator.py
Comprehensive ratio analysis:
- Profitability: ROE, ROA, ROIC, margins
- Leverage: D/E, Interest Coverage, D/EBITDA
- Efficiency: Asset Turnover, Inventory Turnover
- DuPont decomposition

### Step 8: valuation.py
Multiples-based valuation:
- P/E, P/B, P/S, P/FCF
- EV/EBITDA, EV/Revenue
- Implied growth analysis
- Valuation assessment

### Step 9: memo_generator.py
Investment memo generation:
- LLM mode (Claude API) for intelligent narratives
- Rule-based fallback with templates
- Multi-factor recommendation scoring
- Data citation integration

### Step 10: agent.py
Main orchestration:
- Pipeline coordination
- Error handling
- Multi-format export (JSON, MD, HTML, PDF)
- Summary reporting

---

## Key Features

### Analytical Framework

1. **Profitability vs Cash Drivers**
   - The core thesis: "Are reported profits real?"
   - EBIT bridge shows what's driving profits
   - Cash flow bridge shows if profits convert to cash

2. **Earnings Quality Assessment**
   - Accruals analysis identifies aggressive accounting
   - Red flag detection for common manipulation patterns
   - Quality scoring provides objective assessment

3. **Working Capital Efficiency**
   - Cash Conversion Cycle optimization
   - Trend analysis for early warning signs

4. **Valuation Context**
   - Multiple methods for robustness
   - Implied growth analysis
   - Historical comparison

### Export Formats

| Format | Use Case |
|--------|----------|
| **JSON** | Integration with other systems, data analysis |
| **Markdown** | Human-readable memo, easy editing |
| **HTML** | Interactive web viewing, presentation |
| **PDF** | Professional reports, printing |

---

## Configuration

### Environment Variables

```bash
# Optional: For LLM-powered memo generation
export ANTHROPIC_API_KEY=your_api_key_here
```

Without an API key, the agent uses rule-based memo generation.

### Customization

Edit `src/config.py` to adjust:
- Validation thresholds
- Earnings quality ratings
- Valuation parameters
- Output settings

---

## Dependencies

Core dependencies (see `requirements.txt`):

```
yfinance>=0.2.0      # Yahoo Finance data
pandas>=2.0.0        # Data manipulation
numpy>=1.24.0        # Numerical operations
reportlab>=4.0.0     # PDF generation
anthropic>=0.5.0     # LLM integration (optional)
```

---

## Usage Examples

### Basic Usage

```python
from src import FundamentalAnalystAgent

# Initialize agent
agent = FundamentalAnalystAgent("AAPL")

# Run analysis
results = agent.run_analysis()

# Export results
agent.export_results("outputs/AAPL")

# Print summary
agent.print_summary()
```

### Access Individual Components

```python
from src import (
    DataCollector,
    DataProcessor,
    ProfitabilityAnalyzer,
    CashFlowAnalyzer
)

# Step 1: Collect data
collector = DataCollector()
raw_data = collector.collect("MSFT")

# Step 2: Process data
processor = DataProcessor()
processed_data = processor.process(raw_data)

# Step 3: Analyze profitability
prof_analyzer = ProfitabilityAnalyzer(processed_data)
profitability = prof_analyzer.analyze()

# Step 4: Analyze cash flow
cf_analyzer = CashFlowAnalyzer(processed_data)
cash_flow = cf_analyzer.analyze()
```

---

## Troubleshooting

### Common Issues

**1. No data returned for ticker**
- Verify the ticker symbol is correct
- Check internet connection
- Ensure ticker is available on Yahoo Finance

**2. Import errors**
- Ensure you're in the project root directory
- Run from virtual environment with dependencies installed

**3. PDF generation fails**
- Install reportlab: `pip install reportlab`
- Check write permissions for output directory

**4. LLM memo generation not working**
- Set ANTHROPIC_API_KEY environment variable
- Falls back to rule-based generation automatically

---

## Assessment Criteria Alignment

| Criterion | Implementation |
|-----------|----------------|
| **Solution Development (20%)** | Complete 10-step pipeline with 11 modules |
| **Code Quality (15%)** | Well-documented, type-hinted, modular design |
| **Scalability (10%)** | Any ticker supported, configurable thresholds |
| **Performance (15%)** | Error handling, validation, reconciliation checks |
| **Presentation (40%)** | Multi-format output, professional memo |

---

## License

This project is submitted as coursework for IFTE0001: AI Agents in Asset Management.

---

## Author

MSc AI Agents in Asset Management - Track A: Fundamental Analyst Agent