#!/usr/bin/env python3
"""
Phases 1-3 Execution Script - Fundamental Analyst AI Agent
===============================================================================

This script runs only Phases 1-3 of the 10-step analysis pipeline:
- Phase 1: Data Collection (data_collector.py)
- Phase 2: Data Processing (data_processor.py)  
- Phase 3: Profitability Analysis (profitability_analyzer.py)

MSc Coursework: IFTE0001 - AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Author: MSc AI Agents in Asset Management
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Configure Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "MB9136UVECFVPN76"
os.environ["ALPHA_VANTAGE_API_KEY"] = ALPHA_VANTAGE_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from data_collector import DataCollector, CollectedData
from data_processor import DataProcessor, ProcessedData
from profitability_analyzer import ProfitabilityAnalyzer, ProfitabilityAnalysisResult


# =============================================================================
# PHASE 1: DATA COLLECTION
# =============================================================================

def run_phase1_data_collection(ticker: str) -> Tuple[CollectedData, float]:
    """
    Phase 1: Collect financial data from Alpha Vantage/Yahoo Finance.
    
    Returns:
        Tuple of (CollectedData, execution_time_ms)
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 80)
    print(f"\nTarget Ticker: {ticker}")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nData Sources:")
    print("  - Primary: Alpha Vantage API (5 years of annual data)")
    print("  - Fallback: Yahoo Finance (for company metadata)")
    
    # Initialize collector
    collector = DataCollector()
    
    # Collect data
    print("\n[1.1] Fetching company information...")
    raw_data = collector.collect(ticker)
    
    # Display results
    company_info = raw_data.company_info
    statements = raw_data.statements
    validation = raw_data.validation
    
    print(f"\n{'─' * 80}")
    print("COMPANY INFORMATION")
    print(f"{'─' * 80}")
    print(f"  Name:               {company_info.name}")
    print(f"  Ticker:             {company_info.ticker}")
    print(f"  Sector:             {company_info.sector or 'N/A'}")
    print(f"  Industry:           {company_info.industry or 'N/A'}")
    print(f"  Currency:           {company_info.currency}")
    if company_info.current_price:
        print(f"  Current Price:      ${company_info.current_price:,.2f}")
    if company_info.market_cap:
        print(f"  Market Cap:         ${company_info.market_cap/1e9:,.1f}B")
    if company_info.shares_outstanding:
        print(f"  Shares Outstanding: {company_info.shares_outstanding/1e9:.2f}B")
    print(f"  Data Source:        {raw_data.data_source}")
    
    # Display financial statements
    print(f"\n{'─' * 80}")
    print("FINANCIAL STATEMENTS SUMMARY")
    print(f"{'─' * 80}")
    
    # Income Statement Summary
    print(f"\n  INCOME STATEMENT:")
    if statements.income_statement is not None and not statements.income_statement.empty:
        is_df = statements.income_statement
        print(f"    Periods: {len(is_df.columns)} years")
        fiscal_years = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in is_df.columns]
        print(f"    Fiscal Years: {', '.join(fiscal_years)}")
        print(f"    Line Items: {len(is_df)}")
        
        # Key metrics
        print(f"\n    Key Metrics (Latest Year - {fiscal_years[0]}):")
        key_items = ['Total Revenue', 'Gross Profit', 'Operating Income', 'EBIT', 'Net Income']
        for item in key_items:
            if item in is_df.index:
                vals = is_df.loc[item]
                latest = vals.iloc[0] if len(vals) > 0 else None
                if latest is not None and not (hasattr(latest, 'isna') and latest.isna()):
                    try:
                        print(f"      {item:25s}: ${float(latest)/1e6:>12,.0f}M")
                    except:
                        print(f"      {item:25s}: N/A")
    else:
        print("    NOT AVAILABLE")
    
    # Balance Sheet Summary
    print(f"\n  BALANCE SHEET:")
    if statements.balance_sheet is not None and not statements.balance_sheet.empty:
        bs_df = statements.balance_sheet
        print(f"    Periods: {len(bs_df.columns)} years")
        fiscal_years_bs = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in bs_df.columns]
        print(f"    Fiscal Years: {', '.join(fiscal_years_bs)}")
        print(f"    Line Items: {len(bs_df)}")
        
        # Key metrics
        print(f"\n    Key Metrics (Latest Year - {fiscal_years_bs[0]}):")
        key_items = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity',
                     'Cash And Cash Equivalents', 'Current Assets', 'Current Liabilities']
        for item in key_items:
            if item in bs_df.index:
                vals = bs_df.loc[item]
                latest = vals.iloc[0] if len(vals) > 0 else None
                if latest is not None and not (hasattr(latest, 'isna') and latest.isna()):
                    try:
                        print(f"      {item:45s}: ${float(latest)/1e6:>12,.0f}M")
                    except:
                        pass
    else:
        print("    NOT AVAILABLE")
    
    # Cash Flow Summary
    print(f"\n  CASH FLOW STATEMENT:")
    if statements.cash_flow is not None and not statements.cash_flow.empty:
        cf_df = statements.cash_flow
        print(f"    Periods: {len(cf_df.columns)} years")
        fiscal_years_cf = [str(c.year) if hasattr(c, 'year') else str(c)[:4] for c in cf_df.columns]
        print(f"    Fiscal Years: {', '.join(fiscal_years_cf)}")
        print(f"    Line Items: {len(cf_df)}")
        
        # Key metrics
        print(f"\n    Key Metrics (Latest Year - {fiscal_years_cf[0]}):")
        key_items = ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure', 
                     'Depreciation And Amortization', 'Net Income From Continuing Operations']
        for item in key_items:
            if item in cf_df.index:
                vals = cf_df.loc[item]
                latest = vals.iloc[0] if len(vals) > 0 else None
                if latest is not None and not (hasattr(latest, 'isna') and latest.isna()):
                    try:
                        print(f"      {item:45s}: ${float(latest)/1e6:>12,.0f}M")
                    except:
                        pass
    else:
        print("    NOT AVAILABLE")
    
    # Validation Results
    print(f"\n{'─' * 80}")
    print("VALIDATION RESULTS")
    print(f"{'─' * 80}")
    print(f"  Status:           {validation.status.value.upper()}")
    print(f"  Complete Years:   {validation.years_available}")
    
    if validation.errors:
        print(f"  Errors ({len(validation.errors)}):")
        for e in validation.errors:
            print(f"    - {e}")
    
    if validation.warnings:
        print(f"  Warnings ({len(validation.warnings)}):")
        for w in validation.warnings:
            print(f"    - {w}")
    
    if validation.missing_statements:
        print(f"  Missing Statements: {', '.join(validation.missing_statements)}")
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"\n{'─' * 80}")
    print(f"Phase 1 Complete: {execution_time:.1f}ms")
    print(f"{'─' * 80}")
    
    return raw_data, execution_time


# =============================================================================
# PHASE 2: DATA PROCESSING
# =============================================================================

def run_phase2_data_processing(raw_data: CollectedData) -> Tuple[ProcessedData, float]:
    """
    Phase 2: Process and standardize raw financial data.
    
    Returns:
        Tuple of (ProcessedData, execution_time_ms)
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("PHASE 2: DATA PROCESSING")
    print("=" * 80)
    print(f"\nTarget Company: {raw_data.company_info.name} ({raw_data.company_info.ticker})")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nProcessing Steps:")
    print("  1. Field Standardization: Map field names to internal format")
    print("  2. Missing Value Handling: Identify gaps in data")
    print("  3. Derived Field Calculation: Compute EBITDA, Gross Profit, etc.")
    print("  4. Unit Normalization: Convert to millions USD")
    print("  5. Accounting Identity Validation: Verify A = L + E")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Process data
    print("\n[2.1] Processing financial data...")
    processed = processor.process(raw_data)
    
    # Display results
    company_info = processed.company_info
    statements = processed.statements
    quality_metrics = processed.quality_metrics
    
    print(f"\n{'─' * 80}")
    print("PROCESSED DATA OVERVIEW")
    print(f"{'─' * 80}")
    print(f"  Ticker:          {company_info.ticker}")
    print(f"  Company:         {company_info.name}")
    print(f"  Currency:        {processed.currency}")
    print(f"  Periods:         {processed.periods}")
    print(f"  Number of Periods: {processed.num_periods}")
    
    # Data Quality Metrics
    print(f"\n{'─' * 80}")
    print("DATA QUALITY METRICS")
    print(f"{'─' * 80}")
    print(f"  Fields Expected:   {quality_metrics.total_fields_expected}")
    print(f"  Fields Found:      {quality_metrics.fields_found}")
    print(f"  Fields Derived:    {quality_metrics.fields_derived}")
    print(f"  Fields Missing:    {quality_metrics.fields_missing}")
    print(f"  Coverage Ratio:    {quality_metrics.coverage_ratio:.1%}")
    print(f"  Periods Available: {quality_metrics.periods_available}")
    print(f"  Is High Quality:   {quality_metrics.is_high_quality}")
    print(f"  Is Usable:         {quality_metrics.is_usable}")
    
    # Processed Financial Statements
    print(f"\n{'─' * 80}")
    print("PROCESSED INCOME STATEMENT (Millions USD)")
    print(f"{'─' * 80}")
    
    income = statements.income_statement
    if not income.empty and income.shape[0] > 0:
        # Format for display
        display_df = income.copy()
        display_df.columns = statements.periods
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.1f}M" if x is not None and (not hasattr(x, 'isna') or not x.isna()) and x != 0 else "N/A"
            )
        print(display_df.to_string())
    
    print(f"\n{'─' * 80}")
    print("PROCESSED BALANCE SHEET (Millions USD)")
    print(f"{'─' * 80}")
    
    balance = statements.balance_sheet
    if not balance.empty and balance.shape[0] > 0:
        display_df = balance.copy()
        display_df.columns = statements.periods
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.1f}M" if x is not None and (not hasattr(x, 'isna') or not x.isna()) and x != 0 else "N/A"
            )
        print(display_df.to_string())
    
    print(f"\n{'─' * 80}")
    print("PROCESSED CASH FLOW STATEMENT (Millions USD)")
    print(f"{'─' * 80}")
    
    cashflow = statements.cash_flow
    if not cashflow.empty and cashflow.shape[0] > 0:
        display_df = cashflow.copy()
        display_df.columns = statements.periods
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.1f}M" if x is not None and (not hasattr(x, 'isna') or not x.isna()) and x != 0 else "N/A"
            )
        print(display_df.to_string())
    
    # Transformation Log
    print(f"\n{'─' * 80}")
    print("TRANSFORMATION LOG (Audit Trail)")
    print(f"{'─' * 80}")
    
    transformations = processed.transformations
    if transformations:
        print(f"  Total Transformations: {len(transformations)}")
        for i, t in enumerate(transformations[:10], 1):  # Show first 10
            print(f"  {i}. [{t.transformation_type.upper()}] {t.field}: {t.description}")
        if len(transformations) > 10:
            print(f"  ... and {len(transformations) - 10} more")
    else:
        print("  No transformations required")
    
    # Warnings
    if processed.warnings:
        print(f"\n{'─' * 80}")
        print("PROCESSING WARNINGS")
        print(f"{'─' * 80}")
        for w in processed.warnings:
            print(f"  - {w}")
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"\n{'─' * 80}")
    print(f"Phase 2 Complete: {execution_time:.1f}ms")
    print(f"{'─' * 80}")
    
    return processed, execution_time


# =============================================================================
# PHASE 3: PROFITABILITY ANALYSIS
# =============================================================================

def run_phase3_profitability_analysis(processed_data: ProcessedData) -> Tuple[ProfitabilityAnalysisResult, float]:
    """
    Phase 3: Perform EBIT variance bridge and profitability analysis.
    
    Returns:
        Tuple of (ProfitabilityAnalysisResult, execution_time_ms)
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("PHASE 3: PROFITABILITY ANALYSIS")
    print("=" * 80)
    print(f"\nTarget Company: {processed_data.company_info.name} ({processed_data.company_info.ticker})")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAnalysis Components:")
    print("  1. EBIT Variance Bridge: Decompose operating income changes")
    print("  2. Margin Analysis: Gross, Operating, EBITDA, Net margins")
    print("  3. Driver Identification: Volume vs Margin effects")
    print("  4. Insight Generation: Key findings from analysis")
    
    # Initialize analyzer
    analyzer = ProfitabilityAnalyzer(processed_data)
    
    # Run analysis
    print("\n[3.1] Calculating EBIT variance bridge...")
    result = analyzer.analyze()
    
    # Display results
    bridge = result.bridge
    metrics = result.metrics
    
    print(f"\n{'─' * 80}")
    print("EBIT VARIANCE BRIDGE")
    print(f"{'─' * 80}")
    print(f"  Analysis Period: {result.analysis_period}")
    print(f"  Validation Status: {bridge.validation_status.value.upper()}")
    print(f"  Is Reconciled: {bridge.is_reconciled}")
    
    print(f"\n  {'Component':<35} {'Amount ($M)':>15} {'% of Change':>12}")
    print(f"  {'─'*35} {'─'*15} {'─'*12}")
    
    for component in bridge.components:
        if component.amount is not None:
            amount_str = f"${component.amount:,.1f}"
        else:
            amount_str = "N/A"
        
        if component.percentage_of_change is not None:
            pct_str = f"{component.percentage_of_change:+.1f}%"
        else:
            pct_str = ""
        
        if component.is_subtotal:
            print(f"  {component.name:<35} {amount_str:>15} {pct_str:>12}")
        else:
            print(f"    {component.name:<33} {amount_str:>15} {pct_str:>12}")
    
    print(f"\n  {'─'*35} {'─'*15} {'─'*12}")
    print(f"  {'EBIT Change:':<35} ${bridge.ebit_change:+,.1f}M")
    print(f"  {'Reconciliation Difference:':<35} ${bridge.reconciliation_difference:,.2f}M")
    
    # Mathematical Foundation
    print(f"\n{'─' * 80}")
    print("MATHEMATICAL FOUNDATION")
    print(f"{'─' * 80}")
    print("""
  The EBIT Variance Bridge decomposes operating income changes into fundamental drivers:
  
  ΔEBIT = Volume Effect + GM Rate Effect + OpEx Rate Effect
  
  Where:
    Volume Effect    = (R₁ - R₀) × OM₀      (revenue change at prior margin)
    GM Rate Effect   = (GM₁ - GM₀) × R₁     (margin change at new revenue)
    OpEx Rate Effect = -(OE₁ - OE₀) × R₁    (efficiency change)
  
  This decomposition is mathematically guaranteed to reconcile exactly.
    """)
    
    # Key Metrics
    print(f"\n{'─' * 80}")
    print("KEY PROFITABILITY METRICS")
    print(f"{'─' * 80}")
    print(f"  Revenue (Current):     ${metrics.revenue_current:,.1f}M")
    print(f"  Revenue (Prior):       ${metrics.revenue_prior:,.1f}M")
    print(f"  Revenue Change:        ${metrics.revenue_change:+,.1f}M")
    print(f"  Revenue Growth Rate:   {metrics.revenue_growth_rate:+.2f}%")
    print(f"  Gross Profit (Current):${metrics.gross_profit_current:,.1f}M")
    print(f"  Operating Income (Cur):${metrics.operating_income_current:,.1f}M")
    print(f"  Operating Income (Pri):${metrics.operating_income_prior:,.1f}M")
    print(f"  EBIT Change:           ${bridge.ebit_change:+,.1f}M")
    print(f"  EBIT Growth Rate:      {(bridge.ebit_change/bridge.prior_ebit*100):+.2f}%" if bridge.prior_ebit != 0 else "  EBIT Growth Rate:      N/A")
    
    # Margin Analysis
    print(f"\n{'─' * 80}")
    print("MARGIN ANALYSIS")
    print(f"{'─' * 80}")
    print(f"  {'Margin Type':<20} {'Current':>12} {'Prior':>12} {'Change (pp)':>12} {'Trend':<12}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    
    for margin_type, analysis in metrics.margins.items():
        metric_name = margin_type.value.replace('_', ' ').title()
        current_str = f"{analysis.current_value*100:.2f}%"
        prior_str = f"{analysis.prior_value*100:.2f}%"
        change_str = f"{analysis.change:+.2f}pp"
        print(f"  {metric_name:<20} {current_str:>12} {prior_str:>12} {change_str:>12} {analysis.trend_direction:<12}")
    
    # Key Insights
    print(f"\n{'─' * 80}")
    print("KEY INSIGHTS")
    print(f"{'─' * 80}")
    for i, insight in enumerate(result.insights, 1):
        print(f"  {i}. {insight}")
    
    # Warnings
    if result.warnings:
        print(f"\n{'─' * 80}")
        print("ANALYSIS WARNINGS")
        print(f"{'─' * 80}")
        for w in result.warnings:
            print(f"  - {w}")
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"\n{'─' * 80}")
    print(f"Phase 3 Complete: {execution_time:.1f}ms")
    print(f"{'─' * 80}")
    
    return result, execution_time


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for Phases 1-3 execution."""
    
    print("\n" + "=" * 80)
    print("FUNDAMENTAL ANALYST AI AGENT - PHASES 1-3 EXECUTION")
    print("=" * 80)
    print("""
  MSc Coursework: IFTE0001 - AI Agents in Asset Management
  Track A: Fundamental Analyst Agent
  
  Executing:
    Phase 1: Data Collection      - Fetch 5 years of financial statements
    Phase 2: Data Processing      - Standardize and validate data
    Phase 3: Profitability        - EBIT variance bridge analysis
    """)
    
    # Get ticker from command line or use default
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    
    total_start = datetime.now()
    
    # Phase 1: Data Collection
    raw_data, time_p1 = run_phase1_data_collection(ticker)
    
    # Phase 2: Data Processing
    processed_data, time_p2 = run_phase2_data_processing(raw_data)
    
    # Phase 3: Profitability Analysis
    result, time_p3 = run_phase3_profitability_analysis(processed_data)
    
    # Summary
    total_time = (datetime.now() - total_start).total_seconds() * 1000
    
    print("\n" + "=" * 80)
    print("PHASES 1-3 EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\n  Company:      {processed_data.company_info.name} ({processed_data.company_info.ticker})")
    print(f"  Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"  Data Source:   {raw_data.data_source}")
    print(f"\n  {'Phase':<30} {'Status':<15} {'Time':<10}")
    print(f"  {'─'*30} {'─'*15} {'─'*10}")
    print(f"  {'Phase 1: Data Collection':<30} {'COMPLETE':<15} {time_p1:.1f}ms")
    print(f"  {'Phase 2: Data Processing':<30} {'COMPLETE':<15} {time_p2:.1f}ms")
    print(f"  {'Phase 3: Profitability':<30} {'COMPLETE':<15} {time_p3:.1f}ms")
    print(f"  {'─'*30} {'─'*15} {'─'*10}")
    print(f"  {'TOTAL':<30} {'─':<15} {total_time:.1f}ms")
    
    print(f"\n  Key Results:")
    print(f"    - EBIT Change: ${result.bridge.ebit_change:+,.1f}M")
    print(f"    - Primary Driver: {result.insights[1] if len(result.insights) > 1 else 'N/A'}")
    print(f"    - Operating Margin: {result.metrics.margins.get('operating_margin', 'N/A')}")
    print(f"    - Bridge Reconciled: {result.bridge.is_reconciled}")
    
    print("\n" + "=" * 80)
    print("PHASES 1-3 COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

