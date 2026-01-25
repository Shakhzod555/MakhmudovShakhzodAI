#!/usr/bin/env python3
"""
Phases 4-7 Execution Script - Fundamental Analyst AI Agent
===============================================================================

This script runs Phases 4-7 of the 10-step analysis pipeline:
- Phase 4: Cash Flow Analysis (cash_flow_analyzer.py)
- Phase 5: Earnings Quality Analysis (earnings_quality_analyzer.py)
- Phase 6: Working Capital Analysis (working_capital_analyzer.py)
- Phase 7: Financial Ratios (ratio_calculator.py)

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from data_collector import DataCollector, CollectedData
from data_processor import DataProcessor, ProcessedData
from cash_flow_analyzer import CashFlowAnalyzer, CashFlowAnalysisResult
from earnings_quality_analyzer import EarningsQualityAnalyzer, EarningsQualityResult
from working_capital_analyzer import WorkingCapitalAnalyzer, WorkingCapitalAnalysisResult
from ratio_calculator import RatioCalculator, RatioCalculatorResult


# =============================================================================
# PHASE 4: CASH FLOW ANALYSIS
# =============================================================================

def run_phase4_cash_flow_analysis(processed_data: ProcessedData) -> Tuple[CashFlowAnalysisResult, float]:
    """
    Phase 4: Analyze cash flow quality and conversion from Net Income to Free Cash Flow.
    
    Cash Flow Bridge:
        Net Income
          + Depreciation and Amortization
          + Stock-Based Compensation
          + Deferred Tax
          + Change in Working Capital
          + Other Non-Cash Items
        = Operating Cash Flow (OCF)
          - Capital Expenditure
        = Free Cash Flow (FCF)
    
    Returns:
        Tuple of (CashFlowAnalysisResult, execution_time_ms)
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("PHASE 4: CASH FLOW ANALYSIS")
    print("=" * 80)
    print(f"\nTarget Company: {processed_data.company_info.name} ({processed_data.company_info.ticker})")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAnalysis Components:")
    print("  1. Net Income to OCF Bridge: Trace profit to cash conversion")
    print("  2. OCF to Free Cash Flow: Assess cash available for distribution")
    print("  3. Cash Conversion Quality: OCF/Net Income ratio assessment")
    print("  4. CapEx Analysis: Investment patterns and profile")
    print("  5. Working Capital Cash Impact: WC changes on cash flow")
    
    # Initialize analyzer
    analyzer = CashFlowAnalyzer(processed_data)
    
    # Run analysis
    print("\n[4.1] Building cash flow bridge...")
    result = analyzer.analyze()
    
    # Display results
    bridge = result.bridge
    
    print(f"\n{'─' * 80}")
    print("CASH FLOW BRIDGE")
    print(f"{'─' * 80}")
    print(f"  Analysis Period: {result.analysis_period}")
    print(f"  Validation Status: {bridge.validation_status.value.upper()}")
    print(f"  Is Reconciled: {bridge.is_reconciled}")
    
    print(f"\n  {'Component':<35} {'Amount ($M)':>15} {'% of NI':>10}")
    print(f"  {'─'*35} {'─'*15} {'─'*10}")
    
    for item in bridge.items:
        amount_str = f"${item.amount:,.1f}"
        pct_str = f"{item.as_percent_of_ni:.1f}%" if item.as_percent_of_ni else ""
        
        if item.is_subtotal:
            print(f"  {item.name:<35} {amount_str:>15} {pct_str:>10}")
        else:
            print(f"    {item.name:<33} {amount_str:>15} {pct_str:>10}")
    
    # Cash Conversion Metrics
    conv = result.conversion_metrics
    print(f"\n{'─' * 80}")
    print("CASH CONVERSION METRICS")
    print(f"{'─' * 80}")
    print(f"  Cash Conversion Rate:  {conv.cash_conversion_rate*100:.1f}%")
    print(f"  FCF Conversion Rate:   {conv.fcf_conversion_rate*100:.1f}%")
    print(f"  FCF Margin:            {conv.fcf_margin*100:.1f}%")
    print(f"  Quality Rating:        {conv.quality_rating.value}")
    print(f"  {conv.quality_description}")
    
    # CapEx Analysis
    capex = result.capex_analysis
    print(f"\n{'─' * 80}")
    print("CAPITAL EXPENDITURE ANALYSIS")
    print(f"{'─' * 80}")
    print(f"  CapEx Amount:          ${capex.capex_amount:,.1f}M")
    print(f"  CapEx / Revenue:       {capex.capex_to_revenue*100:.1f}%")
    print(f"  CapEx / D&A:           {capex.capex_to_depreciation:.2f}x")
    print(f"  Profile:               {capex.capex_profile.value}")
    print(f"  {capex.profile_description}")
    
    # Working Capital Impact
    wc = result.working_capital_impact
    print(f"\n{'─' * 80}")
    print("WORKING CAPITAL CASH IMPACT")
    print(f"{'─' * 80}")
    print(f"  Total WC Change:       ${wc.total_wc_change:,.1f}M")
    print(f"  AR Change:             ${wc.receivables_change:,.1f}M")
    print(f"  Inventory Change:      ${wc.inventory_change:,.1f}M")
    print(f"  AP Change:             ${wc.payables_change:,.1f}M")
    print(f"  {wc.cash_impact_description}")
    
    # Key Insights
    if result.insights:
        print(f"\n{'─' * 80}")
        print("KEY INSIGHTS")
        print(f"{'─' * 80}")
        for i, insight in enumerate(result.insights, 1):
            print(f"  {i}. {insight}")
    
    # Red Flags
    if result.red_flags:
        print(f"\n{'─' * 80}")
        print("RED FLAGS")
        print(f"{'─' * 80}")
        for flag in result.red_flags:
            print(f"  [{flag.severity.value.upper()}] {flag.title}")
            print(f"    {flag.description}")
    
    # Warnings
    if result.warnings:
        print(f"\n{'─' * 80}")
        print("WARNINGS")
        print(f"{'─' * 80}")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"\n{'─' * 80}")
    print(f"Phase 4 Complete: {execution_time:.1f}ms")
    print(f"{'─' * 80}")
    
    return result, execution_time


# =============================================================================
# PHASE 5: EARNINGS QUALITY ANALYSIS
# =============================================================================

def run_phase5_earnings_quality_analysis(processed_data: ProcessedData) -> Tuple[EarningsQualityResult, float]:
    """
    Phase 5: Assess earnings quality through accruals analysis and red flag detection.
    
    Analytical Framework:
    1. Accruals Analysis: (Net Income - OCF) / Average Total Assets
    2. Cash Conversion Analysis: OCF / Net Income
    3. Growth Divergence Detection: AR vs Revenue, Inventory vs COGS
    4. Red Flag Identification
    
    Returns:
        Tuple of (EarningsQualityResult, execution_time_ms)
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("PHASE 5: EARNINGS QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nTarget Company: {processed_data.company_info.name} ({processed_data.company_info.ticker})")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAnalysis Components:")
    print("  1. Accruals Analysis: Measure earnings backed by cash")
    print("  2. Cash Conversion: OCF/Net Income ratio assessment")
    print("  3. Growth Divergence: AR vs Revenue, Inventory vs COGS")
    print("  4. Red Flag Detection: Earnings manipulation patterns")
    print("  5. Quality Scoring: Composite quality score (0-100)")
    
    # Initialize analyzer
    analyzer = EarningsQualityAnalyzer(processed_data)
    
    # Run analysis
    print("\n[5.1] Assessing earnings quality...")
    result = analyzer.analyze()
    
    # Display results
    accruals = result.accruals_analysis
    
    print(f"\n{'─' * 80}")
    print("OVERALL EARNINGS QUALITY")
    print(f"{'─' * 80}")
    print(f"  Analysis Period:  {result.analysis_period}")
    print(f"  Overall Rating:   {result.overall_rating.value}")
    print(f"  Quality Score:    {result.overall_score:.0f}/100")
    
    # Accruals Analysis
    print(f"\n{'─' * 80}")
    print("ACCRUALS ANALYSIS")
    print(f"{'─' * 80}")
    print(f"  Total Accruals:         ${accruals.total_accruals:,.1f}M")
    print(f"  Average Total Assets:   ${accruals.average_total_assets:,.1f}M")
    print(f"  Accruals Ratio:         {accruals.accruals_ratio*100:.2f}%")
    print(f"  Rating:                 {accruals.accruals_rating.value}")
    print(f"  {accruals.interpretation}")
    
    if accruals.historical_trend and len(accruals.historical_trend) > 1:
        print(f"\n  Historical Accruals Trend:")
        for i, val in enumerate(accruals.historical_trend[:5]):
            period_label = f"Period -{i}" if i > 0 else "Current"
            print(f"    {period_label}: {val*100:.2f}%")
    
    # Cash Conversion
    print(f"\n{'─' * 80}")
    print("CASH CONVERSION")
    print(f"{'─' * 80}")
    print(f"  Cash Conversion Rate:  {result.cash_conversion_rate*100:.1f}%")
    
    # Growth Divergences
    if result.growth_divergences:
        print(f"\n{'─' * 80}")
        print("GROWTH DIVERGENCE ANALYSIS")
        print(f"{'─' * 80}")
        print(f"  {'Metric':<30} {'Divergence':>12} {'Status':<10}")
        print(f"  {'─'*30} {'─'*12} {'─'*10}")
        
        for div in result.growth_divergences:
            status = "FLAGGED" if div.is_flagged else "OK"
            print(f"  {div.metric_name:<30} {div.divergence*100:>+10.1f}pp {status:<10}")
    
    # Quality Scores
    print(f"\n{'─' * 80}")
    print("QUALITY SCORE BREAKDOWN")
    print(f"{'─' * 80}")
    print(f"  {'Component':<25} {'Score':>8} {'Weight':>8} {'Weighted':>10}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10}")
    
    for score in result.quality_scores:
        print(f"  {score.component:<25} {score.score:>8.0f} "
              f"{score.weight*100:>7.0f}% {score.weighted_score:>10.1f}")
    
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10}")
    print(f"  {'TOTAL':<25} {result.overall_score:>8.0f} {'100':>8}% {result.overall_score:>10.1f}")
    
    # Red Flags
    if result.red_flags:
        print(f"\n{'─' * 80}")
        print("RED FLAGS")
        print(f"{'─' * 80}")
        for flag in result.red_flags:
            print(f"  [{flag.severity.value.upper()}] {flag.title}")
            print(f"    {flag.description}")
    
    # Positive Indicators
    if result.positive_indicators:
        print(f"\n{'─' * 80}")
        print("POSITIVE INDICATORS")
        print(f"{'─' * 80}")
        for indicator in result.positive_indicators:
            print(f"  + {indicator}")
    
    # Key Insights
    if result.insights:
        print(f"\n{'─' * 80}")
        print("KEY INSIGHTS")
        print(f"{'─' * 80}")
        for i, insight in enumerate(result.insights, 1):
            print(f"  {i}. {insight}")
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"\n{'─' * 80}")
    print(f"Phase 5 Complete: {execution_time:.1f}ms")
    print(f"{'─' * 80}")
    
    return result, execution_time


# =============================================================================
# PHASE 6: WORKING CAPITAL ANALYSIS
# =============================================================================

def run_phase6_working_capital_analysis(processed_data: ProcessedData) -> Tuple[WorkingCapitalAnalysisResult, float]:
    """
    Phase 6: Analyze working capital efficiency through DSO, DIO, DPO, and CCC.
    
    Working Capital Efficiency Metrics:
    - DSO (Days Sales Outstanding) = (AR / Revenue) × 365
    - DIO (Days Inventory Outstanding) = (Inventory / COGS) × 365
    - DPO (Days Payable Outstanding) = (AP / COGS) × 365
    - CCC (Cash Conversion Cycle) = DSO + DIO - DPO
    
    Returns:
        Tuple of (WorkingCapitalAnalysisResult, execution_time_ms)
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("PHASE 6: WORKING CAPITAL ANALYSIS")
    print("=" * 80)
    print(f"\nTarget Company: {processed_data.company_info.name} ({processed_data.company_info.ticker})")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAnalysis Components:")
    print("  1. DSO (Days Sales Outstanding): Collection efficiency")
    print("  2. DIO (Days Inventory Outstanding): Inventory turnover")
    print("  3. DPO (Days Payable Outstanding): Payment efficiency")
    print("  4. CCC (Cash Conversion Cycle): Overall working capital efficiency")
    print("  5. Working Capital Position: Balance sheet ratios")
    
    # Initialize analyzer
    analyzer = WorkingCapitalAnalyzer(processed_data)
    
    # Run analysis
    print("\n[6.1] Calculating efficiency metrics...")
    result = analyzer.analyze()
    
    # Display results
    print(f"\n{'─' * 80}")
    print("WORKING CAPITAL EFFICIENCY METRICS")
    print(f"{'─' * 80}")
    print(f"  {'Metric':<30} {'Current':>10} {'Prior':>10} {'Change':>10} {'Rating':<12}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
    
    for metric in [result.dso, result.dio, result.dpo]:
        current = f"{metric.current_value:.1f}"
        prior = f"{metric.prior_value:.1f}" if metric.prior_value else "N/A"
        change = f"{metric.change:+.1f}" if metric.change else "N/A"
        print(f"  {metric.name:<30} {current:>10} {prior:>10} {change:>10} {metric.rating.value:<12}")
    
    # Cash Conversion Cycle
    ccc = result.cash_conversion_cycle
    print(f"\n{'─' * 80}")
    print("CASH CONVERSION CYCLE (CCC)")
    print(f"{'─' * 80}")
    print(f"  CCC = DSO + DIO - DPO")
    print(f"  CCC = {ccc.dso_contribution:.1f} + {ccc.dio_contribution:.1f} - {ccc.dpo_contribution:.1f}")
    print(f"  CCC = {ccc.current_value:.1f} days")
    
    if ccc.prior_value is not None:
        print(f"  Prior Period: {ccc.prior_value:.1f} days")
        print(f"  Change: {ccc.change:+.1f} days")
    
    print(f"  Trend: {ccc.trend_direction.value}")
    print(f"  {ccc.interpretation}")
    
    # Working Capital Position
    pos = result.position
    print(f"\n{'─' * 80}")
    print("WORKING CAPITAL POSITION")
    print(f"{'─' * 80}")
    print(f"  Net Working Capital:       ${pos.net_working_capital:,.1f}M")
    print(f"  Operating Working Capital: ${pos.operating_working_capital:,.1f}M")
    print(f"  WC / Revenue:              {pos.wc_to_revenue*100:.1f}%")
    print(f"  Current Ratio:             {pos.current_ratio:.2f}x")
    print(f"  Quick Ratio:               {pos.quick_ratio:.2f}x")
    print(f"  YoY Change:                ${pos.yoy_change:+,.1f}M")
    print(f"  {pos.cash_impact}")
    
    # Component Analysis
    if result.components:
        print(f"\n{'─' * 80}")
        print("COMPONENT ANALYSIS")
        print(f"{'─' * 80}")
        
        for comp in result.components:
            print(f"  {comp.name}:")
            print(f"    Balance:       ${comp.current_balance:,.1f}M")
            print(f"    Days Metric:   {comp.days_metric:.1f} days")
            print(f"    {comp.interpretation}")
            print()
    
    # Alerts
    if result.alerts:
        print(f"{'─' * 80}")
        print("ALERTS")
        print(f"{'─' * 80}")
        for alert in result.alerts:
            print(f"  [{alert.severity.value.upper()}] {alert.alert_type.value}")
            print(f"    {alert.description}")
            print(f"    Recommendation: {alert.recommendation}")
            print()
    
    # Key Insights
    if result.insights:
        print(f"{'─' * 80}")
        print("KEY INSIGHTS")
        print(f"{'─' * 80}")
        for i, insight in enumerate(result.insights, 1):
            print(f"  {i}. {insight}")
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"\n{'─' * 80}")
    print(f"Phase 6 Complete: {execution_time:.1f}ms")
    print(f"{'─' * 80}")
    
    return result, execution_time


# =============================================================================
# PHASE 7: FINANCIAL RATIOS
# =============================================================================

def run_phase7_financial_ratios(processed_data: ProcessedData) -> Tuple[RatioCalculatorResult, float]:
    """
    Phase 7: Calculate comprehensive financial ratios across all categories.
    
    Ratio Categories:
    1. Profitability: ROE, ROA, ROIC, Margins
    2. Liquidity: Current Ratio, Quick Ratio
    3. Solvency: D/E, Interest Coverage, D/EBITDA
    4. Efficiency: Asset Turnover, Inventory Turnover
    5. Valuation: P/E, P/B, EV/EBITDA
    
    Returns:
        Tuple of (RatioCalculatorResult, execution_time_ms)
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("PHASE 7: FINANCIAL RATIOS")
    print("=" * 80)
    print(f"\nTarget Company: {processed_data.company_info.name} ({processed_data.company_info.ticker})")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nRatio Categories:")
    print("  1. Profitability: ROE, ROA, ROIC, Margins")
    print("  2. Liquidity: Current Ratio, Quick Ratio")
    print("  3. Solvency: D/E, Interest Coverage, Debt/EBITDA")
    print("  4. Efficiency: Asset Turnover, Inventory Turnover")
    print("  5. Valuation: P/E, P/B, EV/EBITDA")
    
    # Initialize calculator
    calculator = RatioCalculator(processed_data)
    
    # Run analysis
    print("\n[7.1] Calculating financial ratios...")
    result = calculator.calculate()
    
    # Profitability Ratios
    print(f"\n{'─' * 80}")
    print("PROFITABILITY RATIOS")
    print(f"{'─' * 80}")
    print(f"  {'Ratio':<25} {'Current':>12} {'Prior':>12} {'Interpretation':<15}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'-'*15}")
    
    for ratio in result.profitability_ratios:
        curr = f"{ratio.current_value*100:.1f}%" if ratio.current_value else "N/A"
        prior = f"{ratio.prior_value*100:.1f}%" if ratio.prior_value else "N/A"
        print(f"  {ratio.name:<25} {curr:>12} {prior:>12} {ratio.interpretation.value:<15}")
    
    # Liquidity Ratios
    print(f"\n{'─' * 80}")
    print("LIQUIDITY RATIOS")
    print(f"{'─' * 80}")
    for ratio in result.liquidity_ratios:
        curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
        prior = f"{ratio.prior_value:.2f}x" if ratio.prior_value else "N/A"
        print(f"  {ratio.name:<25} {curr:>12} {prior:>12} {ratio.interpretation.value:<15}")
    
    # Solvency Ratios
    print(f"\n{'─' * 80}")
    print("SOLVENCY RATIOS")
    print(f"{'─' * 80}")
    for ratio in result.solvency_ratios:
        if ratio.is_percentage and ratio.current_value is not None:
            curr = f"{ratio.current_value*100:.1f}%"
            prior = f"{ratio.prior_value*100:.1f}%" if ratio.prior_value else "N/A"
        elif ratio.current_value is not None:
            curr = f"{ratio.current_value:.2f}x"
            prior = f"{ratio.prior_value:.2f}x" if ratio.prior_value else "N/A"
        else:
            curr = "N/A"
            prior = "N/A"
        print(f"  {ratio.name:<25} {curr:>12} {prior:>12} {ratio.interpretation.value:<15}")
    
    # Efficiency Ratios
    print(f"\n{'─' * 80}")
    print("EFFICIENCY RATIOS")
    print(f"{'─' * 80}")
    for ratio in result.efficiency_ratios:
        curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
        prior = f"{ratio.prior_value:.2f}x" if ratio.prior_value else "N/A"
        print(f"  {ratio.name:<25} {curr:>12} {prior:>12} {ratio.interpretation.value:<15}")
    
    # Valuation Ratios
    if result.valuation_ratios:
        print(f"\n{'─' * 80}")
        print("VALUATION RATIOS")
        print(f"{'─' * 80}")
        for ratio in result.valuation_ratios:
            curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
            print(f"  {ratio.name:<25} {curr:>12}")
    
    # DuPont Analysis
    dupont = result.dupont_analysis
    print(f"\n{'─' * 80}")
    print("DUPONT ANALYSIS (ROE Decomposition)")
    print(f"{'─' * 80}")
    print(f"  ROE = Net Margin × Asset Turnover × Equity Multiplier")
    print(f"  {dupont.roe*100:.1f}% = {dupont.net_margin*100:.1f}% × "
          f"{dupont.asset_turnover:.2f}x × {dupont.equity_multiplier:.2f}x")
    print(f"\n  Primary Driver: {dupont.primary_driver}")
    print(f"  {dupont.analysis}")
    
    # Key Insights
    if result.insights:
        print(f"\n{'─' * 80}")
        print("KEY INSIGHTS")
        print(f"{'─' * 80}")
        for i, insight in enumerate(result.insights, 1):
            print(f"  {i}. {insight}")
    
    # Warnings
    if result.warnings:
        print(f"\n{'─' * 80}")
        print("WARNINGS")
        print(f"{'─' * 80}")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds() * 1000
    
    print(f"\n{'─' * 80}")
    print(f"Phase 7 Complete: {execution_time:.1f}ms")
    print(f"{'─' * 80}")
    
    return result, execution_time


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for Phases 4-7 execution."""
    
    print("\n" + "=" * 80)
    print("FUNDAMENTAL ANALYST AI AGENT - PHASES 4-7 EXECUTION")
    print("=" * 80)
    print("""
  MSc Coursework: IFTE0001 - AI Agents in Asset Management
  Track A: Fundamental Analyst Agent
  
  Executing:
    Phase 4: Cash Flow Analysis      - Net Income to FCF bridge
    Phase 5: Earnings Quality        - Accruals and red flags
    Phase 6: Working Capital         - DSO, DIO, DPO, CCC
    Phase 7: Financial Ratios        - Profitability, liquidity, solvency
    """)
    
    # Get ticker from command line or use default
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    
    total_start = datetime.now()
    
    # First, collect and process data
    print("\n[0] Collecting and processing data...")
    collector = DataCollector()
    raw_data = collector.collect(ticker)
    
    processor = DataProcessor()
    processed_data = processor.process(raw_data)
    
    print(f"  Data collected: {raw_data.validation.years_available} years")
    print(f"  Data processed: {processed_data.num_periods} periods")
    print(f"  Data quality: {processed_data.quality_metrics.coverage_ratio:.1%}")
    
    # Phase 4: Cash Flow Analysis
    cf_result, time_p4 = run_phase4_cash_flow_analysis(processed_data)
    
    # Phase 5: Earnings Quality Analysis
    eq_result, time_p5 = run_phase5_earnings_quality_analysis(processed_data)
    
    # Phase 6: Working Capital Analysis
    wc_result, time_p6 = run_phase6_working_capital_analysis(processed_data)
    
    # Phase 7: Financial Ratios
    ratio_result, time_p7 = run_phase7_financial_ratios(processed_data)
    
    # Summary
    total_time = (datetime.now() - total_start).total_seconds() * 1000
    
    print("\n" + "=" * 80)
    print("PHASES 4-7 EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\n  Company:      {processed_data.company_info.name} ({processed_data.company_info.ticker})")
    print(f"  Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"\n  {'Phase':<35} {'Status':<15} {'Time':<10}")
    print(f"  {'─'*35} {'─'*15} {'─'*10}")
    print(f"  {'Phase 4: Cash Flow Analysis':<35} {'COMPLETE':<15} {time_p4:.1f}ms")
    print(f"  {'Phase 5: Earnings Quality':<35} {'COMPLETE':<15} {time_p5:.1f}ms")
    print(f"  {'Phase 6: Working Capital':<35} {'COMPLETE':<15} {time_p6:.1f}ms")
    print(f"  {'Phase 7: Financial Ratios':<35} {'COMPLETE':<15} {time_p7:.1f}ms")
    print(f"  {'─'*35} {'─'*15} {'─'*10}")
    print(f"  {'TOTAL':<35} {'─':<15} {total_time:.1f}ms")
    
    print(f"\n  Key Results:")
    print(f"    Cash Conversion Rate:    {cf_result.conversion_metrics.cash_conversion_rate*100:.1f}%")
    print(f"    Earnings Quality Score:  {eq_result.overall_score:.0f}/100 ({eq_result.overall_rating.value})")
    print(f"    Cash Conversion Cycle:   {wc_result.cash_conversion_cycle.current_value:.0f} days")
    print(f"    ROE:                     {ratio_result.dupont_analysis.roe*100:.1f}%")
    print(f"    ROIC:                    {ratio_result.key_ratios_summary.get('ROIC', 0)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("PHASES 4-7 COMPLETE")
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

