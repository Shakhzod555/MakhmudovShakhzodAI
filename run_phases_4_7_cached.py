#!/usr/bin/env python3
"""
Phases 4-7 Execution - Using Cached Data Only
No API calls - displays results directly to terminal
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import DataCollector
from data_processor import DataProcessor
from cash_flow_analyzer import CashFlowAnalyzer
from earnings_quality_analyzer import EarningsQualityAnalyzer
from working_capital_analyzer import WorkingCapitalAnalyzer
from ratio_calculator import RatioCalculator

def run_phases_4_7():
    """Run Phases 4-7 using cached data only - terminal output only."""
    
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    
    print("\n" + "="*80)
    print("FUNDAMENTAL ANALYST AI AGENT - PHASES 4-7")
    print("Using Cached Data (No API Calls)")
    print("="*80)
    print(f"\nCompany: {ticker}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Use cached data only
    print("\n[1] Loading cached data...")
    collector = DataCollector()
    raw_data = collector.collect(ticker)
    
    processor = DataProcessor()
    processed_data = processor.process(raw_data)
    
    print(f"    ✓ Data loaded: {raw_data.validation.years_available} years")
    print(f"    ✓ Data quality: {processed_data.quality_metrics.coverage_ratio:.1%}")
    
    # ========== PHASE 4: CASH FLOW ==========
    print("\n" + "="*80)
    print("PHASE 4: CASH FLOW ANALYSIS")
    print("="*80)
    
    cf_analyzer = CashFlowAnalyzer(processed_data)
    cf_result = cf_analyzer.analyze()
    bridge = cf_result.bridge
    
    print(f"\n  CASH FLOW BRIDGE ({cf_result.analysis_period})")
    print(f"  {'─'*60}")
    print(f"  {'Component':<40} {'Amount ($M)':>15}")
    print(f"  {'─'*60}")
    
    for item in bridge.items:
        amount = f"${item.amount:,.0f}M" if item.amount else "N/A"
        prefix = "  " if item.is_subtotal else "    "
        print(f"{prefix}{item.name:<38} {amount:>15}")
    
    print(f"\n  CASH CONVERSION METRICS")
    print(f"  {'─'*60}")
    conv = cf_result.conversion_metrics
    print(f"  Cash Conversion Rate:  {conv.cash_conversion_rate*100:.1f}%")
    print(f"  FCF Margin:            {conv.fcf_margin*100:.1f}%")
    print(f"  Quality Rating:        {conv.quality_rating.value}")
    
    capex = cf_result.capex_analysis
    print(f"\n  CAPEX ANALYSIS")
    print(f"  {'─'*60}")
    print(f"  CapEx: ${capex.capex_amount:,.0f}M | CapEx/D&A: {capex.capex_to_depreciation:.2f}x | Profile: {capex.capex_profile.value}")
    
    # ========== PHASE 5: EARNINGS QUALITY ==========
    print("\n" + "="*80)
    print("PHASE 5: EARNINGS QUALITY ANALYSIS")
    print("="*80)
    
    eq_analyzer = EarningsQualityAnalyzer(processed_data)
    eq_result = eq_analyzer.analyze()
    accruals = eq_result.accruals_analysis
    
    print(f"\n  OVERALL EARNINGS QUALITY")
    print(f"  {'─'*60}")
    print(f"  Rating: {eq_result.overall_rating.value} | Score: {eq_result.overall_score:.0f}/100")
    
    print(f"\n  ACCRUALS ANALYSIS")
    print(f"  {'─'*60}")
    print(f"  Total Accruals: ${accruals.total_accruals:,.0f}M")
    print(f"  Accruals Ratio: {accruals.accruals_ratio*100:.2f}%")
    print(f"  Cash Conversion: {eq_result.cash_conversion_rate*100:.1f}%")
    
    print(f"\n  QUALITY SCORES")
    print(f"  {'─'*60}")
    for score in eq_result.quality_scores:
        print(f"  {score.component:<25}: {score.score:.0f}/100 ({score.weight*100:.0f}%)")
    print(f"  {'─'*60}")
    print(f"  TOTAL: {eq_result.overall_score:.0f}/100")
    
    # ========== PHASE 6: WORKING CAPITAL ==========
    print("\n" + "="*80)
    print("PHASE 6: WORKING CAPITAL ANALYSIS")
    print("="*80)
    
    wc_analyzer = WorkingCapitalAnalyzer(processed_data)
    wc_result = wc_analyzer.analyze()
    
    print(f"\n  EFFICIENCY METRICS")
    print(f"  {'─'*60}")
    print(f"  {'Metric':<35} {'Current':>10} {'Prior':>10}")
    for metric in [wc_result.dso, wc_result.dio, wc_result.dpo]:
        curr = f"{metric.current_value:.1f}d"
        prior = f"{metric.prior_value:.1f}d" if metric.prior_value else "N/A"
        print(f"  {metric.name:<35} {curr:>10} {prior:>10}")
    
    ccc = wc_result.cash_conversion_cycle
    print(f"\n  CASH CONVERSION CYCLE")
    print(f"  {'─'*60}")
    print(f"  CCC = DSO + DIO - DPO = {ccc.dso_contribution:.1f} + {ccc.dio_contribution:.1f} - {ccc.dpo_contribution:.1f}")
    print(f"  CCC = {ccc.current_value:.1f} days ({ccc.trend_direction.value})")
    
    pos = wc_result.position
    print(f"\n  WORKING CAPITAL POSITION")
    print(f"  {'─'*60}")
    print(f"  Net Working Capital:  ${pos.net_working_capital:,.0f}M")
    print(f"  Current Ratio:        {pos.current_ratio:.2f}x")
    print(f"  Quick Ratio:          {pos.quick_ratio:.2f}x")
    
    # ========== PHASE 7: FINANCIAL RATIOS ==========
    print("\n" + "="*80)
    print("PHASE 7: FINANCIAL RATIOS")
    print("="*80)
    
    ratio_calc = RatioCalculator(processed_data)
    ratio_result = ratio_calc.calculate()
    
    print(f"\n  PROFITABILITY")
    print(f"  {'─'*60}")
    for ratio in ratio_result.profitability_ratios:
        curr = f"{ratio.current_value*100:.1f}%" if ratio.current_value else "N/A"
        print(f"  {ratio.name:<30} {curr:>10} [{ratio.interpretation.value}]")
    
    print(f"\n  LIQUIDITY")
    print(f"  {'─'*60}")
    for ratio in ratio_result.liquidity_ratios:
        curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
        print(f"  {ratio.name:<30} {curr:>10}")
    
    print(f"\n  SOLVENCY")
    print(f"  {'─'*60}")
    for ratio in ratio_result.solvency_ratios:
        if ratio.current_value is not None:
            curr = f"{ratio.current_value*100:.1f}%" if ratio.is_percentage else f"{ratio.current_value:.2f}x"
        else:
            curr = "N/A"
        print(f"  {ratio.name:<30} {curr:>10}")
    
    print(f"\n  EFFICIENCY")
    print(f"  {'─'*60}")
    for ratio in ratio_result.efficiency_ratios:
        curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
        print(f"  {ratio.name:<30} {curr:>10}")
    
    dupont = ratio_result.dupont_analysis
    print(f"\n  DUPONT ANALYSIS (ROE Decomposition)")
    print(f"  {'─'*60}")
    print(f"  ROE = Net Margin × Asset Turnover × Equity Multiplier")
    print(f"  {dupont.roe*100:.1f}% = {dupont.net_margin*100:.1f}% × {dupont.asset_turnover:.2f}x × {dupont.equity_multiplier:.2f}x")
    print(f"  Primary Driver: {dupont.primary_driver}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("SUMMARY - PHASES 4-7")
    print("="*80)
    print(f"""
  Company:  {processed_data.company_info.name} ({processed_data.company_info.ticker})
  
  KEY METRICS:
    ├─ Cash Conversion Rate:    {cf_result.conversion_metrics.cash_conversion_rate*100:.1f}%
    ├─ Earnings Quality:        {eq_result.overall_score:.0f}/100 ({eq_result.overall_rating.value})
    ├─ Cash Conversion Cycle:   {wc_result.cash_conversion_cycle.current_value:.0f} days
    ├─ ROE:                     {ratio_result.dupont_analysis.roe*100:.1f}%
    └─ ROIC:                    {ratio_result.key_ratios_summary.get('ROIC', 0)*100:.1f}%
  
  STATUS: COMPLETE (using cached data)
    """)
    print("="*80)

if __name__ == "__main__":
    run_phases_4_7()

