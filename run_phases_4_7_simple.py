#!/usr/bin/env python3
"""
Phases 4-7 Execution Script - Fundamental Analyst AI Agent
Simple version with fixed output handling
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

ALPHA_VANTAGE_API_KEY = "MB9136UVECFVPN76"
os.environ["ALPHA_VANTAGE_API_KEY"] = ALPHA_VANTAGE_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import DataCollector
from data_processor import DataProcessor
from cash_flow_analyzer import CashFlowAnalyzer
from earnings_quality_analyzer import EarningsQualityAnalyzer
from working_capital_analyzer import WorkingCapitalAnalyzer
from ratio_calculator import RatioCalculator

def safe_get_value(val, default="N/A"):
    """Safely get value from various types."""
    if val is None:
        return default
    if hasattr(val, 'value'):
        return val.value
    return val

def run_phases_4_7():
    """Run Phases 4-7 and display detailed output."""
    
    print("\n" + "="*80)
    print("FUNDAMENTAL ANALYST AI AGENT - PHASES 4-7")
    print("="*80)
    
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    
    # Data collection
    print("\n[0] Collecting and processing data...")
    collector = DataCollector()
    raw_data = collector.collect(ticker)
    
    processor = DataProcessor()
    processed_data = processor.process(raw_data)
    
    print(f"  Data: {raw_data.validation.years_available} years, Quality: {processed_data.quality_metrics.coverage_ratio:.1%}")
    
    # ========== PHASE 4: CASH FLOW ANALYSIS ==========
    print("\n" + "="*80)
    print("PHASE 4: CASH FLOW ANALYSIS")
    print("="*80)
    
    cf_analyzer = CashFlowAnalyzer(processed_data)
    cf_result = cf_analyzer.analyze()
    bridge = cf_result.bridge
    
    print(f"\nCash Flow Bridge ({cf_result.analysis_period}):")
    print(f"  {'Component':<35} {'Amount ($M)':>15} {'% of NI':>10}")
    print(f"  {'-'*35} {'-'*15} {'-'*10}")
    
    for item in bridge.items:
        amount = f"${item.amount:,.1f}" if item.amount else "N/A"
        pct = f"{item.as_percent_of_ni:.1f}%" if item.as_percent_of_ni else ""
        prefix = "  " if item.is_subtotal else "    "
        print(f"{prefix}{item.name:<33} {amount:>15} {pct:>10}")
    
    conv = cf_result.conversion_metrics
    print(f"\nCash Conversion Metrics:")
    print(f"  Cash Conversion Rate:  {conv.cash_conversion_rate*100:.1f}%")
    print(f"  FCF Margin:            {conv.fcf_margin*100:.1f}%")
    print(f"  Quality Rating:        {safe_get_value(conv.quality_rating)}")
    
    capex = cf_result.capex_analysis
    print(f"\nCapEx Analysis:")
    print(f"  CapEx Amount:          ${capex.capex_amount:,.1f}M")
    print(f"  CapEx / Revenue:       {capex.capex_to_revenue*100:.1f}%")
    print(f"  CapEx / D&A:           {capex.capex_to_depreciation:.2f}x")
    print(f"  Profile:               {safe_get_value(capex.capex_profile)}")
    
    if cf_result.insights:
        print(f"\nKey Insights:")
        for i, insight in enumerate(cf_result.insights, 1):
            print(f"  {i}. {insight}")
    
    # ========== PHASE 5: EARNINGS QUALITY ==========
    print("\n" + "="*80)
    print("PHASE 5: EARNINGS QUALITY ANALYSIS")
    print("="*80)
    
    eq_analyzer = EarningsQualityAnalyzer(processed_data)
    eq_result = eq_analyzer.analyze()
    accruals = eq_result.accruals_analysis
    
    print(f"\nOverall Earnings Quality:")
    print(f"  Rating:   {safe_get_value(eq_result.overall_rating)}")
    print(f"  Score:    {eq_result.overall_score:.0f}/100")
    
    print(f"\nAccruals Analysis:")
    print(f"  Total Accruals:     ${accruals.total_accruals:,.1f}M")
    print(f"  Accruals Ratio:     {accruals.accruals_ratio*100:.2f}%")
    print(f"  Rating:             {safe_get_value(accruals.accruals_rating)}")
    
    print(f"\nCash Conversion: {eq_result.cash_conversion_rate*100:.1f}%")
    
    print(f"\nQuality Score Breakdown:")
    print(f"  {'Component':<25} {'Score':>8} {'Weight':>8}")
    for score in eq_result.quality_scores:
        print(f"  {score.component:<25} {score.score:>8.0f} {score.weight*100:>7.0f}%")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<25} {eq_result.overall_score:>8.0f} {'100':>8}%")
    
    if eq_result.positive_indicators:
        print(f"\nPositive Indicators:")
        for ind in eq_result.positive_indicators:
            print(f"  + {ind}")
    
    if eq_result.insights:
        print(f"\nKey Insights:")
        for i, insight in enumerate(eq_result.insights, 1):
            print(f"  {i}. {insight}")
    
    # ========== PHASE 6: WORKING CAPITAL ==========
    print("\n" + "="*80)
    print("PHASE 6: WORKING CAPITAL ANALYSIS")
    print("="*80)
    
    wc_analyzer = WorkingCapitalAnalyzer(processed_data)
    wc_result = wc_analyzer.analyze()
    
    print(f"\nEfficiency Metrics:")
    print(f"  {'Metric':<30} {'Current':>10} {'Prior':>10} {'Rating':<12}")
    for metric in [wc_result.dso, wc_result.dio, wc_result.dpo]:
        curr = f"{metric.current_value:.1f}"
        prior = f"{metric.prior_value:.1f}" if metric.prior_value else "N/A"
        print(f"  {metric.name:<30} {curr:>10} {prior:>10} {safe_get_value(metric.rating):<12}")
    
    ccc = wc_result.cash_conversion_cycle
    print(f"\nCash Conversion Cycle:")
    print(f"  CCC = DSO + DIO - DPO")
    print(f"  CCC = {ccc.dso_contribution:.1f} + {ccc.dio_contribution:.1f} - {ccc.dpo_contribution:.1f}")
    print(f"  CCC = {ccc.current_value:.1f} days")
    print(f"  Trend: {safe_get_value(ccc.trend_direction)}")
    
    pos = wc_result.position
    print(f"\nWorking Capital Position:")
    print(f"  Net Working Capital:  ${pos.net_working_capital:,.1f}M")
    print(f"  Current Ratio:        {pos.current_ratio:.2f}x")
    print(f"  Quick Ratio:          {pos.quick_ratio:.2f}x")
    
    if wc_result.insights:
        print(f"\nKey Insights:")
        for i, insight in enumerate(wc_result.insights, 1):
            print(f"  {i}. {insight}")
    
    # ========== PHASE 7: FINANCIAL RATIOS ==========
    print("\n" + "="*80)
    print("PHASE 7: FINANCIAL RATIOS")
    print("="*80)
    
    ratio_calc = RatioCalculator(processed_data)
    ratio_result = ratio_calc.calculate()
    
    print(f"\nProfitability Ratios:")
    print(f"  {'Ratio':<25} {'Current':>12} {'Interpretation':<15}")
    for ratio in ratio_result.profitability_ratios:
        curr = f"{ratio.current_value*100:.1f}%" if ratio.current_value else "N/A"
        print(f"  {ratio.name:<25} {curr:>12} {safe_get_value(ratio.interpretation):<15}")
    
    print(f"\nLiquidity Ratios:")
    for ratio in ratio_result.liquidity_ratios:
        curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
        print(f"  {ratio.name:<25} {curr:>12}")
    
    print(f"\nSolvency Ratios:")
    for ratio in ratio_result.solvency_ratios:
        if ratio.current_value is not None:
            curr = f"{ratio.current_value*100:.1f}%" if ratio.is_percentage else f"{ratio.current_value:.2f}x"
        else:
            curr = "N/A"
        print(f"  {ratio.name:<25} {curr:>12}")
    
    print(f"\nEfficiency Ratios:")
    for ratio in ratio_result.efficiency_ratios:
        curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
        print(f"  {ratio.name:<25} {curr:>12}")
    
    if ratio_result.valuation_ratios:
        print(f"\nValuation Ratios:")
        for ratio in ratio_result.valuation_ratios:
            curr = f"{ratio.current_value:.2f}x" if ratio.current_value else "N/A"
            print(f"  {ratio.name:<25} {curr:>12}")
    
    dupont = ratio_result.dupont_analysis
    print(f"\nDuPont Analysis (ROE Decomposition):")
    print(f"  ROE = Net Margin × Asset Turnover × Equity Multiplier")
    print(f"  {dupont.roe*100:.1f}% = {dupont.net_margin*100:.1f}% × {dupont.asset_turnover:.2f}x × {dupont.equity_multiplier:.2f}x")
    print(f"  Primary Driver: {dupont.primary_driver}")
    
    if ratio_result.insights:
        print(f"\nKey Insights:")
        for i, insight in enumerate(ratio_result.insights, 1):
            print(f"  {i}. {insight}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("PHASES 4-7 SUMMARY")
    print("="*80)
    print(f"\n  Company:  {processed_data.company_info.name} ({processed_data.company_info.ticker})")
    print(f"\n  Key Results:")
    print(f"    Cash Conversion Rate:    {cf_result.conversion_metrics.cash_conversion_rate*100:.1f}%")
    print(f"    Earnings Quality Score:  {eq_result.overall_score:.0f}/100 ({safe_get_value(eq_result.overall_rating)})")
    print(f"    Cash Conversion Cycle:   {wc_result.cash_conversion_cycle.current_value:.0f} days")
    print(f"    ROE:                     {ratio_result.dupont_analysis.roe*100:.1f}%")
    roic_val = ratio_result.key_ratios_summary.get('ROIC', 0)
    print(f"    ROIC:                    {(roic_val*100 if roic_val else 0):.1f}%")
    print("\n" + "="*80)
    print("PHASES 4-7 COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        run_phases_4_7()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

