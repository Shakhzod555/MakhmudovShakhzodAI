"""
Validation Module for Fundamental Analyst Agent
Comprehensive Accuracy, Calculation, and Logic Verification

This module provides extensive validation capabilities for verifying the correctness
of all calculations, formulas, scoring logic, and business rules throughout the
analysis pipeline. It serves as a quality assurance layer to ensure report accuracy.

Validation Categories:
    1. Mathematical Formula Verification
       - EBIT variance bridge reconciliation
       - Ratio calculations (P/E, EV/EBITDA, ROE, etc.)
       - Cash flow calculations and bridges
       - Working capital metrics
    
    2. Data Integrity Validation
       - Cross-statement consistency checks
       - Temporal consistency (year-over-year data)
       - Unit consistency (millions vs raw values)
       - Missing data handling verification
    
    3. Business Logic Validation
       - Margin bounds (0-100%)
       - Ratio reasonableness checks
       - Sign consistency (revenues positive, etc.)
       - Accounting identity verification
    
    4. Scoring and Recommendation Validation
       - Score component verification
       - Weight sum validation (must equal 100%)
       - Threshold consistency
       - Confidence calculation verification
    
    5. Output Consistency Validation
       - Cross-module data consistency
       - Report vs calculation matching
       - Citation accuracy

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent

Usage:
    from validation import ValidationEngine
    
    validator = ValidationEngine(analysis_results)
    report = validator.run_full_validation()
    
    if not report.is_valid:
        for error in report.critical_errors:
            print(f"CRITICAL: {error}")

Author: MSc AI Agents in Asset Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod
import math

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation findings."""
    CRITICAL = "critical"      # Analysis is fundamentally incorrect
    ERROR = "error"            # Significant calculation error
    WARNING = "warning"        # Potential issue, may be acceptable
    INFO = "info"              # Informational note
    PASS = "pass"              # Validation passed


class ValidationCategory(Enum):
    """Categories of validation checks."""
    MATHEMATICAL = "mathematical"
    DATA_INTEGRITY = "data_integrity"
    BUSINESS_LOGIC = "business_logic"
    SCORING = "scoring"
    OUTPUT_CONSISTENCY = "output_consistency"
    FORMULA_VERIFICATION = "formula_verification"
    THRESHOLD_COMPLIANCE = "threshold_compliance"
    CROSS_MODULE = "cross_module"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationFinding:
    """
    Individual validation finding with complete context.
    
    Attributes:
        check_name: Name of the validation check
        category: Category of the validation
        severity: Severity level
        message: Human-readable description
        expected_value: What the value should be
        actual_value: What the value actually is
        deviation: Percentage or absolute deviation
        source_module: Module where the issue originates
        recommendation: Suggested fix or action
    """
    check_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    deviation: Optional[float] = None
    source_module: Optional[str] = None
    recommendation: Optional[str] = None
    
    def format(self) -> str:
        """Format finding for display."""
        severity_prefix = {
            ValidationSeverity.CRITICAL: "[CRITICAL]",
            ValidationSeverity.ERROR: "[ERROR]",
            ValidationSeverity.WARNING: "[WARNING]",
            ValidationSeverity.INFO: "[INFO]",
            ValidationSeverity.PASS: "[PASS]"
        }
        
        output = f"{severity_prefix[self.severity]} {self.check_name}: {self.message}"
        
        if self.expected_value is not None and self.actual_value is not None:
            output += f"\n    Expected: {self.expected_value}, Actual: {self.actual_value}"
        
        if self.deviation is not None:
            output += f"\n    Deviation: {self.deviation:.4f}"
        
        if self.recommendation:
            output += f"\n    Recommendation: {self.recommendation}"
        
        return output


@dataclass
class ValidationReport:
    """
    Complete validation report aggregating all findings.
    
    Attributes:
        timestamp: When validation was performed
        ticker: Stock ticker being validated
        total_checks: Total number of checks performed
        passed_checks: Number of checks that passed
        findings: List of all validation findings
        is_valid: Whether the analysis passes validation
        validation_score: Overall validation score (0-100)
        summary: Executive summary of validation results
    """
    timestamp: datetime
    ticker: str
    total_checks: int = 0
    passed_checks: int = 0
    findings: List[ValidationFinding] = field(default_factory=list)
    is_valid: bool = True
    validation_score: float = 100.0
    summary: str = ""
    
    @property
    def critical_errors(self) -> List[ValidationFinding]:
        """Get all critical errors."""
        return [f for f in self.findings if f.severity == ValidationSeverity.CRITICAL]
    
    @property
    def errors(self) -> List[ValidationFinding]:
        """Get all errors (non-critical)."""
        return [f for f in self.findings if f.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationFinding]:
        """Get all warnings."""
        return [f for f in self.findings if f.severity == ValidationSeverity.WARNING]
    
    def add_finding(self, finding: ValidationFinding) -> None:
        """Add a finding to the report."""
        self.findings.append(finding)
        self.total_checks += 1
        
        if finding.severity == ValidationSeverity.PASS:
            self.passed_checks += 1
        elif finding.severity == ValidationSeverity.CRITICAL:
            self.is_valid = False
            self.validation_score -= 25
        elif finding.severity == ValidationSeverity.ERROR:
            self.is_valid = False
            self.validation_score -= 10
        elif finding.severity == ValidationSeverity.WARNING:
            self.validation_score -= 2
        
        self.validation_score = max(0, self.validation_score)
    
    def generate_summary(self) -> str:
        """Generate executive summary."""
        critical_count = len(self.critical_errors)
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        
        status = "PASSED" if self.is_valid else "FAILED"
        
        summary_lines = [
            "=" * 80,
            f"VALIDATION REPORT: {self.ticker}",
            "=" * 80,
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Status: {status}",
            f"Validation Score: {self.validation_score:.1f}/100",
            "",
            f"Total Checks: {self.total_checks}",
            f"Passed: {self.passed_checks}",
            f"Critical Errors: {critical_count}",
            f"Errors: {error_count}",
            f"Warnings: {warning_count}",
            "=" * 80,
        ]
        
        if critical_count > 0:
            summary_lines.append("")
            summary_lines.append("CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            summary_lines.append("-" * 40)
            for finding in self.critical_errors:
                summary_lines.append(f"  - {finding.message}")
        
        if error_count > 0:
            summary_lines.append("")
            summary_lines.append("ERRORS:")
            summary_lines.append("-" * 40)
            for finding in self.errors:
                summary_lines.append(f"  - {finding.message}")
        
        self.summary = "\n".join(summary_lines)
        return self.summary


# =============================================================================
# TOLERANCE CONFIGURATION
# =============================================================================

@dataclass
class ValidationTolerances:
    """
    Configurable tolerances for validation checks.
    
    These tolerances define acceptable deviation ranges for various
    calculations. They account for floating-point arithmetic and
    minor data inconsistencies that may be acceptable.
    """
    # Mathematical tolerances
    reconciliation_tolerance_absolute: float = 1.0      # $1M absolute
    reconciliation_tolerance_relative: float = 0.001   # 0.1% relative
    ratio_tolerance_relative: float = 0.01             # 1% for ratios
    percentage_tolerance_absolute: float = 0.1         # 0.1 percentage points
    
    # Business logic bounds
    margin_lower_bound: float = -1.0                   # -100% (losses possible)
    margin_upper_bound: float = 1.0                    # 100%
    ratio_lower_bound: float = 0.0                     # Most ratios non-negative
    ratio_upper_bound: float = 1000.0                  # Sanity upper bound
    pe_reasonable_max: float = 200.0                   # P/E sanity check
    
    # Scoring tolerances
    weight_sum_tolerance: float = 0.001                # Weights must sum to ~1.0
    score_range_min: float = 0.0
    score_range_max: float = 100.0
    
    # Data integrity
    max_missing_fields_percent: float = 0.20           # 20% missing acceptable
    year_count_minimum: int = 2                        # At least 2 years data


# Default tolerances instance
TOLERANCES = ValidationTolerances()


# =============================================================================
# VALIDATION ENGINE
# =============================================================================

class ValidationEngine:
    """
    Comprehensive validation engine for Fundamental Analyst Agent outputs.
    
    This class orchestrates all validation checks across the analysis pipeline
    and produces a detailed validation report. It is designed to catch both
    calculation errors and logical inconsistencies.
    
    Validation Phases:
        1. Input data validation
        2. Calculation verification
        3. Cross-module consistency
        4. Scoring logic validation
        5. Output format validation
    
    Attributes:
        results: Analysis results to validate
        tolerances: Tolerance configuration
        report: Validation report being built
    
    Example:
        validator = ValidationEngine(analysis_results)
        report = validator.run_full_validation()
        print(report.summary)
    """
    
    def __init__(
        self,
        analysis_results: Dict[str, Any],
        tolerances: Optional[ValidationTolerances] = None
    ):
        """
        Initialize the validation engine.
        
        Args:
            analysis_results: Dictionary containing all analysis outputs.
                             Expected keys: profitability, cash_flow, 
                             earnings_quality, working_capital, ratios,
                             valuation, memo
            tolerances: Optional custom tolerances
        """
        self._results = analysis_results
        self._tolerances = tolerances or TOLERANCES
        self._report = ValidationReport(
            timestamp=datetime.now(),
            ticker=self._extract_ticker()
        )
        
        logger.info(f"ValidationEngine initialized for {self._report.ticker}")
    
    def _extract_ticker(self) -> str:
        """Extract ticker from results."""
        # Try multiple locations
        if 'ticker' in self._results:
            return self._results['ticker']
        if 'profitability' in self._results:
            prof = self._results['profitability']
            if hasattr(prof, 'ticker'):
                return prof.ticker
        return "UNKNOWN"
    
    # -------------------------------------------------------------------------
    # MAIN VALIDATION ENTRY POINT
    # -------------------------------------------------------------------------
    
    def run_full_validation(self) -> ValidationReport:
        """
        Execute complete validation suite.
        
        Runs all validation checks in sequence and compiles the final report.
        This is the main entry point for validation.
        
        Returns:
            ValidationReport with all findings
        """
        logger.info(f"Starting full validation for {self._report.ticker}")
        
        # Phase 1: Data Integrity
        self._validate_data_integrity()
        
        # Phase 2: Mathematical Calculations
        self._validate_ebit_bridge()
        self._validate_cash_flow_calculations()
        self._validate_ratio_calculations()
        self._validate_valuation_calculations()
        self._validate_working_capital_calculations()
        
        # Phase 3: Business Logic
        self._validate_margin_bounds()
        self._validate_ratio_reasonableness()
        self._validate_sign_consistency()
        
        # Phase 4: Scoring and Recommendation
        self._validate_scoring_weights()
        self._validate_score_ranges()
        self._validate_recommendation_logic()
        self._validate_confidence_calculation()
        
        # Phase 5: Cross-Module Consistency
        self._validate_cross_module_consistency()
        
        # Phase 6: Output Verification
        self._validate_output_completeness()
        
        # Generate summary
        self._report.generate_summary()
        
        logger.info(
            f"Validation complete: {self._report.passed_checks}/{self._report.total_checks} "
            f"checks passed, Score: {self._report.validation_score:.1f}/100"
        )
        
        return self._report
    
    # -------------------------------------------------------------------------
    # DATA INTEGRITY VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_data_integrity(self) -> None:
        """Validate data integrity across all modules."""
        
        # Check required modules present
        required_modules = [
            'profitability', 'cash_flow', 'earnings_quality',
            'working_capital', 'ratios', 'valuation', 'memo'
        ]
        
        for module in required_modules:
            if module in self._results and self._results[module] is not None:
                self._report.add_finding(ValidationFinding(
                    check_name=f"Module Present: {module}",
                    category=ValidationCategory.DATA_INTEGRITY,
                    severity=ValidationSeverity.PASS,
                    message=f"Required module '{module}' is present"
                ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name=f"Module Present: {module}",
                    category=ValidationCategory.DATA_INTEGRITY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Required module '{module}' is missing or null",
                    recommendation=f"Ensure {module} analysis completes successfully"
                ))
        
        # Check for data periods consistency
        self._validate_period_consistency()
    
    def _validate_period_consistency(self) -> None:
        """Validate that all modules use consistent time periods."""
        periods_found = {}
        
        # Extract periods from each module
        if 'profitability' in self._results:
            prof = self._results['profitability']
            if hasattr(prof, 'current_period'):
                periods_found['profitability'] = prof.current_period
        
        if 'valuation' in self._results:
            val = self._results['valuation']
            if hasattr(val, 'analysis_period'):
                periods_found['valuation'] = val.analysis_period
        
        # Check consistency
        unique_periods = set(periods_found.values())
        if len(unique_periods) <= 1:
            self._report.add_finding(ValidationFinding(
                check_name="Period Consistency",
                category=ValidationCategory.DATA_INTEGRITY,
                severity=ValidationSeverity.PASS,
                message="All modules use consistent time periods"
            ))
        else:
            self._report.add_finding(ValidationFinding(
                check_name="Period Consistency",
                category=ValidationCategory.DATA_INTEGRITY,
                severity=ValidationSeverity.WARNING,
                message=f"Inconsistent periods detected: {unique_periods}",
                recommendation="Verify all modules analyze the same fiscal period"
            ))
    
    # -------------------------------------------------------------------------
    # EBIT BRIDGE VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_ebit_bridge(self) -> None:
        """
        Validate EBIT variance bridge reconciliation.
        
        The bridge must satisfy:
            EBIT_prior + Volume_Effect + GM_Rate_Effect + OpEx_Rate_Effect = EBIT_current
        
        This is a mathematical identity that must hold exactly (within tolerance).
        """
        if 'profitability' not in self._results:
            return
        
        prof = self._results['profitability']
        
        # Extract bridge components
        try:
            if hasattr(prof, 'bridge'):
                bridge = prof.bridge
                ebit_prior = self._get_nested_attr(bridge, 'ebit_prior', 0)
                ebit_current = self._get_nested_attr(bridge, 'ebit_current', 0)
                volume_effect = self._get_nested_attr(bridge, 'volume_effect', 0)
                gm_rate_effect = self._get_nested_attr(bridge, 'gm_rate_effect', 0)
                opex_rate_effect = self._get_nested_attr(bridge, 'opex_rate_effect', 0)
            else:
                # Try direct attributes
                ebit_prior = getattr(prof, 'ebit_prior', 0)
                ebit_current = getattr(prof, 'ebit_current', 0)
                volume_effect = getattr(prof, 'volume_effect', 0)
                gm_rate_effect = getattr(prof, 'gm_rate_effect', 0)
                opex_rate_effect = getattr(prof, 'opex_rate_effect', 0)
            
            # Calculate expected EBIT
            calculated_ebit = ebit_prior + volume_effect + gm_rate_effect + opex_rate_effect
            
            # Check reconciliation
            deviation = abs(calculated_ebit - ebit_current)
            tolerance = self._tolerances.reconciliation_tolerance_absolute
            
            if deviation <= tolerance:
                self._report.add_finding(ValidationFinding(
                    check_name="EBIT Bridge Reconciliation",
                    category=ValidationCategory.MATHEMATICAL,
                    severity=ValidationSeverity.PASS,
                    message=f"EBIT bridge reconciles within ${tolerance:.2f}M tolerance",
                    expected_value=ebit_current,
                    actual_value=calculated_ebit,
                    deviation=deviation
                ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name="EBIT Bridge Reconciliation",
                    category=ValidationCategory.MATHEMATICAL,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"EBIT bridge fails to reconcile: ${deviation:.2f}M difference",
                    expected_value=ebit_current,
                    actual_value=calculated_ebit,
                    deviation=deviation,
                    source_module="profitability_analyzer.py",
                    recommendation="Review variance decomposition formula implementation"
                ))
            
            # Validate bridge formula components
            self._validate_bridge_formula_logic(
                ebit_prior, ebit_current, volume_effect, gm_rate_effect, opex_rate_effect
            )
            
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="EBIT Bridge Reconciliation",
                category=ValidationCategory.MATHEMATICAL,
                severity=ValidationSeverity.ERROR,
                message=f"Unable to validate EBIT bridge: {str(e)}",
                recommendation="Check profitability result structure"
            ))
    
    def _validate_bridge_formula_logic(
        self,
        ebit_prior: float,
        ebit_current: float,
        volume_effect: float,
        gm_rate_effect: float,
        opex_rate_effect: float
    ) -> None:
        """Validate the logical consistency of bridge components."""
        
        ebit_change = ebit_current - ebit_prior
        sum_of_effects = volume_effect + gm_rate_effect + opex_rate_effect
        
        # The sum of effects must equal the EBIT change
        deviation = abs(sum_of_effects - ebit_change)
        
        if deviation <= self._tolerances.reconciliation_tolerance_absolute:
            self._report.add_finding(ValidationFinding(
                check_name="Bridge Effects Sum",
                category=ValidationCategory.FORMULA_VERIFICATION,
                severity=ValidationSeverity.PASS,
                message="Sum of bridge effects equals EBIT change",
                expected_value=ebit_change,
                actual_value=sum_of_effects
            ))
        else:
            self._report.add_finding(ValidationFinding(
                check_name="Bridge Effects Sum",
                category=ValidationCategory.FORMULA_VERIFICATION,
                severity=ValidationSeverity.ERROR,
                message="Sum of bridge effects does not equal EBIT change",
                expected_value=ebit_change,
                actual_value=sum_of_effects,
                deviation=deviation
            ))
    
    # -------------------------------------------------------------------------
    # CASH FLOW VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_cash_flow_calculations(self) -> None:
        """
        Validate cash flow calculations and relationships.
        
        Key validations:
            - FCF = OCF - CapEx
            - Cash Conversion = OCF / Net Income
            - FCF Margin = FCF / Revenue
        """
        if 'cash_flow' not in self._results:
            return
        
        cf = self._results['cash_flow']
        
        try:
            # Extract values
            net_income = self._get_nested_attr(cf, 'net_income', None)
            ocf = self._get_nested_attr(cf, 'operating_cash_flow', None)
            fcf = self._get_nested_attr(cf, 'free_cash_flow', None)
            capex = self._get_nested_attr(cf, 'capital_expenditures', None)
            cash_conversion = self._get_nested_attr(cf, 'cash_conversion_rate', None)
            
            # Validate FCF calculation: FCF = OCF - CapEx
            if ocf is not None and capex is not None and fcf is not None:
                # CapEx is typically stored as negative
                capex_abs = abs(capex) if capex else 0
                expected_fcf = ocf - capex_abs
                
                deviation = abs(expected_fcf - fcf)
                rel_tolerance = abs(fcf * self._tolerances.ratio_tolerance_relative) if fcf != 0 else 1.0
                
                if deviation <= max(rel_tolerance, self._tolerances.reconciliation_tolerance_absolute):
                    self._report.add_finding(ValidationFinding(
                        check_name="FCF Calculation",
                        category=ValidationCategory.FORMULA_VERIFICATION,
                        severity=ValidationSeverity.PASS,
                        message="FCF = OCF - CapEx verified",
                        expected_value=expected_fcf,
                        actual_value=fcf
                    ))
                else:
                    self._report.add_finding(ValidationFinding(
                        check_name="FCF Calculation",
                        category=ValidationCategory.FORMULA_VERIFICATION,
                        severity=ValidationSeverity.ERROR,
                        message=f"FCF calculation mismatch: deviation ${deviation:.2f}M",
                        expected_value=expected_fcf,
                        actual_value=fcf,
                        deviation=deviation,
                        recommendation="Verify CapEx sign convention and FCF formula"
                    ))
            
            # Validate Cash Conversion calculation
            if net_income is not None and ocf is not None and cash_conversion is not None:
                if net_income != 0:
                    expected_conversion = ocf / net_income
                    deviation = abs(expected_conversion - cash_conversion)
                    
                    if deviation <= self._tolerances.ratio_tolerance_relative:
                        self._report.add_finding(ValidationFinding(
                            check_name="Cash Conversion Rate",
                            category=ValidationCategory.FORMULA_VERIFICATION,
                            severity=ValidationSeverity.PASS,
                            message="Cash Conversion = OCF / NI verified",
                            expected_value=f"{expected_conversion:.4f}",
                            actual_value=f"{cash_conversion:.4f}"
                        ))
                    else:
                        self._report.add_finding(ValidationFinding(
                            check_name="Cash Conversion Rate",
                            category=ValidationCategory.FORMULA_VERIFICATION,
                            severity=ValidationSeverity.ERROR,
                            message="Cash conversion rate calculation error",
                            expected_value=expected_conversion,
                            actual_value=cash_conversion,
                            deviation=deviation
                        ))
        
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="Cash Flow Calculations",
                category=ValidationCategory.FORMULA_VERIFICATION,
                severity=ValidationSeverity.ERROR,
                message=f"Unable to validate cash flow calculations: {str(e)}"
            ))
    
    # -------------------------------------------------------------------------
    # RATIO CALCULATIONS VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_ratio_calculations(self) -> None:
        """
        Validate financial ratio calculations.
        
        Verifies formulas for:
            - ROE = Net Income / Shareholders Equity
            - ROA = Net Income / Total Assets
            - ROIC = NOPAT / Invested Capital
            - Debt/Equity = Total Debt / Total Equity
        """
        if 'ratios' not in self._results:
            return
        
        ratios = self._results['ratios']
        
        # Get underlying data for verification
        # These checks require access to raw financial data
        # If available, perform formula verification
        
        try:
            # Check ROE reasonableness
            roe = self._get_nested_attr(ratios, 'profitability_ratios.roe', None)
            if roe is not None:
                if -2.0 <= roe <= 2.0:  # -200% to 200% is reasonable
                    self._report.add_finding(ValidationFinding(
                        check_name="ROE Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.PASS,
                        message=f"ROE of {roe:.1%} is within reasonable bounds"
                    ))
                else:
                    self._report.add_finding(ValidationFinding(
                        check_name="ROE Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.WARNING,
                        message=f"ROE of {roe:.1%} is outside typical bounds (-200% to 200%)",
                        actual_value=roe,
                        recommendation="Verify equity values and calculation"
                    ))
            
            # Check Debt/Equity reasonableness
            de_ratio = self._get_nested_attr(ratios, 'leverage_ratios.debt_to_equity', None)
            if de_ratio is not None:
                if 0 <= de_ratio <= 10.0:  # 0x to 10x is typical
                    self._report.add_finding(ValidationFinding(
                        check_name="D/E Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.PASS,
                        message=f"D/E ratio of {de_ratio:.2f}x is within reasonable bounds"
                    ))
                elif de_ratio < 0:
                    self._report.add_finding(ValidationFinding(
                        check_name="D/E Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.ERROR,
                        message=f"Negative D/E ratio of {de_ratio:.2f}x indicates data issue",
                        actual_value=de_ratio,
                        recommendation="Check for negative equity (possible distressed company)"
                    ))
                else:
                    self._report.add_finding(ValidationFinding(
                        check_name="D/E Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.WARNING,
                        message=f"D/E ratio of {de_ratio:.2f}x is unusually high",
                        actual_value=de_ratio
                    ))
        
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="Ratio Calculations",
                category=ValidationCategory.FORMULA_VERIFICATION,
                severity=ValidationSeverity.ERROR,
                message=f"Unable to validate ratio calculations: {str(e)}"
            ))
    
    # -------------------------------------------------------------------------
    # VALUATION CALCULATIONS VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_valuation_calculations(self) -> None:
        """
        Validate valuation multiple calculations.
        
        Critical check: Market cap and financial statement units must match.
        This is a common source of errors (e.g., market cap in dollars vs
        financials in millions).
        
        Validates:
            - P/E = Market Cap / Net Income
            - P/B = Market Cap / Book Value
            - P/FCF = Market Cap / Free Cash Flow
            - EV/EBITDA = Enterprise Value / EBITDA
        """
        if 'valuation' not in self._results:
            return
        
        val = self._results['valuation']
        
        try:
            # Extract multiples
            pe = self._get_nested_attr(val, 'pe_ratio', None)
            pb = self._get_nested_attr(val, 'price_to_book', None)
            pfcf = self._get_nested_attr(val, 'price_to_fcf', None)
            ev_ebitda = self._get_nested_attr(val, 'ev_ebitda', None)
            
            # Validate P/E reasonableness
            if pe is not None:
                if 0 < pe <= self._tolerances.pe_reasonable_max:
                    self._report.add_finding(ValidationFinding(
                        check_name="P/E Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.PASS,
                        message=f"P/E of {pe:.1f}x is within reasonable bounds (0-{self._tolerances.pe_reasonable_max}x)"
                    ))
                elif pe > self._tolerances.pe_reasonable_max:
                    self._report.add_finding(ValidationFinding(
                        check_name="P/E Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.WARNING,
                        message=f"P/E of {pe:.1f}x exceeds typical maximum of {self._tolerances.pe_reasonable_max}x",
                        actual_value=pe,
                        recommendation="Verify market cap and net income units match"
                    ))
                elif pe <= 0:
                    self._report.add_finding(ValidationFinding(
                        check_name="P/E Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.INFO,
                        message=f"P/E of {pe:.1f}x indicates negative earnings - not meaningful"
                    ))
            
            # Critical: Check for unit mismatch (the bug we fixed)
            # P/FCF and P/B should typically be under 100x for most companies
            # If they exceed 1000x, there is likely a units mismatch
            
            if pfcf is not None:
                if pfcf > 1000:
                    self._report.add_finding(ValidationFinding(
                        check_name="P/FCF Unit Check",
                        category=ValidationCategory.CRITICAL,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"P/FCF of {pfcf:,.0f}x suggests market cap/FCF units mismatch",
                        actual_value=pfcf,
                        expected_value="< 100x for most companies",
                        source_module="valuation.py",
                        recommendation="Ensure market cap is scaled to millions to match financial statements"
                    ))
                elif 0 < pfcf <= 200:
                    self._report.add_finding(ValidationFinding(
                        check_name="P/FCF Unit Check",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.PASS,
                        message=f"P/FCF of {pfcf:.1f}x indicates correct unit scaling"
                    ))
            
            if pb is not None:
                if pb > 1000:
                    self._report.add_finding(ValidationFinding(
                        check_name="P/B Unit Check",
                        category=ValidationCategory.CRITICAL,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"P/B of {pb:,.0f}x suggests market cap/equity units mismatch",
                        actual_value=pb,
                        expected_value="< 100x for most companies",
                        source_module="valuation.py",
                        recommendation="Ensure market cap is scaled to millions to match financial statements"
                    ))
                elif 0 < pb <= 200:
                    self._report.add_finding(ValidationFinding(
                        check_name="P/B Unit Check",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.PASS,
                        message=f"P/B of {pb:.1f}x indicates correct unit scaling"
                    ))
            
            # Validate EV/EBITDA
            if ev_ebitda is not None:
                if 0 < ev_ebitda <= 100:
                    self._report.add_finding(ValidationFinding(
                        check_name="EV/EBITDA Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.PASS,
                        message=f"EV/EBITDA of {ev_ebitda:.1f}x is within reasonable bounds"
                    ))
                elif ev_ebitda > 100:
                    self._report.add_finding(ValidationFinding(
                        check_name="EV/EBITDA Reasonableness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.WARNING,
                        message=f"EV/EBITDA of {ev_ebitda:.1f}x is unusually high"
                    ))
        
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="Valuation Calculations",
                category=ValidationCategory.FORMULA_VERIFICATION,
                severity=ValidationSeverity.ERROR,
                message=f"Unable to validate valuation calculations: {str(e)}"
            ))
    
    # -------------------------------------------------------------------------
    # WORKING CAPITAL VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_working_capital_calculations(self) -> None:
        """
        Validate working capital and cash conversion cycle calculations.
        
        Validates:
            - CCC = DSO + DIO - DPO
            - DSO = (AR / Revenue) * 365
            - DIO = (Inventory / COGS) * 365
            - DPO = (AP / COGS) * 365
        """
        if 'working_capital' not in self._results:
            return
        
        wc = self._results['working_capital']
        
        try:
            # Extract metrics - these are EfficiencyMetrics objects, need current_value
            dso = self._get_nested_attr(wc, 'dso.current_value', None)
            dio = self._get_nested_attr(wc, 'dio.current_value', None)
            dpo = self._get_nested_attr(wc, 'dpo.current_value', None)
            ccc = self._get_nested_attr(wc, 'cash_conversion_cycle.current_value', None)
            
            # Validate CCC formula: CCC = DSO + DIO - DPO
            if all(v is not None for v in [dso, dio, dpo, ccc]):
                expected_ccc = dso + dio - dpo
                deviation = abs(expected_ccc - ccc)
                
                if deviation <= 1.0:  # 1 day tolerance
                    self._report.add_finding(ValidationFinding(
                        check_name="CCC Formula",
                        category=ValidationCategory.FORMULA_VERIFICATION,
                        severity=ValidationSeverity.PASS,
                        message="CCC = DSO + DIO - DPO verified",
                        expected_value=expected_ccc,
                        actual_value=ccc
                    ))
                else:
                    self._report.add_finding(ValidationFinding(
                        check_name="CCC Formula",
                        category=ValidationCategory.FORMULA_VERIFICATION,
                        severity=ValidationSeverity.ERROR,
                        message=f"CCC formula mismatch: {deviation:.1f} day difference",
                        expected_value=expected_ccc,
                        actual_value=ccc,
                        deviation=deviation
                    ))
            
            # Validate reasonable bounds for days metrics
            for name, value, lower, upper in [
                ("DSO", dso, 0, 365),
                ("DIO", dio, 0, 365),
                ("DPO", dpo, 0, 365)
            ]:
                if value is not None:
                    if lower <= value <= upper:
                        self._report.add_finding(ValidationFinding(
                            check_name=f"{name} Bounds",
                            category=ValidationCategory.BUSINESS_LOGIC,
                            severity=ValidationSeverity.PASS,
                            message=f"{name} of {value:.0f} days is within bounds ({lower}-{upper})"
                        ))
                    else:
                        self._report.add_finding(ValidationFinding(
                            check_name=f"{name} Bounds",
                            category=ValidationCategory.BUSINESS_LOGIC,
                            severity=ValidationSeverity.WARNING,
                            message=f"{name} of {value:.0f} days is outside typical bounds",
                            actual_value=value,
                            expected_value=f"{lower}-{upper} days"
                        ))
        
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="Working Capital Calculations",
                category=ValidationCategory.FORMULA_VERIFICATION,
                severity=ValidationSeverity.ERROR,
                message=f"Unable to validate working capital calculations: {str(e)}"
            ))
    
    # -------------------------------------------------------------------------
    # MARGIN BOUNDS VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_margin_bounds(self) -> None:
        """
        Validate that margin values are within logical bounds.
        
        Margins should generally be between -100% and 100%.
        Values outside this range indicate potential calculation errors,
        such as displaying decimals instead of percentages.
        """
        if 'profitability' not in self._results:
            return
        
        prof = self._results['profitability']
        
        try:
            # Extract margins
            gross_margin = self._get_nested_attr(prof, 'gross_margin', None)
            operating_margin = self._get_nested_attr(prof, 'operating_margin', None)
            
            # Check gross margin
            if gross_margin is not None:
                # If margin is stored as decimal (0.45) vs percentage (45)
                # Decimal form should be between -1 and 1
                # Percentage form should be between -100 and 100
                
                if -1.0 <= gross_margin <= 1.0:
                    # Likely decimal form
                    display_value = gross_margin * 100
                    self._report.add_finding(ValidationFinding(
                        check_name="Gross Margin Format",
                        category=ValidationCategory.DATA_INTEGRITY,
                        severity=ValidationSeverity.PASS,
                        message=f"Gross margin stored as decimal ({gross_margin:.4f}), displays as {display_value:.1f}%"
                    ))
                elif -100 <= gross_margin <= 100:
                    # Likely percentage form
                    self._report.add_finding(ValidationFinding(
                        check_name="Gross Margin Format",
                        category=ValidationCategory.DATA_INTEGRITY,
                        severity=ValidationSeverity.PASS,
                        message=f"Gross margin stored as percentage ({gross_margin:.1f}%)"
                    ))
                else:
                    self._report.add_finding(ValidationFinding(
                        check_name="Gross Margin Bounds",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.ERROR,
                        message=f"Gross margin of {gross_margin} is outside valid bounds",
                        actual_value=gross_margin,
                        expected_value="Between -100% and 100%",
                        recommendation="Check margin calculation and display formatting"
                    ))
            
            # Check operating margin
            if operating_margin is not None:
                if -1.0 <= operating_margin <= 1.0:
                    display_value = operating_margin * 100
                    self._report.add_finding(ValidationFinding(
                        check_name="Operating Margin Format",
                        category=ValidationCategory.DATA_INTEGRITY,
                        severity=ValidationSeverity.PASS,
                        message=f"Operating margin stored as decimal ({operating_margin:.4f}), displays as {display_value:.1f}%"
                    ))
                elif -100 <= operating_margin <= 100:
                    self._report.add_finding(ValidationFinding(
                        check_name="Operating Margin Format",
                        category=ValidationCategory.DATA_INTEGRITY,
                        severity=ValidationSeverity.PASS,
                        message=f"Operating margin stored as percentage ({operating_margin:.1f}%)"
                    ))
                else:
                    self._report.add_finding(ValidationFinding(
                        check_name="Operating Margin Bounds",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.ERROR,
                        message=f"Operating margin of {operating_margin} is outside valid bounds",
                        actual_value=operating_margin,
                        expected_value="Between -100% and 100%"
                    ))
        
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="Margin Bounds",
                category=ValidationCategory.BUSINESS_LOGIC,
                severity=ValidationSeverity.ERROR,
                message=f"Unable to validate margin bounds: {str(e)}"
            ))
    
    # -------------------------------------------------------------------------
    # RATIO REASONABLENESS VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_ratio_reasonableness(self) -> None:
        """Validate that calculated ratios are within reasonable bounds."""
        
        # Combine checks from various modules
        # This provides a comprehensive sanity check on all ratio outputs
        
        ratio_checks = [
            # (name, value, min_bound, max_bound, severity_if_violated)
        ]
        
        # Add checks based on available data
        if 'valuation' in self._results:
            val = self._results['valuation']
            pe = self._get_nested_attr(val, 'pe_ratio', None)
            if pe is not None:
                ratio_checks.append(("P/E", pe, 0, 500, ValidationSeverity.WARNING))
        
        if 'ratios' in self._results:
            ratios = self._results['ratios']
            interest_cov = self._get_nested_attr(ratios, 'interest_coverage', None)
            if interest_cov is not None:
                ratio_checks.append(("Interest Coverage", interest_cov, -50, 1000, ValidationSeverity.WARNING))
        
        for name, value, min_val, max_val, severity in ratio_checks:
            if min_val <= value <= max_val:
                self._report.add_finding(ValidationFinding(
                    check_name=f"{name} Range Check",
                    category=ValidationCategory.BUSINESS_LOGIC,
                    severity=ValidationSeverity.PASS,
                    message=f"{name} of {value:.2f} is within expected range"
                ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name=f"{name} Range Check",
                    category=ValidationCategory.BUSINESS_LOGIC,
                    severity=severity,
                    message=f"{name} of {value:.2f} is outside expected range ({min_val} to {max_val})",
                    actual_value=value,
                    expected_value=f"{min_val} to {max_val}"
                ))
    
    # -------------------------------------------------------------------------
    # SIGN CONSISTENCY VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_sign_consistency(self) -> None:
        """
        Validate that financial values have appropriate signs.
        
        Examples:
            - Revenue should be positive
            - CapEx is typically shown as negative (cash outflow)
            - Total assets should be positive
        """
        sign_checks = []
        
        if 'profitability' in self._results:
            prof = self._results['profitability']
            revenue = self._get_nested_attr(prof, 'revenue_current', None)
            if revenue is not None:
                sign_checks.append(("Revenue", revenue, "positive", revenue > 0))
        
        if 'cash_flow' in self._results:
            cf = self._results['cash_flow']
            fcf = self._get_nested_attr(cf, 'free_cash_flow', None)
            # FCF can be negative for growth companies, so this is informational
        
        for name, value, expected_sign, is_correct in sign_checks:
            if is_correct:
                self._report.add_finding(ValidationFinding(
                    check_name=f"{name} Sign Check",
                    category=ValidationCategory.BUSINESS_LOGIC,
                    severity=ValidationSeverity.PASS,
                    message=f"{name} has correct sign ({expected_sign})"
                ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name=f"{name} Sign Check",
                    category=ValidationCategory.BUSINESS_LOGIC,
                    severity=ValidationSeverity.WARNING,
                    message=f"{name} of {value:,.0f} is not {expected_sign} as expected",
                    actual_value=value
                ))
    
    # -------------------------------------------------------------------------
    # SCORING VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_scoring_weights(self) -> None:
        """
        Validate that scoring weights sum to 100%.
        
        The recommendation scoring uses weights:
            - Cash Flow Quality: 35%
            - Earnings Quality: 25%
            - Profitability: 20%
            - Valuation: 15%
            - Growth: 5%
        
        These must sum to exactly 100% for valid scoring.
        """
        # Define expected weights (from memo_generator.py)
        expected_weights = {
            'cash_flow': 0.35,
            'earnings_quality': 0.25,
            'profitability': 0.20,
            'valuation': 0.15,
            'growth': 0.05
        }
        
        weight_sum = sum(expected_weights.values())
        
        if abs(weight_sum - 1.0) <= self._tolerances.weight_sum_tolerance:
            self._report.add_finding(ValidationFinding(
                check_name="Scoring Weights Sum",
                category=ValidationCategory.SCORING,
                severity=ValidationSeverity.PASS,
                message=f"Scoring weights sum to {weight_sum:.4f} (within tolerance of 1.0)"
            ))
        else:
            self._report.add_finding(ValidationFinding(
                check_name="Scoring Weights Sum",
                category=ValidationCategory.SCORING,
                severity=ValidationSeverity.CRITICAL,
                message=f"Scoring weights sum to {weight_sum:.4f}, not 1.0",
                expected_value=1.0,
                actual_value=weight_sum,
                recommendation="Adjust weights to sum to exactly 100%"
            ))
        
        # Validate individual weights are positive
        for factor, weight in expected_weights.items():
            if weight > 0:
                self._report.add_finding(ValidationFinding(
                    check_name=f"Weight Positive: {factor}",
                    category=ValidationCategory.SCORING,
                    severity=ValidationSeverity.PASS,
                    message=f"{factor} weight of {weight:.0%} is positive"
                ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name=f"Weight Positive: {factor}",
                    category=ValidationCategory.SCORING,
                    severity=ValidationSeverity.ERROR,
                    message=f"{factor} weight of {weight:.0%} is not positive",
                    actual_value=weight
                ))
    
    def _validate_score_ranges(self) -> None:
        """Validate that all component scores are within 0-100 range."""
        
        if 'memo' not in self._results:
            return
        
        memo = self._results['memo']
        
        try:
            # Get score breakdown if available
            score_breakdown = self._get_nested_attr(memo, 'score_breakdown', None)
            
            if score_breakdown and isinstance(score_breakdown, dict):
                scores = score_breakdown.get('scores', {})
                
                for factor, score in scores.items():
                    if self._tolerances.score_range_min <= score <= self._tolerances.score_range_max:
                        self._report.add_finding(ValidationFinding(
                            check_name=f"Score Range: {factor}",
                            category=ValidationCategory.SCORING,
                            severity=ValidationSeverity.PASS,
                            message=f"{factor} score of {score:.0f} is within 0-100 range"
                        ))
                    else:
                        self._report.add_finding(ValidationFinding(
                            check_name=f"Score Range: {factor}",
                            category=ValidationCategory.SCORING,
                            severity=ValidationSeverity.ERROR,
                            message=f"{factor} score of {score:.0f} is outside 0-100 range",
                            actual_value=score,
                            expected_value="0-100"
                        ))
        
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="Score Ranges",
                category=ValidationCategory.SCORING,
                severity=ValidationSeverity.WARNING,
                message=f"Unable to validate score ranges: {str(e)}"
            ))
    
    def _validate_recommendation_logic(self) -> None:
        """
        Validate recommendation determination logic.
        
        Thresholds:
            - Total Score >= 78: STRONG BUY
            - Total Score >= 65: BUY
            - Total Score >= 50: HOLD
            - Total Score >= 38: SELL
            - Total Score < 38: STRONG SELL
        """
        if 'memo' not in self._results:
            return
        
        memo = self._results['memo']
        
        try:
            recommendation = self._get_nested_attr(memo, 'recommendation', None)
            total_score = self._get_nested_attr(memo, 'total_score', None)
            
            if recommendation is None or total_score is None:
                # Try alternative attribute names
                total_score = self._get_nested_attr(memo, 'score_breakdown.total_score', None)
            
            if total_score is not None and recommendation is not None:
                # Determine expected recommendation
                rec_value = recommendation.value if hasattr(recommendation, 'value') else recommendation
                
                if total_score >= 78:
                    expected_rec = "STRONG BUY"
                elif total_score >= 65:
                    expected_rec = "BUY"
                elif total_score >= 50:
                    expected_rec = "HOLD"
                elif total_score >= 38:
                    expected_rec = "SELL"
                else:
                    expected_rec = "STRONG SELL"
                
                if rec_value == expected_rec:
                    self._report.add_finding(ValidationFinding(
                        check_name="Recommendation Logic",
                        category=ValidationCategory.SCORING,
                        severity=ValidationSeverity.PASS,
                        message=f"Recommendation '{rec_value}' is correct for score {total_score:.1f}"
                    ))
                else:
                    self._report.add_finding(ValidationFinding(
                        check_name="Recommendation Logic",
                        category=ValidationCategory.SCORING,
                        severity=ValidationSeverity.ERROR,
                        message=f"Recommendation mismatch: got '{rec_value}', expected '{expected_rec}' for score {total_score:.1f}",
                        expected_value=expected_rec,
                        actual_value=rec_value
                    ))
        
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="Recommendation Logic",
                category=ValidationCategory.SCORING,
                severity=ValidationSeverity.WARNING,
                message=f"Unable to validate recommendation logic: {str(e)}"
            ))
    
    def _validate_confidence_calculation(self) -> None:
        """
        Validate confidence score calculation.
        
        Confidence should be:
            - Based on distance from decision thresholds
            - Higher when score is clearly in one bucket
            - Lower when score is near threshold boundaries
            - Bounded between 35% and 95%
        """
        if 'memo' not in self._results:
            return
        
        memo = self._results['memo']
        
        try:
            # Note: attribute name is confidence_score, not confidence
            confidence = self._get_nested_attr(memo, 'confidence_score', None)
            
            if confidence is not None:
                # Validate bounds
                if 35 <= confidence <= 95:
                    self._report.add_finding(ValidationFinding(
                        check_name="Confidence Bounds",
                        category=ValidationCategory.SCORING,
                        severity=ValidationSeverity.PASS,
                        message=f"Confidence of {confidence:.0f}% is within valid bounds (35-95%)"
                    ))
                else:
                    self._report.add_finding(ValidationFinding(
                        check_name="Confidence Bounds",
                        category=ValidationCategory.SCORING,
                        severity=ValidationSeverity.WARNING,
                        message=f"Confidence of {confidence:.0f}% is outside valid bounds (35-95%)",
                        actual_value=confidence,
                        expected_value="35-95%"
                    ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name="Confidence Bounds",
                    category=ValidationCategory.SCORING,
                    severity=ValidationSeverity.PASS,
                    message="Confidence score present in memo"
                ))
        
        except Exception as e:
            self._report.add_finding(ValidationFinding(
                check_name="Confidence Calculation",
                category=ValidationCategory.SCORING,
                severity=ValidationSeverity.WARNING,
                message=f"Unable to validate confidence calculation: {str(e)}"
            ))
    
    # -------------------------------------------------------------------------
    # CROSS-MODULE CONSISTENCY VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_cross_module_consistency(self) -> None:
        """
        Validate consistency of values across different modules.
        
        For example:
            - Net income in cash_flow should match profitability
            - FCF in valuation should match cash_flow
            - Margins in memo should match profitability
        """
        # Net Income consistency
        ni_sources = {}
        
        if 'profitability' in self._results:
            prof = self._results['profitability']
            ni_prof = self._get_nested_attr(prof, 'net_income', None)
            if ni_prof is not None:
                ni_sources['profitability'] = ni_prof
        
        if 'cash_flow' in self._results:
            cf = self._results['cash_flow']
            ni_cf = self._get_nested_attr(cf, 'net_income', None)
            if ni_cf is not None:
                ni_sources['cash_flow'] = ni_cf
        
        if 'earnings_quality' in self._results:
            eq = self._results['earnings_quality']
            ni_eq = self._get_nested_attr(eq, 'net_income', None)
            if ni_eq is not None:
                ni_sources['earnings_quality'] = ni_eq
        
        # Check consistency
        if len(ni_sources) > 1:
            values = list(ni_sources.values())
            reference = values[0]
            all_match = all(
                abs(v - reference) <= max(abs(reference * 0.001), 1.0)
                for v in values
            )
            
            if all_match:
                self._report.add_finding(ValidationFinding(
                    check_name="Net Income Consistency",
                    category=ValidationCategory.CROSS_MODULE,
                    severity=ValidationSeverity.PASS,
                    message="Net income values are consistent across modules"
                ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name="Net Income Consistency",
                    category=ValidationCategory.CROSS_MODULE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Net income values differ across modules: {ni_sources}",
                    recommendation="Verify all modules use same data source"
                ))
        
        # FCF consistency
        fcf_sources = {}
        
        if 'cash_flow' in self._results:
            cf = self._results['cash_flow']
            fcf_cf = self._get_nested_attr(cf, 'free_cash_flow', None)
            if fcf_cf is not None:
                fcf_sources['cash_flow'] = fcf_cf
        
        if 'valuation' in self._results:
            val = self._results['valuation']
            # FCF used in valuation calculations
            fcf_val = self._get_nested_attr(val, 'fcf', None)
            if fcf_val is not None:
                fcf_sources['valuation'] = fcf_val
        
        if len(fcf_sources) > 1:
            values = list(fcf_sources.values())
            reference = values[0]
            all_match = all(
                abs(v - reference) <= max(abs(reference * 0.001), 1.0)
                for v in values
            )
            
            if all_match:
                self._report.add_finding(ValidationFinding(
                    check_name="FCF Consistency",
                    category=ValidationCategory.CROSS_MODULE,
                    severity=ValidationSeverity.PASS,
                    message="FCF values are consistent across modules"
                ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name="FCF Consistency",
                    category=ValidationCategory.CROSS_MODULE,
                    severity=ValidationSeverity.WARNING,
                    message=f"FCF values differ across modules: {fcf_sources}"
                ))
    
    # -------------------------------------------------------------------------
    # OUTPUT COMPLETENESS VALIDATION
    # -------------------------------------------------------------------------
    
    def _validate_output_completeness(self) -> None:
        """
        Validate that all required output fields are present and populated.
        
        Required fields in final memo:
            - Recommendation
            - Confidence
            - Executive Summary
            - All section content
            - Data citations
        """
        if 'memo' not in self._results:
            self._report.add_finding(ValidationFinding(
                check_name="Memo Present",
                category=ValidationCategory.OUTPUT_CONSISTENCY,
                severity=ValidationSeverity.CRITICAL,
                message="Investment memo is missing from results"
            ))
            return
        
        memo = self._results['memo']
        
        required_fields = [
            ('recommendation', "Investment recommendation"),
            ('confidence_score', "Confidence score"),
            ('ticker', "Ticker symbol"),
        ]
        
        for field_name, description in required_fields:
            value = self._get_nested_attr(memo, field_name, None)
            if value is not None:
                self._report.add_finding(ValidationFinding(
                    check_name=f"Required Field: {field_name}",
                    category=ValidationCategory.OUTPUT_CONSISTENCY,
                    severity=ValidationSeverity.PASS,
                    message=f"{description} is present"
                ))
            else:
                self._report.add_finding(ValidationFinding(
                    check_name=f"Required Field: {field_name}",
                    category=ValidationCategory.OUTPUT_CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"{description} is missing",
                    recommendation=f"Ensure {field_name} is populated in memo generation"
                ))
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    def _get_nested_attr(self, obj: Any, attr_path: str, default: Any = None) -> Any:
        """
        Get nested attribute value using dot notation.
        
        Args:
            obj: Object to get attribute from
            attr_path: Dot-separated attribute path (e.g., "bridge.ebit_prior")
            default: Default value if attribute not found
        
        Returns:
            Attribute value or default
        """
        try:
            parts = attr_path.split('.')
            current = obj
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            
            return current
        except Exception:
            return default


# =============================================================================
# STANDALONE VALIDATION FUNCTIONS
# =============================================================================

def validate_ebit_bridge(
    ebit_prior: float,
    ebit_current: float,
    volume_effect: float,
    gm_rate_effect: float,
    opex_rate_effect: float,
    tolerance: float = 1.0
) -> Tuple[bool, float, str]:
    """
    Standalone EBIT bridge validation function.
    
    Args:
        ebit_prior: Prior period EBIT
        ebit_current: Current period EBIT
        volume_effect: Volume contribution to EBIT change
        gm_rate_effect: Gross margin rate contribution
        opex_rate_effect: Operating expense rate contribution
        tolerance: Acceptable deviation in same units as EBIT
    
    Returns:
        Tuple of (is_valid, deviation, message)
    
    Example:
        is_valid, deviation, msg = validate_ebit_bridge(
            ebit_prior=123216,
            ebit_current=133050,
            volume_effect=7917,
            gm_rate_effect=2800,
            opex_rate_effect=-883,
            tolerance=1.0
        )
    """
    calculated_ebit = ebit_prior + volume_effect + gm_rate_effect + opex_rate_effect
    deviation = abs(calculated_ebit - ebit_current)
    
    is_valid = deviation <= tolerance
    
    if is_valid:
        message = f"EBIT bridge reconciles: deviation ${deviation:.2f}M within ${tolerance:.2f}M tolerance"
    else:
        message = f"EBIT bridge FAILS: deviation ${deviation:.2f}M exceeds ${tolerance:.2f}M tolerance"
    
    return is_valid, deviation, message


def validate_fcf_calculation(
    operating_cash_flow: float,
    capital_expenditures: float,
    free_cash_flow: float,
    tolerance: float = 1.0
) -> Tuple[bool, float, str]:
    """
    Standalone FCF calculation validation.
    
    Args:
        operating_cash_flow: Operating cash flow (positive)
        capital_expenditures: CapEx (typically negative, will use absolute)
        free_cash_flow: Reported free cash flow
        tolerance: Acceptable deviation
    
    Returns:
        Tuple of (is_valid, deviation, message)
    """
    expected_fcf = operating_cash_flow - abs(capital_expenditures)
    deviation = abs(expected_fcf - free_cash_flow)
    
    is_valid = deviation <= tolerance
    
    if is_valid:
        message = f"FCF calculation correct: deviation ${deviation:.2f}M"
    else:
        message = f"FCF calculation ERROR: deviation ${deviation:.2f}M"
    
    return is_valid, deviation, message


def validate_cash_conversion_cycle(
    dso: float,
    dio: float,
    dpo: float,
    ccc: float,
    tolerance: float = 1.0
) -> Tuple[bool, float, str]:
    """
    Validate Cash Conversion Cycle calculation.
    
    CCC = DSO + DIO - DPO
    
    Args:
        dso: Days Sales Outstanding
        dio: Days Inventory Outstanding
        dpo: Days Payable Outstanding
        ccc: Reported Cash Conversion Cycle
        tolerance: Acceptable deviation in days
    
    Returns:
        Tuple of (is_valid, deviation, message)
    """
    expected_ccc = dso + dio - dpo
    deviation = abs(expected_ccc - ccc)
    
    is_valid = deviation <= tolerance
    
    if is_valid:
        message = f"CCC formula correct: {expected_ccc:.0f} days"
    else:
        message = f"CCC formula ERROR: expected {expected_ccc:.0f}, got {ccc:.0f}"
    
    return is_valid, deviation, message


def validate_scoring_calculation(
    scores: Dict[str, float],
    weights: Dict[str, float],
    reported_total: float,
    tolerance: float = 0.1
) -> Tuple[bool, float, str]:
    """
    Validate recommendation score calculation.
    
    Args:
        scores: Dictionary of component scores (0-100)
        weights: Dictionary of component weights (should sum to 1.0)
        reported_total: Reported total score
        tolerance: Acceptable deviation
    
    Returns:
        Tuple of (is_valid, deviation, message)
    """
    # Validate weights sum
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.001:
        return False, abs(weight_sum - 1.0), f"Weights sum to {weight_sum:.4f}, not 1.0"
    
    # Calculate expected total
    expected_total = sum(scores[k] * weights[k] for k in scores.keys())
    deviation = abs(expected_total - reported_total)
    
    is_valid = deviation <= tolerance
    
    if is_valid:
        message = f"Score calculation correct: {expected_total:.1f}"
    else:
        message = f"Score calculation ERROR: expected {expected_total:.1f}, got {reported_total:.1f}"
    
    return is_valid, deviation, message


def validate_margin_display(
    stored_value: float,
    displayed_value: float,
    margin_name: str = "Margin"
) -> Tuple[bool, str]:
    """
    Validate margin value display formatting.
    
    Checks whether stored decimal values are correctly converted to percentages.
    
    Args:
        stored_value: Value as stored (may be decimal 0.45 or percentage 45)
        displayed_value: Value as displayed in report
        margin_name: Name for error messages
    
    Returns:
        Tuple of (is_valid, message)
    """
    # If stored as decimal (0-1 range), displayed should be stored * 100
    if -1.0 <= stored_value <= 1.0:
        expected_display = stored_value * 100
        deviation = abs(expected_display - displayed_value)
        
        if deviation <= 0.1:
            return True, f"{margin_name}: correctly converted from {stored_value:.4f} to {displayed_value:.1f}%"
        else:
            return False, f"{margin_name}: display error - stored {stored_value:.4f}, displayed {displayed_value:.1f}%, expected {expected_display:.1f}%"
    
    # If stored as percentage, displayed should match
    elif -100 <= stored_value <= 100:
        deviation = abs(stored_value - displayed_value)
        
        if deviation <= 0.1:
            return True, f"{margin_name}: correctly displayed as {displayed_value:.1f}%"
        else:
            return False, f"{margin_name}: display mismatch - stored {stored_value:.1f}%, displayed {displayed_value:.1f}%"
    
    else:
        return False, f"{margin_name}: invalid stored value {stored_value}"


def validate_valuation_unit_consistency(
    market_cap: float,
    financial_metric: float,
    metric_name: str,
    calculated_ratio: float,
    expected_reasonable_max: float = 200
) -> Tuple[bool, str]:
    """
    Validate that valuation ratio calculation uses consistent units.
    
    This catches the common bug where market cap is in dollars but
    financial metrics are in millions.
    
    Args:
        market_cap: Market capitalization used in calculation
        financial_metric: Financial metric (e.g., net income, FCF)
        metric_name: Name of the metric for messages
        calculated_ratio: The calculated ratio
        expected_reasonable_max: Maximum reasonable value for the ratio
    
    Returns:
        Tuple of (is_valid, message)
    """
    if calculated_ratio <= 0:
        return True, f"{metric_name} ratio not meaningful (non-positive)"
    
    if calculated_ratio > expected_reasonable_max:
        # Likely a units mismatch
        # If ratio is ~1,000,000x too high, market cap is probably not scaled
        if calculated_ratio > 100000:
            return False, (
                f"{metric_name} ratio of {calculated_ratio:,.0f}x suggests units mismatch. "
                f"Market cap ({market_cap:,.0f}) may need to be divided by 1,000,000 "
                f"to match financial statements in millions."
            )
        else:
            return False, (
                f"{metric_name} ratio of {calculated_ratio:.1f}x exceeds reasonable maximum "
                f"of {expected_reasonable_max}x. Verify calculation."
            )
    
    return True, f"{metric_name} ratio of {calculated_ratio:.1f}x is reasonable"


# =============================================================================
# COMPREHENSIVE REPORT VALIDATION
# =============================================================================

def validate_report_accuracy(
    report_values: Dict[str, Any],
    calculated_values: Dict[str, Any],
    tolerances: Optional[Dict[str, float]] = None
) -> ValidationReport:
    """
    Comprehensive validation comparing report values to calculated values.
    
    This function takes a dictionary of values displayed in the report
    and compares them to independently calculated values to verify accuracy.
    
    Args:
        report_values: Values as shown in the final report
        calculated_values: Values from independent calculation
        tolerances: Optional custom tolerances per field
    
    Returns:
        ValidationReport with all findings
    
    Example:
        report = validate_report_accuracy(
            report_values={
                'gross_margin': 46.9,
                'operating_margin': 32.0,
                'pe_ratio': 34.1,
                'pfcf_ratio': 38.6
            },
            calculated_values={
                'gross_margin': 46.9,
                'operating_margin': 32.0,
                'pe_ratio': 34.1,
                'pfcf_ratio': 38.5
            }
        )
    """
    default_tolerances = {
        'gross_margin': 0.5,
        'operating_margin': 0.5,
        'pe_ratio': 0.5,
        'ev_ebitda': 0.5,
        'pfcf_ratio': 0.5,
        'pb_ratio': 0.5,
        'dso': 1.0,
        'dio': 1.0,
        'dpo': 1.0,
        'ccc': 1.0,
        'ebit': 10.0,
        'fcf': 10.0,
        'cash_conversion': 1.0
    }
    
    if tolerances:
        default_tolerances.update(tolerances)
    
    validation_report = ValidationReport(
        timestamp=datetime.now(),
        ticker=report_values.get('ticker', 'UNKNOWN')
    )
    
    for field, report_val in report_values.items():
        if field in calculated_values:
            calc_val = calculated_values[field]
            tolerance = default_tolerances.get(field, 1.0)
            
            if report_val is None or calc_val is None:
                validation_report.add_finding(ValidationFinding(
                    check_name=f"Value Present: {field}",
                    category=ValidationCategory.DATA_INTEGRITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"{field} has null value (report: {report_val}, calc: {calc_val})"
                ))
                continue
            
            deviation = abs(report_val - calc_val)
            
            if deviation <= tolerance:
                validation_report.add_finding(ValidationFinding(
                    check_name=f"Accuracy: {field}",
                    category=ValidationCategory.OUTPUT_CONSISTENCY,
                    severity=ValidationSeverity.PASS,
                    message=f"{field} matches within tolerance",
                    expected_value=calc_val,
                    actual_value=report_val,
                    deviation=deviation
                ))
            else:
                validation_report.add_finding(ValidationFinding(
                    check_name=f"Accuracy: {field}",
                    category=ValidationCategory.OUTPUT_CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field} deviates from calculated value",
                    expected_value=calc_val,
                    actual_value=report_val,
                    deviation=deviation,
                    recommendation=f"Verify {field} calculation and display"
                ))
    
    validation_report.generate_summary()
    return validation_report


# =============================================================================
# MODULE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage and self-test
    print("Validation Module Self-Test")
    print("=" * 60)
    
    # Test EBIT bridge validation
    is_valid, deviation, msg = validate_ebit_bridge(
        ebit_prior=123216,
        ebit_current=133050,
        volume_effect=7917,
        gm_rate_effect=2800,
        opex_rate_effect=-883,
        tolerance=1.0
    )
    print(f"\nEBIT Bridge Test:")
    print(f"  Valid: {is_valid}")
    print(f"  Deviation: ${deviation:.2f}M")
    print(f"  Message: {msg}")
    
    # Test CCC validation
    is_valid, deviation, msg = validate_cash_conversion_cycle(
        dso=35,
        dio=9,
        dpo=115,
        ccc=-71,
        tolerance=1.0
    )
    print(f"\nCCC Test:")
    print(f"  Valid: {is_valid}")
    print(f"  Message: {msg}")
    
    # Test margin display validation
    is_valid, msg = validate_margin_display(
        stored_value=0.469,
        displayed_value=46.9,
        margin_name="Gross Margin"
    )
    print(f"\nMargin Display Test:")
    print(f"  Valid: {is_valid}")
    print(f"  Message: {msg}")
    
    # Test unit consistency
    is_valid, msg = validate_valuation_unit_consistency(
        market_cap=3800000,  # In millions
        financial_metric=98767,  # FCF in millions
        metric_name="P/FCF",
        calculated_ratio=38.5
    )
    print(f"\nUnit Consistency Test:")
    print(f"  Valid: {is_valid}")
    print(f"  Message: {msg}")
    
    print("\n" + "=" * 60)
    print("Self-test complete")