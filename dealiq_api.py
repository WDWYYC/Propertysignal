"""
DealIQ – Real Estate Algorithm Engine
FastAPI Backend

Algorithms:
  1. Rental Income Estimator
  2. Cash Flow Engine
  3. Investment Metrics (Cap Rate, IRR, Cash-on-Cash)

Install:
  pip install fastapi uvicorn numpy

Run:
  uvicorn dealiq_api:app --reload --port 8000

Docs (auto-generated):
  http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Union
import math
import numpy as np

app = FastAPI(title="DealIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Response Models ────────────────────────────────────────────────────────

class RentalEstimateResponse(BaseModel):
    estimated_monthly_rent: float
    rent_range_low: float
    rent_range_high: float
    confidence_pct: float
    rent_per_sqft: float
    gross_annual_rent: float
    methodology: str

class CashFlowResponse(BaseModel):
    monthly_mortgage: float
    monthly_tax: float
    monthly_insurance: float
    monthly_hoa: float
    monthly_vacancy_loss: float
    monthly_mgmt_fee: float
    monthly_maintenance: float
    total_monthly_expenses: float
    gross_monthly_income: float
    net_monthly_cash_flow: float
    net_annual_cash_flow: float
    expense_ratio_pct: float

class InvestmentMetricsResponse(BaseModel):
    cap_rate_pct: float
    cash_on_cash_return_pct: float
    irr_pct: float
    gross_rent_multiplier: float
    break_even_years: float
    total_return_pct: float
    equity_at_exit: float
    projected_exit_price: float
    net_monthly_cash_flow: float
    net_annual_cash_flow: float
    deal_rating: str
    deal_rating_reason: str

# ─── Request Model ───────────────────────────────────────────────────────────

class FullAnalysisRequest(BaseModel):
    """Convenience endpoint – runs all three algorithms in one call."""
    purchase_price: Union[float, str]
    sqft: Union[float, str]
    bedrooms: Union[float, str]
    bathrooms: Union[float, str]
    property_type: str = "single_family"
    market_tier: str = "mid_cost"
    year_built: Optional[Union[int, str]] = None
    has_garage: bool = False
    has_pool: bool = False
    down_payment_pct: Union[float, str] = 0.20
    interest_rate: Union[float, str] = 0.07
    loan_term_years: Union[float, str] = 30
    annual_appreciation_rate: Union[float, str] = 0.04
    hold_years: Union[float, str] = 5
    monthly_hoa: Union[float, str] = 0.0
    vacancy_rate: Union[float, str] = 0.08
    mgmt_fee_rate: Union[float, str] = 0.08
    closing_costs_pct: Union[float, str] = 0.03

    def __init__(self, **data):
        super().__init__(**data)
        for field in ['purchase_price', 'sqft', 'bedrooms', 'bathrooms',
                      'down_payment_pct', 'interest_rate', 'loan_term_years',
                      'annual_appreciation_rate', 'hold_years', 'monthly_hoa',
                      'vacancy_rate', 'mgmt_fee_rate', 'closing_costs_pct']:
            val = getattr(self, field)
            if isinstance(val, str):
                setattr(self, field, float(val))

    class Config:
        json_schema_extra = {
            "example": {
                "purchase_price": 389000,
                "sqft": 1840,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "property_type": "single_family",
                "market_tier": "mid_cost",
                "year_built": 1998,
                "has_garage": True,
                "has_pool": False,
                "down_payment_pct": 0.20,
                "interest_rate": 0.07,
                "loan_term_years": 30,
                "annual_appreciation_rate": 0.04,
                "hold_years": 5,
                "monthly_hoa": 0,
                "vacancy_rate": 0.08,
                "mgmt_fee_rate": 0.08
            }
        }

# ─── Algorithm 1: Rental Income Estimator ───────────────────────────────────

def estimate_rental_income(
    purchase_price: float,
    sqft: float,
    bedrooms: float,
    bathrooms: float,
    market_tier: str = "mid_cost",
    property_type: str = "single_family",
    has_garage: bool = False,
    has_pool: bool = False,
) -> RentalEstimateResponse:

    base_rent_per_sqft = {
        "budget": 0.65, "mid_cost": 0.92, "mid": 0.92,
        "premium": 1.35, "luxury": 1.85
    }.get(market_tier.lower(), 0.92)

    base_rent = sqft * base_rent_per_sqft

    bedroom_multipliers = {1: 0.85, 2: 0.95, 3: 1.0, 4: 1.08, 5: 1.14}
    base_rent *= bedroom_multipliers.get(int(bedrooms), 1.0)

    if bathrooms >= 2: base_rent *= 1.03
    if has_garage: base_rent *= 1.02
    if has_pool: base_rent *= 1.04

    gross_rent_ratio = 0.0042 if market_tier in ["budget", "mid_cost", "mid"] else 0.0035
    price_based_rent = purchase_price * gross_rent_ratio
    estimated_rent = (base_rent * 0.6) + (price_based_rent * 0.4)

    confidence = 78 if market_tier in ["budget", "mid_cost", "mid"] else 72

    return RentalEstimateResponse(
        estimated_monthly_rent=round(estimated_rent, 2),
        rent_range_low=round(estimated_rent * 0.83, 2),
        rent_range_high=round(estimated_rent * 1.17, 2),
        confidence_pct=confidence,
        rent_per_sqft=round(estimated_rent / sqft, 3),
        gross_annual_rent=round(estimated_rent * 12, 2),
        methodology="Hybrid: $/sqft market rates + GRM price-based estimate"
    )

# ─── Algorithm 2: Cash Flow Engine ──────────────────────────────────────────

def calculate_cash_flow(
    purchase_price: float,
    monthly_rent: float,
    down_payment_pct: float = 0.20,
    interest_rate: float = 0.07,
    loan_term_years: float = 30,
    monthly_hoa: float = 0.0,
    vacancy_rate: float = 0.08,
    mgmt_fee_rate: float = 0.08,
    closing_costs_pct: float = 0.03,
) -> CashFlowResponse:

    loan_amount = purchase_price * (1 - down_payment_pct)
    monthly_rate = interest_rate / 12
    n_payments = int(loan_term_years * 12)

    if monthly_rate > 0:
        mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    else:
        mortgage = loan_amount / n_payments

    tax = purchase_price * 0.0115 / 12
    insurance = purchase_price * 0.006 / 12
    vacancy_loss = monthly_rent * vacancy_rate
    mgmt_fee = monthly_rent * mgmt_fee_rate
    maintenance = purchase_price * 0.01 / 12

    total_expenses = mortgage + tax + insurance + monthly_hoa + vacancy_loss + mgmt_fee + maintenance
    net_cash_flow = monthly_rent - total_expenses

    return CashFlowResponse(
        monthly_mortgage=round(mortgage, 2),
        monthly_tax=round(tax, 2),
        monthly_insurance=round(insurance, 2),
        monthly_hoa=round(monthly_hoa, 2),
        monthly_vacancy_loss=round(vacancy_loss, 2),
        monthly_mgmt_fee=round(mgmt_fee, 2),
        monthly_maintenance=round(maintenance, 2),
        total_monthly_expenses=round(total_expenses, 2),
        gross_monthly_income=round(monthly_rent, 2),
        net_monthly_cash_flow=round(net_cash_flow, 2),
        net_annual_cash_flow=round(net_cash_flow * 12, 2),
        expense_ratio_pct=round((total_expenses / monthly_rent) * 100, 1) if monthly_rent > 0 else 0
    )

# ─── Algorithm 3: Investment Metrics ────────────────────────────────────────

def calculate_investment_metrics(
    purchase_price: float,
    cash_flow: CashFlowResponse,
    rental: RentalEstimateResponse,
    down_payment_pct: float = 0.20,
    annual_appreciation_rate: float = 0.04,
    hold_years: float = 5,
    closing_costs_pct: float = 0.03,
    interest_rate: float = 0.07,
    loan_term_years: float = 30,
) -> InvestmentMetricsResponse:

    noi = (rental.estimated_monthly_rent * 12) - (
        (cash_flow.monthly_tax + cash_flow.monthly_insurance +
         cash_flow.monthly_hoa + cash_flow.monthly_vacancy_loss +
         cash_flow.monthly_mgmt_fee + cash_flow.monthly_maintenance) * 12
    )
    cap_rate = (noi / purchase_price) * 100

    down_payment = purchase_price * down_payment_pct
    closing_costs = purchase_price * closing_costs_pct
    total_invested = down_payment + closing_costs
    coc_return = (cash_flow.net_annual_cash_flow / total_invested) * 100

    exit_price = purchase_price * (1 + annual_appreciation_rate) ** hold_years
    loan_amount = purchase_price * (1 - down_payment_pct)
    monthly_rate = interest_rate / 12
    n_payments = int(loan_term_years * 12)

    if monthly_rate > 0:
        balance = loan_amount * (1 + monthly_rate)**(hold_years*12) - \
                  cash_flow.monthly_mortgage * ((1 + monthly_rate)**(hold_years*12) - 1) / monthly_rate
    else:
        balance = loan_amount - (cash_flow.monthly_mortgage * hold_years * 12)

    equity_at_exit = exit_price - max(balance, 0)
    total_cash_flow = cash_flow.net_annual_cash_flow * hold_years
    total_return = ((equity_at_exit - total_invested + total_cash_flow) / total_invested) * 100

    cash_flows = [-total_invested] + [cash_flow.net_annual_cash_flow] * int(hold_years)
    cash_flows[-1] += equity_at_exit

    def npv(rate, cfs):
        return sum(cf / (1 + rate)**i for i, cf in enumerate(cfs))

    irr = 0.1
    for _ in range(1000):
        f = npv(irr, cash_flows)
        df = sum(-i * cf / (1 + irr)**(i+1) for i, cf in enumerate(cash_flows))
        if abs(df) < 1e-10: break
        irr = irr - f / df
        irr = max(-0.99, min(irr, 10.0))

    grm = purchase_price / rental.gross_annual_rent if rental.gross_annual_rent > 0 else 0

    if cash_flow.net_annual_cash_flow > 0:
        break_even = total_invested / cash_flow.net_annual_cash_flow
    else:
        break_even = 99

    if cap_rate >= 6 and coc_return >= 8:
        rating, reason = "Excellent", "Strong cap rate and cash-on-cash return"
    elif cap_rate >= 5 and coc_return >= 5:
        rating, reason = "Good", "Solid fundamentals with positive cash flow"
    elif cap_rate >= 4:
        rating, reason = "Fair", "Moderate returns, appreciation dependent"
    else:
        rating, reason = "Poor", "Below market returns, consider alternatives"

    return InvestmentMetricsResponse(
        cap_rate_pct=round(cap_rate, 2),
        cash_on_cash_return_pct=round(coc_return, 2),
        irr_pct=round(irr * 100, 2),
        gross_rent_multiplier=round(grm, 2),
        break_even_years=round(break_even, 1),
        total_return_pct=round(total_return, 2),
        equity_at_exit=round(equity_at_exit, 2),
        projected_exit_price=round(exit_price, 2),
        net_monthly_cash_flow=round(cash_flow.net_monthly_cash_flow, 2),
        net_annual_cash_flow=round(cash_flow.net_annual_cash_flow, 2),
        deal_rating=rating,
        deal_rating_reason=reason
    )

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "online", "service": "DealIQ API", "version": "1.0.0"}

@app.post("/api/full-analysis")
def full_analysis(req: FullAnalysisRequest):
    rental = estimate_rental_income(
        purchase_price=float(req.purchase_price),
        sqft=float(req.sqft),
        bedrooms=float(req.bedrooms),
        bathrooms=float(req.bathrooms),
        market_tier=req.market_tier,
        property_type=req.property_type,
        has_garage=req.has_garage,
        has_pool=req.has_pool,
    )
    cash_flow = calculate_cash_flow(
        purchase_price=float(req.purchase_price),
        monthly_rent=rental.estimated_monthly_rent,
        down_payment_pct=float(req.down_payment_pct),
        interest_rate=float(req.interest_rate),
        loan_term_years=float(req.loan_term_years),
        monthly_hoa=float(req.monthly_hoa),
        vacancy_rate=float(req.vacancy_rate),
        mgmt_fee_rate=float(req.mgmt_fee_rate),
        closing_costs_pct=float(req.closing_costs_pct),
    )
    metrics = calculate_investment_metrics(
        purchase_price=float(req.purchase_price),
        cash_flow=cash_flow,
        rental=rental,
        down_payment_pct=float(req.down_payment_pct),
        annual_appreciation_rate=float(req.annual_appreciation_rate),
        hold_years=float(req.hold_years),
        closing_costs_pct=float(req.closing_costs_pct),
        interest_rate=float(req.interest_rate),
        loan_term_years=float(req.loan_term_years),
    )
    return {
        "property_summary": {
            "purchase_price": req.purchase_price,
            "sqft": req.sqft,
            "bedrooms": req.bedrooms,
            "bathrooms": req.bathrooms,
            "property_type": req.property_type,
            "market_tier": req.market_tier,
        },
        "rental_estimate": rental,
        "cash_flow": cash_flow,
        "investment_metrics": metrics,
    }
