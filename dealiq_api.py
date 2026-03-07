"""
DealIQ — Real Estate Algorithm Engine
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
from typing import Optional
import math
import numpy as np

app = FastAPI(
    title="DealIQ Algorithm Engine",
    description="Real estate investment analysis APIs",
    version="1.0.0"
)

# Allow requests from the React frontend (localhost:3000 in dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────

# Market-level cap rate benchmarks by property type
MARKET_CAP_RATES = {
    "single_family":   {"low": 0.04, "mid": 0.065, "high": 0.10},
    "multi_family":    {"low": 0.045, "mid": 0.07, "high": 0.11},
    "condo":           {"low": 0.035, "mid": 0.055, "high": 0.09},
    "commercial":      {"low": 0.055, "mid": 0.08, "high": 0.12},
}

# Regional rent-per-sqft benchmarks ($/sqft/month)
REGIONAL_RENT_SQFT = {
    "high_cost":   {"base": 2.80, "std": 0.50},   # NYC, SF, LA, Boston
    "mid_cost":    {"base": 1.60, "std": 0.35},   # Denver, Austin, Nashville, Tampa
    "low_cost":    {"base": 0.95, "std": 0.25},   # Detroit, Memphis, Cleveland
}

DEFAULT_EXPENSE_RATIOS = {
    "property_tax_rate":   0.012,    # 1.2% of property value annually
    "insurance_rate":      0.005,    # 0.5% of property value annually
    "maintenance_rate":    0.01,     # 1% of property value annually
    "vacancy_rate":        0.08,     # 8% vacancy
    "mgmt_fee_rate":       0.08,     # 8% of gross rents
    "capex_reserve_rate":  0.05,     # 5% of gross rents for capex
}


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class RentalEstimateRequest(BaseModel):
    sqft: float = Field(..., gt=0, description="Property square footage")
    bedrooms: int = Field(..., ge=0, le=20)
    bathrooms: float = Field(..., ge=0, le=20)
    property_type: str = Field(default="single_family",
        description="single_family | multi_family | condo | commercial")
    market_tier: str = Field(default="mid_cost",
        description="high_cost | mid_cost | low_cost")
    year_built: Optional[int] = Field(default=None)
    has_garage: bool = Field(default=False)
    has_pool: bool = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "sqft": 1840,
                "bedrooms": 3,
                "bathrooms": 2,
                "property_type": "single_family",
                "market_tier": "mid_cost",
                "year_built": 1998,
                "has_garage": True,
                "has_pool": False
            }
        }


class RentalEstimateResponse(BaseModel):
    estimated_monthly_rent: float
    rent_range_low: float
    rent_range_high: float
    confidence_pct: int
    rent_per_sqft: float
    gross_annual_rent: float
    methodology: str


class CashFlowRequest(BaseModel):
    purchase_price: float = Field(..., gt=0)
    monthly_rent: float = Field(..., gt=0)
    down_payment_pct: float = Field(default=0.20, ge=0.0, le=1.0,
        description="Down payment as decimal (e.g. 0.20 = 20%)")
    interest_rate: float = Field(default=0.07, ge=0.0, le=0.30,
        description="Annual mortgage interest rate as decimal")
    loan_term_years: int = Field(default=30, ge=5, le=40)
    # Overridable expenses (use None to auto-calculate from defaults)
    annual_property_tax: Optional[float] = None
    annual_insurance: Optional[float] = None
    monthly_hoa: float = Field(default=0.0, ge=0)
    vacancy_rate: float = Field(default=0.08, ge=0.0, le=1.0)
    mgmt_fee_rate: float = Field(default=0.08, ge=0.0, le=0.50)
    maintenance_annual: Optional[float] = None
    capex_reserve_annual: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "purchase_price": 389000,
                "monthly_rent": 2950,
                "down_payment_pct": 0.20,
                "interest_rate": 0.07,
                "loan_term_years": 30,
                "monthly_hoa": 0,
                "vacancy_rate": 0.08,
                "mgmt_fee_rate": 0.08
            }
        }


class CashFlowResponse(BaseModel):
    # Income
    gross_monthly_rent: float
    effective_gross_income_monthly: float   # after vacancy
    # Expenses breakdown
    mortgage_payment: float
    property_tax_monthly: float
    insurance_monthly: float
    hoa_monthly: float
    maintenance_monthly: float
    capex_reserve_monthly: float
    mgmt_fee_monthly: float
    total_expenses_monthly: float
    # Bottom line
    net_monthly_cash_flow: float
    net_annual_cash_flow: float
    # Loan info
    loan_amount: float
    down_payment: float
    total_cash_invested: float   # down payment + closing costs est.


class InvestmentMetricsRequest(BaseModel):
    purchase_price: float = Field(..., gt=0)
    monthly_rent: float = Field(..., gt=0)
    down_payment_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    interest_rate: float = Field(default=0.07)
    loan_term_years: int = Field(default=30)
    annual_appreciation_rate: float = Field(default=0.04, ge=0.0, le=0.30,
        description="Expected annual property appreciation as decimal")
    hold_years: int = Field(default=5, ge=1, le=30,
        description="Expected hold period for IRR calculation")
    # Optional overrides
    annual_property_tax: Optional[float] = None
    annual_insurance: Optional[float] = None
    monthly_hoa: float = Field(default=0.0)
    vacancy_rate: float = Field(default=0.08)
    mgmt_fee_rate: float = Field(default=0.08)
    maintenance_annual: Optional[float] = None
    capex_reserve_annual: Optional[float] = None
    closing_costs_pct: float = Field(default=0.03, ge=0.0, le=0.10,
        description="Closing costs as % of purchase price")
    selling_costs_pct: float = Field(default=0.06, ge=0.0, le=0.15,
        description="Selling costs (agent fees etc.) as % of exit price")

    class Config:
        json_schema_extra = {
            "example": {
                "purchase_price": 389000,
                "monthly_rent": 2950,
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
    deal_rating: str          # "Excellent" | "Good" | "Fair" | "Poor"
    deal_rating_reason: str


class FullAnalysisRequest(BaseModel):
    """Convenience endpoint — runs all three algorithms in one call."""
    purchase_price: float
    sqft: float
    bedrooms: int
    bathrooms: float
    property_type: str = "single_family"
    market_tier: str = "mid_cost"
    year_built: Optional[int] = None
    has_garage: bool = False
    has_pool: bool = False
    down_payment_pct: float = 0.20
    interest_rate: float = 0.07
    loan_term_years: int = 30
    annual_appreciation_rate: float = 0.04
    hold_years: int = 5
    monthly_hoa: float = 0.0
    vacancy_rate: float = 0.08
    mgmt_fee_rate: float = 0.08
    closing_costs_pct: float = 0.03

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


# ─────────────────────────────────────────────
# ALGORITHM 1: RENTAL INCOME ESTIMATOR
# ─────────────────────────────────────────────

def compute_rental_estimate(req: RentalEstimateRequest) -> RentalEstimateResponse:
    """
    Estimates market rent using:
    - Base rent-per-sqft for market tier
    - Bedroom/bathroom adjustment factors
    - Age depreciation curve
    - Amenity premiums (garage, pool)
    - Property type modifier
    """

    # 1. Base rent from sqft × market rate
    region = REGIONAL_RENT_SQFT.get(req.market_tier, REGIONAL_RENT_SQFT["mid_cost"])
    base_rate = region["base"]          # $/sqft/month
    std = region["std"]

    # 2. Bedroom adjustment (diminishing returns above 3BR)
    br_adjustments = {0: 0.55, 1: 0.75, 2: 0.90, 3: 1.00, 4: 1.10, 5: 1.17}
    br_factor = br_adjustments.get(req.bedrooms, 1.17 + (req.bedrooms - 5) * 0.04)

    # 3. Bathroom adjustment
    ba_factor = 1.0 + max(0, (req.bathrooms - 1.5)) * 0.04

    # 4. Property type modifier
    type_mods = {
        "single_family": 1.00,
        "multi_family":  0.92,   # units typically rent for slightly less
        "condo":         0.97,
        "commercial":    1.20,
    }
    type_factor = type_mods.get(req.property_type, 1.00)

    # 5. Age depreciation (newer = higher rent)
    age_factor = 1.0
    if req.year_built:
        age = 2025 - req.year_built
        if age <= 5:
            age_factor = 1.12
        elif age <= 10:
            age_factor = 1.07
        elif age <= 20:
            age_factor = 1.03
        elif age <= 35:
            age_factor = 1.00
        elif age <= 50:
            age_factor = 0.96
        else:
            age_factor = 0.91

    # 6. Amenity premiums (flat monthly $)
    amenity_premium = 0
    if req.has_garage:
        amenity_premium += 100 if req.market_tier == "mid_cost" else 150
    if req.has_pool:
        amenity_premium += 75 if req.market_tier == "mid_cost" else 125

    # 7. Compose estimate
    adjusted_rate = base_rate * br_factor * ba_factor * type_factor * age_factor
    base_rent = req.sqft * adjusted_rate + amenity_premium

    # 8. Confidence — reduced for unusual sqft or older properties
    confidence = 84
    if req.sqft > 4000 or req.sqft < 400:
        confidence -= 8
    if req.year_built and (2025 - req.year_built) > 60:
        confidence -= 5
    if req.market_tier == "high_cost":
        confidence -= 3   # more volatile markets
    confidence = max(60, min(95, confidence))

    # 9. Range based on market std dev
    range_factor = std / base_rate
    rent_low  = round(base_rent * (1 - range_factor * 0.8), -1)
    rent_high = round(base_rent * (1 + range_factor * 0.8), -1)
    base_rent = round(base_rent, -1)

    return RentalEstimateResponse(
        estimated_monthly_rent=base_rent,
        rent_range_low=rent_low,
        rent_range_high=rent_high,
        confidence_pct=confidence,
        rent_per_sqft=round(adjusted_rate, 3),
        gross_annual_rent=round(base_rent * 12, 2),
        methodology=(
            f"Estimate based on ${adjusted_rate:.2f}/sqft ({req.market_tier} market), "
            f"{req.bedrooms}BR factor={br_factor:.2f}, "
            f"age factor={age_factor:.2f}, "
            f"amenity premium=${amenity_premium}/mo"
        )
    )


# ─────────────────────────────────────────────
# ALGORITHM 2: CASH FLOW ENGINE
# ─────────────────────────────────────────────

def compute_mortgage_payment(principal: float, annual_rate: float, years: int) -> float:
    """Standard amortizing mortgage payment (P&I)."""
    if annual_rate == 0:
        return principal / (years * 12)
    r = annual_rate / 12
    n = years * 12
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def compute_cash_flow(req: CashFlowRequest) -> CashFlowResponse:
    """
    Full income-expense cash flow model.
    All dollar values returned as monthly figures except where noted.
    """

    # Loan mechanics
    down_payment = req.purchase_price * req.down_payment_pct
    loan_amount  = req.purchase_price - down_payment
    mortgage     = compute_mortgage_payment(loan_amount, req.interest_rate, req.loan_term_years)

    # Closing cost estimate
    closing_costs = req.purchase_price * 0.03
    total_cash_invested = down_payment + closing_costs

    # ── INCOME ──
    effective_monthly = req.monthly_rent * (1 - req.vacancy_rate)

    # ── EXPENSES ──
    # Property tax
    prop_tax_monthly = (
        req.annual_property_tax / 12
        if req.annual_property_tax is not None
        else (req.purchase_price * DEFAULT_EXPENSE_RATIOS["property_tax_rate"]) / 12
    )

    # Insurance
    insurance_monthly = (
        req.annual_insurance / 12
        if req.annual_insurance is not None
        else (req.purchase_price * DEFAULT_EXPENSE_RATIOS["insurance_rate"]) / 12
    )

    # Maintenance
    maintenance_monthly = (
        req.maintenance_annual / 12
        if req.maintenance_annual is not None
        else (req.purchase_price * DEFAULT_EXPENSE_RATIOS["maintenance_rate"]) / 12
    )

    # CapEx reserve
    capex_monthly = (
        req.capex_reserve_annual / 12
        if req.capex_reserve_annual is not None
        else (req.monthly_rent * 12 * DEFAULT_EXPENSE_RATIOS["capex_reserve_rate"]) / 12
    )

    # Property management
    mgmt_monthly = req.monthly_rent * req.mgmt_fee_rate

    total_expenses = (
        mortgage
        + prop_tax_monthly
        + insurance_monthly
        + req.monthly_hoa
        + maintenance_monthly
        + capex_monthly
        + mgmt_monthly
    )

    net_monthly = effective_monthly - total_expenses

    return CashFlowResponse(
        gross_monthly_rent=round(req.monthly_rent, 2),
        effective_gross_income_monthly=round(effective_monthly, 2),
        mortgage_payment=round(mortgage, 2),
        property_tax_monthly=round(prop_tax_monthly, 2),
        insurance_monthly=round(insurance_monthly, 2),
        hoa_monthly=round(req.monthly_hoa, 2),
        maintenance_monthly=round(maintenance_monthly, 2),
        capex_reserve_monthly=round(capex_monthly, 2),
        mgmt_fee_monthly=round(mgmt_monthly, 2),
        total_expenses_monthly=round(total_expenses, 2),
        net_monthly_cash_flow=round(net_monthly, 2),
        net_annual_cash_flow=round(net_monthly * 12, 2),
        loan_amount=round(loan_amount, 2),
        down_payment=round(down_payment, 2),
        total_cash_invested=round(total_cash_invested, 2),
    )


# ─────────────────────────────────────────────
# ALGORITHM 3: INVESTMENT METRICS
# ─────────────────────────────────────────────

def compute_irr(cash_flows: list[float]) -> float:
    """
    Newton-Raphson IRR solver.
    cash_flows[0] = initial investment (negative)
    cash_flows[1:] = annual net cash flows + terminal proceeds
    Returns annual IRR as decimal. Returns NaN if no solution found.
    """
    if not cash_flows or cash_flows[0] >= 0:
        return float('nan')

    # Try numpy's IRR via NPV root-finding
    def npv(rate, flows):
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))

    def npv_deriv(rate, flows):
        return sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(flows))

    rate = 0.10   # initial guess
    for _ in range(1000):
        npv_val   = npv(rate, cash_flows)
        npv_d_val = npv_deriv(rate, cash_flows)
        if abs(npv_d_val) < 1e-12:
            break
        new_rate = rate - npv_val / npv_d_val
        if abs(new_rate - rate) < 1e-8:
            return new_rate
        rate = new_rate

    return float('nan')


def compute_loan_balance(principal: float, annual_rate: float, years: int, periods_paid: int) -> float:
    """Remaining loan balance after `periods_paid` monthly payments."""
    if annual_rate == 0:
        return principal - (principal / (years * 12)) * periods_paid
    r = annual_rate / 12
    n = years * 12
    payment = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return principal * (1 + r) ** periods_paid - payment * ((1 + r) ** periods_paid - 1) / r


def compute_investment_metrics(req: InvestmentMetricsRequest) -> InvestmentMetricsResponse:
    """
    Computes:
    - Cap Rate
    - Cash-on-Cash Return
    - IRR (leveraged, after financing)
    - Gross Rent Multiplier
    - Break-even horizon
    - Total Return
    """

    # --- Build the cash flow model first ---
    cf_req = CashFlowRequest(
        purchase_price=req.purchase_price,
        monthly_rent=req.monthly_rent,
        down_payment_pct=req.down_payment_pct,
        interest_rate=req.interest_rate,
        loan_term_years=req.loan_term_years,
        annual_property_tax=req.annual_property_tax,
        annual_insurance=req.annual_insurance,
        monthly_hoa=req.monthly_hoa,
        vacancy_rate=req.vacancy_rate,
        mgmt_fee_rate=req.mgmt_fee_rate,
        maintenance_annual=req.maintenance_annual,
        capex_reserve_annual=req.capex_reserve_annual,
    )
    cf = compute_cash_flow(cf_req)

    # ── CAP RATE ──
    # NOI = gross rents × (1 - vacancy) - operating expenses (excl. debt service)
    annual_noi = (
        cf.effective_gross_income_monthly * 12
        - (cf.property_tax_monthly + cf.insurance_monthly + cf.hoa_monthly
           + cf.maintenance_monthly + cf.capex_reserve_monthly + cf.mgmt_fee_monthly) * 12
    )
    cap_rate = annual_noi / req.purchase_price

    # ── CASH-ON-CASH ──
    annual_cash_flow = cf.net_annual_cash_flow
    cash_on_cash = annual_cash_flow / cf.total_cash_invested if cf.total_cash_invested > 0 else 0

    # ── GRM ──
    gross_annual_rent = req.monthly_rent * 12
    grm = req.purchase_price / gross_annual_rent if gross_annual_rent > 0 else 0

    # ── IRR ──
    # Year 0: initial outlay (negative)
    initial_investment = -cf.total_cash_invested

    # Years 1..N: annual net cash flows
    irr_cash_flows = [initial_investment]
    for year in range(1, req.hold_years + 1):
        # Assume rents grow ~2% per year (conservative)
        rent_growth = (1.02) ** (year - 1)
        annual_cf = annual_cash_flow * rent_growth
        irr_cash_flows.append(annual_cf)

    # Terminal year: add net sale proceeds
    exit_price = req.purchase_price * (1 + req.annual_appreciation_rate) ** req.hold_years
    selling_costs = exit_price * req.selling_costs_pct
    remaining_loan = compute_loan_balance(
        cf.loan_amount, req.interest_rate, req.loan_term_years, req.hold_years * 12
    )
    net_sale_proceeds = exit_price - selling_costs - remaining_loan
    irr_cash_flows[-1] += net_sale_proceeds

    irr_raw = compute_irr(irr_cash_flows)
    irr_pct = irr_raw * 100 if not math.isnan(irr_raw) else 0.0

    # ── BREAK-EVEN ──
    # How many months of cash flow to recoup total cash invested
    if annual_cash_flow > 0:
        break_even_years = cf.total_cash_invested / annual_cash_flow
    else:
        break_even_years = 99.0   # never breaks even on cash flow alone

    # ── TOTAL RETURN ──
    total_equity_at_exit = net_sale_proceeds
    total_cash_from_ops  = annual_cash_flow * req.hold_years   # simplified
    total_return = (total_equity_at_exit + total_cash_from_ops - cf.total_cash_invested) / cf.total_cash_invested * 100

    # ── DEAL RATING ──
    if cap_rate >= 0.08 and cash_on_cash >= 0.10 and irr_pct >= 15:
        rating = "Excellent"
        reason = f"Cap rate {cap_rate*100:.1f}% + CoC {cash_on_cash*100:.1f}% + IRR {irr_pct:.1f}% all above target thresholds."
    elif cap_rate >= 0.06 and cash_on_cash >= 0.07 and irr_pct >= 10:
        rating = "Good"
        reason = f"Solid fundamentals. Cap rate {cap_rate*100:.1f}%, CoC {cash_on_cash*100:.1f}%, IRR {irr_pct:.1f}%."
    elif cap_rate >= 0.04 and annual_cash_flow > 0:
        rating = "Fair"
        reason = f"Marginal returns. Cap rate {cap_rate*100:.1f}% is below 6% target. Cash flow is positive but thin."
    else:
        rating = "Poor"
        reason = f"Below investment thresholds. Cap rate {cap_rate*100:.1f}% and/or negative cash flow."

    return InvestmentMetricsResponse(
        cap_rate_pct=round(cap_rate * 100, 2),
        cash_on_cash_return_pct=round(cash_on_cash * 100, 2),
        irr_pct=round(irr_pct, 2),
        gross_rent_multiplier=round(grm, 2),
        break_even_years=round(break_even_years, 1),
        total_return_pct=round(total_return, 1),
        equity_at_exit=round(total_equity_at_exit, 2),
        projected_exit_price=round(exit_price, 2),
        net_monthly_cash_flow=round(cf.net_monthly_cash_flow, 2),
        net_annual_cash_flow=round(annual_cash_flow, 2),
        deal_rating=rating,
        deal_rating_reason=reason,
    )


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "DealIQ Algorithm Engine v1.0"}


@app.post("/api/rental-estimate", response_model=RentalEstimateResponse, tags=["Algorithms"])
def rental_estimate(req: RentalEstimateRequest):
    """
    **Rental Income Estimator**

    Estimates potential monthly rent based on property characteristics
    and market tier using comparable-rental methodology.
    """
    try:
        return compute_rental_estimate(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/cash-flow", response_model=CashFlowResponse, tags=["Algorithms"])
def cash_flow(req: CashFlowRequest):
    """
    **Cash Flow Engine**

    Full P&L model: mortgage, taxes, insurance, HOA, maintenance,
    CapEx reserve, management fees → monthly & annual net cash flow.
    """
    try:
        return compute_cash_flow(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/investment-metrics", response_model=InvestmentMetricsResponse, tags=["Algorithms"])
def investment_metrics(req: InvestmentMetricsRequest):
    """
    **Investment Metrics**

    Computes Cap Rate, Cash-on-Cash Return, IRR (leveraged),
    GRM, break-even horizon, total return, and deal rating.
    """
    try:
        return compute_investment_metrics(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/full-analysis", tags=["Algorithms"])
def full_analysis(req: FullAnalysisRequest):
    """
    **Full Deal Analysis** (all three algorithms in one call)

    Convenience endpoint that runs Rental Estimator → Cash Flow →
    Investment Metrics and returns a unified analysis object.
    """
    try:
        # Step 1: estimate rent
        rent_req = RentalEstimateRequest(
            sqft=req.sqft,
            bedrooms=req.bedrooms,
            bathrooms=req.bathrooms,
            property_type=req.property_type,
            market_tier=req.market_tier,
            year_built=req.year_built,
            has_garage=req.has_garage,
            has_pool=req.has_pool,
        )
        rent = compute_rental_estimate(rent_req)

        # Step 2: cash flow using estimated rent
        metrics_req = InvestmentMetricsRequest(
            purchase_price=req.purchase_price,
            monthly_rent=rent.estimated_monthly_rent,
            down_payment_pct=req.down_payment_pct,
            interest_rate=req.interest_rate,
            loan_term_years=req.loan_term_years,
            annual_appreciation_rate=req.annual_appreciation_rate,
            hold_years=req.hold_years,
            monthly_hoa=req.monthly_hoa,
            vacancy_rate=req.vacancy_rate,
            mgmt_fee_rate=req.mgmt_fee_rate,
            closing_costs_pct=req.closing_costs_pct,
        )
        metrics = compute_investment_metrics(metrics_req)

        return {
            "property_summary": {
                "purchase_price": req.purchase_price,
                "sqft": req.sqft,
                "bedrooms": req.bedrooms,
                "bathrooms": req.bathrooms,
                "property_type": req.property_type,
                "market_tier": req.market_tier,
            },
            "rental_estimate": rent,
            "investment_metrics": metrics,
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ─────────────────────────────────────────────
# RUN (dev only)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dealiq_api:app", host="0.0.0.0", port=8000, reload=True)
