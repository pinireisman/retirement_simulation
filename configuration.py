MARKET = "IL"  #"US", "UK", "IL"
ANNUAL = True

portfolio_real_return_mean_by_market = {
    "US": 0.03,  # long‑run real return µ in US
    "UK": 0.053,  # long‑run real return µ in UK
    "IL": 0.04  # 0.055 (Gemini) 0.05 (chatgpt) long‑run real return µ in IL, 0.4 is conservative
}
portfolio_real_return_sd_by_market = {
    "US": 0.12,
    "UK": 0.20,
    "IL": 0.12  # 0.155 (Gemini) 0.117 (chatgpt), 0.23 for high volatility
}
housing_growth_mean_by_market = {
    "US": 0.01,
    "UK": 0.024,
    "IL": 0.018
}
housing_growth_sd_by_market = {
    "US": 0.06,
    "UK": 0.07,
    "IL": 0.08
}