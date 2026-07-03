MARKET = "IL"  #"US", "UK", "IL"
ANNUAL = True
FAT_TAIL_DF = 5  # 10 for negligible fat tail, 3 for very fat tail - if None - use normal distribution

portfolio_real_return_mean_by_market = {
    "US": 0.03,  # long‑run real return µ in US
    "UK": 0.053,  # long‑run real return µ in UK
    "IL": 0.042  # 0.055 (Gemini) 0.05 (chatgpt) long‑run real return µ in IL, 0.42 assumes 75/25 stocks portfolio
}
portfolio_real_return_sd_by_market = {
    "US": 0.12,
    "UK": 0.20,
    "IL": 0.13 # 0.155 (Gemini) 0.117 (chatgpt), 0.18 for high volatility, 0.13 assumes 75/25 stocks portfolio
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