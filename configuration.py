MARKET = "IL"  #"US", "UK", "IL"

real_return_mean_by_market = {
    "US": 0.03,  # long‑run real return µ in US
    "UK": 0.053,  # long‑run real return µ in UK
    "IL": 0.055  # 0.055 (Gemini) 0.05 (chatgpt) long‑run real return µ in IL
}
real_return_sd_by_market = {
    "US": 0.12,
    "UK": 0.20,
    "IL": 0.23 # 0.155 (Gemini) 0.117 (chatgpt)
}
growth_mean_by_market = {
    "US": 0.01,
    "UK": 0.024,
    "IL": 0.018
}
growth_sd_by_market = {
    "US": 0.06,
    "UK": 0.07,
    "IL": 0.08
}