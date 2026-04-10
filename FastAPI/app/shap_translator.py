# Maps your ML features to plain English AML concepts
SHAP_GLOSSARY = {
    "Payment Format_target_enc": "Payment was made via ACH, a channel with historically high fraud rates in our system.",
    "Receiving Currency_target_enc": "Transaction involved a high-risk receiving currency.",
    "Payment Currency_target_enc": "Transaction involved a high-risk payment currency.",
    "anomaly_score": "Transaction behavior is statistically anomalous compared to this account's normal baseline.",
    "txns_in_directed_pair": "Abnormally high volume of transactions with this specific counterparty, suggesting a coordinated loop.",
    "is_toxic_corridor": "Funds moved through a known high-risk banking corridor.",
    "burst_score_1h": "Sudden burst of rapid transactions within a single hour, indicative of automated bot activity (Smurfing).",
    "txn_in_hour": "Unusually high number of transactions processed within a single hour.",
    "amount_vs_baseline_ratio": "Transaction amount is significantly higher than this account's historical average.",
    "counterparty_diversity_28d": "Account has been interacting with an unusually high number of distinct counterparties.",
    "flag_heavy_structuring": "Triggered the heavy structuring rule based on amount patterns.",
    "From Bank_freq_enc": "Originating bank has a very low transaction frequency, potentially indicating a shell or newly created institution.",
    "To Bank_freq_enc": "Receiving bank has a very low transaction frequency."
}

def translate_shap_for_llm(shap_explanation_list: list) -> str:
    """
    Converts API SHAP JSON into a readable string for the LLM prompt.
    """
    context = []
    for item in shap_explanation_list:
        feat = item["feature"]
        # Default description if feature is not in our glossary
        desc = SHAP_GLOSSARY.get(feat, f"Feature '{feat}' showed unusual activity (Value: {item['value']:.2f}).")
        
        direction = "increased" if item["shap_value"] > 0 else "decreased"
        context.append(f"- {desc} This {direction} the fraud risk score.")
        
    return "\n".join(context)