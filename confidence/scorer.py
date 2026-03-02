def compute_confidence(retrieval_score, alignment_score):

    final_score = (0.4 * retrieval_score) + (0.6 * alignment_score)

    if final_score >= 85:
        risk = "Low"
    elif final_score >= 70:
        risk = "Medium"
    else:
        risk = "High"

    return round(final_score, 2), risk