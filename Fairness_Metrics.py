import numpy as np
import math

# Compute the absolute difference between metric scores for male ('m') and female ('f') users.
def compute_recGap(metric_by_gender):

    rec_gap = abs(metric_by_gender['m'] - metric_by_gender['f'])
    print("RecGap Score:", rec_gap, "\n")


def compute_compounding_factor(df, metric_by_gender, epsilon=1e-10):
#  Count number of users per gender
    g_counts = df['gender'].value_counts()
    total_users = len(df)
    #  Compute original data distribution (P)
    # p[0] = proportion of male users, p[1] = proportion of female users
    p = np.array([g_count / total_users for g_count in g_counts])
    print("Original data distribution:")
    print("Males:", p[0])
    print("Females:", p[1])
    #Compute weighted total metric value
    total_metric = p[0] * metric_by_gender['m'] + p[1] * metric_by_gender['f']
    q_male = (p[0] * metric_by_gender['m']) / total_metric
    q_female = (p[1] * metric_by_gender['f']) / total_metric

    print("\nMetric score distribution:")
    print("Males:", q_male)
    print("Females:", q_female)
    # Compute performance-based distribution (Q)
    # q_male = proportion of performance attributed to males
    # q_female = proportion of performance attributed to females
    q_male = max(q_male, epsilon)
    q_female = max(q_female, epsilon)

    # Compute the KL divergence:
    kl_divergence = p[0] * math.log(p[0] / q_male) + p[1] * math.log(p[1] / q_female)

    print("\nCompounding Factor (KL Divergence):", kl_divergence)