import numpy as np

class Config:
    """Configuration settings for the fuel consumption analysis"""
    
    RANDOM_SEED = 42
    N_SAMPLES = 100
    SPEED_RANGE = (10, 140)
    NOISE_STD = 0.75
    
    TRUE_A = 10
    TRUE_B = -0.6
    TRUE_C = 0.005
    
    OUTLIERS_SPEED = [20, 25, 30]
    OUTLIERS_CONSUMPTION = [15, 17, 14]
    
    N_BOOTSTRAP_ITERATIONS = 1000
    CONFIDENCE_LEVEL = 95
    
    RANSAC_MIN_SAMPLES = 50
    RANSAC_RESIDUAL_THRESHOLD = 1.0
    RANSAC_MAX_TRIALS = 100
    
    PLOT_RANGE = (0, 150)
    PLOT_POINTS = 200
    FIGURE_SIZE = (14, 8)
    
    OUTPUT_PLOT_FILE = 'team(2)_plots.png'
    OUTPUT_SUMMARY_FILE = 'team(2)_summary.csv'
    OUTPUT_METRICS_FILE = 'team(2)_metrics.csv'
