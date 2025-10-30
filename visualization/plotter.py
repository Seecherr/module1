import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config.settings import Config

class ResultPlotter:
    """Creates visualization plots"""
    
    def __init__(self):
        sns.set_theme(style="whitegrid")
        self.fig = None
        self.ax = None
    
    def create_plot(self, data_dict):
        """Create main comparison plot"""
        self.fig, self.ax = plt.subplots(figsize=Config.FIGURE_SIZE)
        
        # Plot experimental points
        self._plot_experimental_points(data_dict)
        
        # Plot models
        self._plot_models(data_dict)
        
        # Plot optimal speed lines
        self._plot_optimal_speed(data_dict)
        
        # Configure plot
        self._configure_plot()
        
        return self.fig
    
    def _plot_experimental_points(self, data):
        """Plot experimental data points"""
        self.ax.scatter(data['X'][data['inlier_mask']], data['y'][data['inlier_mask']], 
                       color='blue', alpha=0.7, label='Експериментальні точки (Inliers)')
        self.ax.scatter(data['X'][data['outlier_mask']], data['y'][data['outlier_mask']], 
                       color='red', marker='x', s=100, label='Викиди (Outliers)')
    
    def _plot_models(self, data):
        """Plot model predictions"""
        # Linear model
        self.ax.plot(data['X_plot'], data['y_linear'], color='orange', linestyle='--', 
                    label=f'Лінійна модель (R²={data["metrics_df"].loc[0, "R^2"]:.2f})')
        
        # Quadratic model
        self.ax.plot(data['X_plot'], data['y_quad'], color='green', linestyle='-', linewidth=2, 
                    label=f'Квадратична модель (R²={data["metrics_df"].loc[1, "R^2"]:.2f})')
        
        # RANSAC model
        self.ax.plot(data['X_plot'], data['y_ransac'], color='purple', linestyle='-.', linewidth=2, 
                    label=f'RANSAC модель (v_opt={data["v_opt_ransac"]:.1f} км/год)')
    
    def _plot_optimal_speed(self, data):
        """Plot optimal speed indicators"""
        # Optimal speed line
        self.ax.axvline(x=data['v_opt'], color='green', linestyle=':', 
                       label=f'Оптимальна швидкість (Std. Model): {data["v_opt"]:.1f} км/год')
        self.ax.axhline(y=data['y_opt'], color='green', linestyle=':', xmax=data['v_opt']/150)
        
        # Confidence interval
        self.ax.axvspan(data['ci_lower'], data['ci_upper'], color='green', alpha=0.1, 
                       label=f'95% CI для v_opt [{data["ci_lower"]:.1f}, {data["ci_upper"]:.1f}]')
    
    def _configure_plot(self):
        """Configure plot appearance"""
        self.ax.set_title('Модель витрат палива автомобіля (Команда 2)', fontsize=16)
        self.ax.set_xlabel('Швидкість (км/год)', fontsize=12)
        self.ax.set_ylabel('Витрата палива (л/100 км)', fontsize=12)
        self.ax.legend(loc='upper left')
        self.ax.set_ylim(bottom=0)
        self.ax.set_xlim(*Config.PLOT_RANGE)
        self.ax.grid(True)
    
    def save_plot(self, filename=Config.OUTPUT_PLOT_FILE):
        """Save plot to file"""
        if self.fig:
            self.fig.savefig(filename)
            print(f"Графік збережено у файл: {filename}")
