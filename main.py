import numpy as np

from data.generator import DataGenerator
from data.processor import DataProcessor
from models.linear_model import LinearRegressionModel
from models.quadratic_model import QuadraticRegressionModel
from models.ransac_model import RANSACRegressionModel
from analysis.metrics import MetricsCalculator
from analysis.optimizer import FuelOptimizer
from analysis.bootstrap import BootstrapAnalyzer
from visualization.plotter import ResultPlotter
from utils.file_handler import FileHandler


def _section(title: str) -> None:
    """Print a standardized section banner."""
    print(title)


def main() -> None:
    """Main execution function"""
    print("==============================================")
    print("==============================================\n")

    # Step 1: Generate data
    _section("----------- ЕТАП 1: ГЕНЕРАЦІЯ ДАНИХ -----------")
    X, y, v_data, y_data = DataGenerator.generate_data()
    print(f"Згенеровано {len(X)} точок даних (включаючи 3 викиди).\n")

    # Step 2: Prepare features
    X_poly, poly_features = DataProcessor.create_polynomial_features(X)
    X_plot = DataProcessor.create_plot_data()
    X_plot_poly = poly_features.transform(X_plot)

    # Step 3: Train models
    _section("----------- ЕТАП 2: НАВЧАННЯ МОДЕЛЕЙ -----------")

    # Linear model
    linear_model = LinearRegressionModel()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    y_plot_linear = linear_model.predict(X_plot)

    # Quadratic model
    quadratic_model = QuadraticRegressionModel()
    quadratic_model.fit(X_poly, y)
    y_pred_quad = quadratic_model.predict(X_poly)
    y_plot_quad = quadratic_model.predict(X_plot_poly)

    # RANSAC model
    ransac_model = RANSACRegressionModel()
    ransac_model.fit(X_poly, y)
    y_plot_ransac = ransac_model.predict(X_plot_poly)
    outlier_info = ransac_model.get_outlier_info(y)

    # Display model equations
    lin_coef, lin_intercept = linear_model.get_coefficients()
    quad_coef, quad_intercept = quadratic_model.get_coefficients()
    ransac_coef, ransac_intercept = ransac_model.get_coefficients()

    print(f"Лінійна модель: y = {lin_intercept:.3f} + {lin_coef[0]:.3f} * v")
    print(
        "Квадратична модель: "
        f"y = {quad_intercept:.3f} + {quad_coef[0]:.3f} * v + {quad_coef[1]:.3f} * v^2"
    )
    print(
        "RANSAC модель: "
        f"y = {ransac_intercept:.3f} + {ransac_coef[0]:.3f} * v + {ransac_coef[1]:.3f} * v^2\n"
    )

    # Step 4: Calculate metrics
    _section("----------- ЕТАП 3: ПОРІВНЯННЯ МЕТРИК -----------")
    metrics_calc = MetricsCalculator()
    metrics_list = [
        metrics_calc.calculate_all_metrics(y, y_pred_linear, "Linear"),
        metrics_calc.calculate_all_metrics(y, y_pred_quad, "Quadratic"),
    ]
    metrics_df = metrics_calc.create_metrics_dataframe(metrics_list)
    print(metrics_df.to_string(index=False))
    print()

    # Step 5: Optimization
    _section("----------- ЕТАП 4: ОПТИМІЗАЦІЯ -----------")
    v_opt = quadratic_model.get_optimal_speed()
    y_opt = FuelOptimizer.calculate_optimal_consumption(quadratic_model, v_opt)
    v_opt_ransac = ransac_model.get_optimal_speed()

    print(f"Розрахункова оптимальна швидкість (v_opt): {v_opt:.2f} км/год")
    print(f"Мінімальна прогнозована витрата палива: {y_opt:.2f} л/100 км")
    print(f"RANSAC v_opt: {v_opt_ransac:.2f} км/год\n")

    # Step 6: Bootstrap analysis
    bootstrap_analyzer = BootstrapAnalyzer()
    ci_lower, ci_upper, v_opt_samples = bootstrap_analyzer.analyze(X_poly, y)

    print("Бутстреп завершено.")
    print(
        f"95% довірчий інтервал для v_opt: [{ci_lower:.2f}, {ci_upper:.2f}] км/год\n"
    )

    # Step 7: RANSAC results
    _section("----------- ЕТАП 6: РОБАСТНА РЕГРЕСІЯ (RANSAC) -----------")
    print(f"RANSAC ідентифікував {outlier_info['n_outliers']} викидів.\n")

    # Step 8: Visualization
    _section("----------- ЕТАП 7: ВІЗУАЛІЗАЦІЯ -----------")
    plot_data = {
        "X": X.ravel(),
        "y": y,
        "inlier_mask": ransac_model.inlier_mask,
        "outlier_mask": outlier_info["outlier_mask"],
        "X_plot": X_plot.ravel(),
        "y_linear": y_plot_linear,
        "y_quad": y_plot_quad,
        "y_ransac": y_plot_ransac,
        "metrics_df": metrics_df,
        "v_opt": v_opt,
        "y_opt": y_opt,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "v_opt_ransac": v_opt_ransac,
    }

    plotter = ResultPlotter()
    plotter.create_plot(plot_data)
    plotter.save_plot()
    print()

    # Step 9: Save results
    _section("----------- ЕТАП 8: ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ -----------")
    FileHandler.save_summary(quadratic_model, v_opt, y_opt, ci_lower, ci_upper, v_opt_ransac)
    FileHandler.save_metrics(metrics_df)

    print("Підсумкові коефіцієнти збережено у: team(2)_summary.csv")
    print("Підсумкові метрики збережено у: team(2)_metrics.csv\n")

    print("==============================================")
    print("РОБОТУ КОМАНДИ 2 ЗАВЕРШЕНО.")
    print("==============================================")

    # Display plot if running in interactive environment
    # plt.show()


if __name__ == "__main__":
    main()
