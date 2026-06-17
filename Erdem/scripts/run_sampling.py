# scripts/run_sampling.py
from src.sampling.core import create_sample, save_samples, plot_samples

if __name__ == "__main__":
    sobol_df, sobol_quality = create_sample("sobol")
    save_samples(sobol_df, sobol_quality, "sobol", "training")

    lhs_df, lhs_quality = create_sample("lhs")
    save_samples(lhs_df, lhs_quality, "lhs", "training")

    plot_samples(sobol_df, sobol_quality, lhs_df, lhs_quality)