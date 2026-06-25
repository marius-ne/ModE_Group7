# scripts/run_sampling.py
from src.sampling.core import create_sample, save_samples, plot_samples, nr_datapoints, generate_test_sample

if __name__ == "__main__":
    # sobol_df, sobol_quality = create_sample("sobol")
    # save_samples(sobol_df, sobol_quality, "sobol", "training")
    #
    # lhs_df, lhs_quality = create_sample("lhs")
    # save_samples(lhs_df, lhs_quality, "lhs", "training")
    #nr_datapoints(10) # number of ratos if sampling methos is "log" or for random sampling
    generate_test_sample(10)
    #log_df = create_sample("log")
    #save_samples(log_df, None, "log", "training_20")

    # plot_samples(sobol_df, sobol_quality, lhs_df, lhs_quality)
    0