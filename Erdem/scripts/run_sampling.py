# scripts/run_sampling.py
from src.sampling.core import create_sample, save_samples, plot_samples

if __name__ == "__main__":

    # ============================================================
    # Global parameter
    # ============================================================
    N_SAMPLES_TRAIN = 40
    N_SAMPLES_TEST = 10
    N_CORNER = 4
    N_EDGES = 4

    SAMPLE_TYPE = "test" # "training" or "test"
    TYPE_CONFIG = {
        "training": N_SAMPLES_TRAIN,
        "test": N_SAMPLES_TEST
    }
    FILE_SUFFIX = str(TYPE_CONFIG[SAMPLE_TYPE])

    # ============================================================
    # Sampling config per method
    # ============================================================
    SAMPLING_CONFIG = {
        "sobol": {
            "n_total": TYPE_CONFIG[SAMPLE_TYPE],
            "n_corner": N_CORNER,
            "n_edges": N_EDGES,
            "save_quality": True,
        },
        "lhs": {
            "n_total": TYPE_CONFIG[SAMPLE_TYPE],
            "n_corner": N_CORNER,
            "n_edges": 0,
            "save_quality": True,
        },
        "log": {
            "n_total": TYPE_CONFIG[SAMPLE_TYPE],
            "n_corner": 0,
            "n_edges": 0,
            "save_quality": False,  # 1D, keine quality json
        },
        "random": {
            "n_total": TYPE_CONFIG[SAMPLE_TYPE],
            "n_corner": 0,
            "n_edges": 0,
            "save_quality": True,
        },
    }

    sample_dict = {}

    # ============================================================
    # Create and save samples
    # ============================================================
    for method in SAMPLING_CONFIG:
        cfg = SAMPLING_CONFIG[method]
        print(f"\n--- {method.upper()} Sampling ---")

        result = create_sample(
            method,
            n_total=cfg["n_total"],
            n_corner=cfg["n_corner"],
            n_edges=cfg["n_edges"],
        )

        if method == "log":
            samples = result
            sample_quality = None
        else:
            samples, sample_quality = result

        save_samples(
            samples=samples,
            sample_quality=sample_quality if cfg["save_quality"] else None,
            sampling_method=method,
            sample_type=SAMPLE_TYPE,
            file_name=FILE_SUFFIX,
        )

        sample_dict[method] = (samples, sample_quality)


    # ============================================================
    # Visualize samples
    # ============================================================

    # Plot Sobol and LH sampling
    # plot_samples(sobol_df, sobol_quality, lhs_df, lhs_quality)