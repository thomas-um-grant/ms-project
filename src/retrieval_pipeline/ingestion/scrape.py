# Load Hugging Face ViDoRe datasets
import logging

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIDORE_DATASETS = [
    "vidore/arxivqa_test_subsampled_beir",
    "vidore/docvqa_test_subsampled_beir",
    "vidore/infovqa_test_subsampled_beir",
    "vidore/tabfquad_test_subsampled_beir",
    "vidore/tatdqa_test_beir",
    "vidore/shiftproject_test_beir",
    "vidore/syntheticDocQA_artificial_intelligence_test_beir",
    "vidore/syntheticDocQA_energy_test_beir",
    "vidore/syntheticDocQA_government_reports_test_beir",
    "vidore/syntheticDocQA_healthcare_industry_test_beir",
    "vidore/esg_reports_v2",
    "vidore/biomedical_lectures_v2",
    "vidore/economics_reports_v2",
    "vidore/esg_reports_human_labeled_v2",
]


def load_vidore_dataset():
    """Load the ViDoRe dataset from Hugging Face."""
    try:
        # Load the ViDoRe dataset
        dataset = load_dataset("vidore/vidore")
        logger.info("ViDoRe dataset loaded successfully.")

        # Extract corpus
        corpus = dataset["corpus"]

    except Exception as e:
        logger.exception(f"Failed to load ViDoRe datasets: {e}")
        raise


def main():
    """Main function to load the ViDoRe datasets."""
    try:
        for dataset_name in VIDORE_DATASETS:
            logger.info(f"Loading dataset: {dataset_name} ...")
            load_dataset(dataset_name)

        logger.info("All ViDoRe datasets loaded successfully.")

    except Exception as e:
        msg = f"An error occurred while loading the datasets: {e}"
        logger.exception(msg)


if __name__ == "__main__":
    main()
