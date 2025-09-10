from enum import Enum


class BEIRDatasets(Enum):
    ARXIVQA_DATASET = "vidore/arxivqa_test_subsampled_beir"
    DOCVQA_DATASET = "vidore/docvqa_test_subsampled_beir"
    INFOVQA_DATASET = "vidore/infovqa_test_subsampled_beir"
    TABFQUAD_DATASET = "vidore/tabfquad_test_subsampled_beir"
    TATQDA_DATASET = "vidore/tatdqa_test_beir"
    GOVERNMENTAL_DATASET = "vidore/syntheticDocQA_government_reports_test_beir"
    AI_DATASET = "vidore/syntheticDocQA_artificial_intelligence_test_beir"
    HEALTHCARE_DATASET = "vidore/syntheticDocQA_healthcare_industry_test_beir"
    ESG_DATASET = "vidore/esg_reports_v2"
    VIDORE_BIOMEDICAL_DATASET = "vidore/biomedical_lectures_v2"
    ECONOMICS_DATASET = "vidore/economics_reports_v2"
    CONSULTING_DATASET = "sherpa/consulting_dataset"
    CONSULTING_LIGHT_DATASET = "sherpa/consulting_light_dataset"
