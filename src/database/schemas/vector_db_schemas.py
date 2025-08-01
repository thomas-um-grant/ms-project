from weaviate.classes.config import DataType, Property


class CollectionSchemas:
    @staticmethod
    def multimodal_image_schema():
        return {
            "properties": [
                Property(name="dataset_name", data_type=DataType.TEXT),
                Property(name="corpus_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="image_path", data_type=DataType.TEXT),
            ],
            # Use default vector configuration
        }

    @staticmethod
    def traditional_text_schema():
        return {
            "properties": [
                Property(name="dataset_name", data_type=DataType.TEXT),
                Property(name="corpus_id", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            # Use default vector configuration
        }
