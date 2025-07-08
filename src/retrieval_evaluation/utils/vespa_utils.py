import base64
import logging
import os
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image
from vespa.application import Vespa
from vespa.deployment import VespaCloud
from vespa.io import VespaResponse

# Vespa
from vespa.package import (
    HNSW,
    ApplicationPackage,
    Document,
    Field,
    FieldSet,
    FirstPhaseRanking,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)

load_dotenv()
logger = logging.getLogger(__name__)


def get_base64_image(image: Image.Image) -> str:
    """
    Convert PIL image to base64 string.

    Args:
    image: PIL Image object

    Returns:
    str: Base64 encoded string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return str(base64.b64encode(buffered.getvalue()), "utf-8")


def resize_image(image: Image.Image, max_dim: int = 2048) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.

    Args:
    image: PIL Image object
    max_dim: Maximum dimension (width or height)

    Returns:
    Image.Image: Resized image
    """
    img_width, img_height = image.size
    aspect_ratio = img_width / img_height

    if img_width > max_dim:
        new_width = max_dim
        new_height = int(new_width / aspect_ratio)
    else:
        new_width = img_width
        new_height = img_height

    if new_height > max_dim:
        new_height = max_dim
        new_width = int(new_height * aspect_ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)


def define_vespa_schema() -> Schema:
    """
    Define Vespa schema for PDF pages.

    Returns:
    Schema: Vespa schema
    """
    print("Defining Vespa schema")

    return Schema(
        name="beir_document",
        document=Document(
            fields=[
                Field(
                    name="id",
                    type="string",
                    indexing=["summary", "index"],
                    match=["word"],
                ),
                Field(
                    name="dataset_name", type="string", indexing=["summary", "index"]
                ),
                Field(name="corpus_id", type="int", indexing=["summary", "index"]),
                Field(name="doc_id", type="string", indexing=["summary", "index"]),
                Field(name="image", type="raw", indexing=["summary"]),
                Field(
                    name="embedding",
                    type="tensor<float>(patch{}, v[128])",
                    indexing=["attribute"],
                ),
                Field(
                    name="binary_embedding",
                    type="tensor<int8>(patch{}, v[16])",
                    indexing=["attribute", "index"],
                    ann=HNSW(
                        distance_metric="hamming",
                        max_links_per_node=32,
                        neighbors_to_explore_at_insert=400,
                    ),
                ),
            ]
        ),
        fieldsets=[
            FieldSet(name="default", fields=["dataset_name", "corpus_id", "doc_id"])
        ],
    )


def add_rank_profiles(schema: Schema) -> None:
    """
    Add rank profiles to the schema.

    Args:
    schema: Vespa schema to add rank profiles to
    """
    # Default rank profile
    default_profile = RankProfile(
        name="default",
        inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
        functions=[
            Function(
                name="bm25_score", expression="bm25(dataset_name) + bm25(corpus_id)"
            ),
        ],
        first_phase=FirstPhaseRanking(expression="bm25_score"),
    )
    schema.add_rank_profile(default_profile)

    # Define inputs for query tensors
    input_query_tensors = []
    MAX_QUERY_TERMS = 256
    for i in range(MAX_QUERY_TERMS):
        input_query_tensors.append((f"query(rq{i})", "tensor<float>(v[128])"))
    input_query_tensors.append(("query(qt)", "tensor<float>(querytoken{}, v[128])"))

    # Add retrieval-and-rerank profile
    retrieval_profile = RankProfile(
        name="retrieval-and-rerank",
        inputs=input_query_tensors,
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * attribute(embedding) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            )
        ],
        first_phase=FirstPhaseRanking(expression="max_sim"),
    )
    schema.add_rank_profile(retrieval_profile)

    # Define binary inputs
    binary_input_query_tensors = []
    for i in range(MAX_QUERY_TERMS):
        binary_input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))
    binary_input_query_tensors.append(
        ("query(qt)", "tensor<float>(querytoken{}, v[128])")
    )
    binary_input_query_tensors.append(
        ("query(qtb)", "tensor<int8>(querytoken{}, v[16])")
    )

    # Add binary retrieval profile
    binary_profile = RankProfile(
        name="retrieval-and-rerank-binary",
        inputs=binary_input_query_tensors,
        functions=[
            Function(
                name="max_sim",
                expression="""
                    sum(
                        reduce(
                            sum(
                                query(qt) * unpack_bits(attribute(binary_embedding)) , v
                            ),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
            Function(
                name="max_sim_binary",
                expression="""
                    sum(
                      reduce(
                        1/(1 + sum(
                            hamming(query(qtb), attribute(binary_embedding)) ,v)
                        ),
                        max,
                        patch
                      ),
                      querytoken
                    )
                """,
            ),
        ],
        first_phase=FirstPhaseRanking(expression="max_sim_binary"),
        second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
    )
    schema.add_rank_profile(binary_profile)

    # Add maxsim bruteforce profile
    bruteforce_profile = RankProfile(
        name="maxsim_bruteforce",
        inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
        functions=[
            Function(
                name="max_sim_bruteforce",
                expression="""
                    sum(
                        reduce(
                            sum(query(qt) * attribute(embedding), v),
                            max, patch
                        ),
                        querytoken
                    )
                """,
            ),
        ],
        first_phase=FirstPhaseRanking(expression="max_sim_bruteforce"),
    )
    schema.add_rank_profile(bruteforce_profile)


def create_vespa_application(
    vespa_app_name: str,
    schema: Schema,
) -> ApplicationPackage:
    """
    Create Vespa application package.

    Args:
    schema: Vespa schema

    Returns:
    ApplicationPackage: Vespa application package
    """
    return ApplicationPackage(name=vespa_app_name, schema=[schema])


def deploy_to_vespa_cloud(app_package: ApplicationPackage) -> Vespa:
    """
    Deploy application to Vespa Cloud.

    Args:
    app_package: Vespa application package

    Returns:
    Vespa: Vespa application instance
    """
    tenant_name = "sherpa-dev"

    # Vespa key/cert directory
    home = os.path.expanduser("~")
    path_to_api_key = f"{home}/.vespa/{tenant_name}.api-key.pem"

    vespa_cloud = VespaCloud(
        tenant=tenant_name,
        application=app_package.name,
        key_location=path_to_api_key,
        application_package=app_package,
    )

    print("Deploying to Vespa Cloud (may take a few minutes)...")
    return vespa_cloud.deploy()


def feed_data_to_vespa(app: Vespa, vespa_feed: list[dict]) -> None:
    """
    Feed data to Vespa application.

    Args:
    app: Vespa application
    vespa_feed: List of documents to feed
    """
    failed = []

    def callback(response: VespaResponse, id: str):
        if not response.is_successful():
            print(
                f"Failed to feed document {id} with status code {response.status_code}: Reason {response.get_json()}"
            )
            failed.append(id)

    print(f"Number of documents to feed: {len(vespa_feed)}")
    print(
        f"Feeding data into Vespa: Example document ID: {vespa_feed[0]['id'] if vespa_feed else 'None'}"
    )

    app.feed_async_iterable(
        vespa_feed,
        schema="beir_document",
        max_queue_size=100,
        max_workers=4,
        max_connections=1,
        callback=callback,
    )

    # Retry failed documents up to 3 times
    retries = 3
    for attempt in range(retries):
        if not failed:
            break
        print(f"Retrying {len(failed)} failed documents (attempt {attempt + 1})")
        to_retry = [doc for doc in vespa_feed if doc["id"] in failed]
        failed.clear()
        app.feed_async_iterable(
            to_retry,
            schema="beir_document",
            max_queue_size=50,
            max_workers=2,
            max_connections=1,
            callback=callback,
        )


def connect_existing_vespa(
    vespa_app_name: str,
    tenant_name: str = "sherpa-dev",
    instance_name: str = "default",
) -> Vespa:
    """
    Connect to existing Vespa application.

    Args:
    app_package: Vespa application package
    """
    vespa_app_url = os.getenv("VESPA_EVALS_APP_URL")

    # Vespa key/cert directory
    home = os.path.expanduser("~")
    app_dir = f"{home}/.vespa/{tenant_name}.{vespa_app_name}.{instance_name}"
    cert_path = f"{app_dir}/data-plane-public-cert.pem"
    key_path = f"{app_dir}/data-plane-private-key.pem"

    vespa_app = Vespa(
        url=vespa_app_url,
        cert=cert_path,
        key=key_path,
    )
    status_resp = vespa_app.get_application_status()
    if status_resp.status_code != 200:
        raise Exception(f"Failed to connect to Vespa at {vespa_app_url}")
    else:
        logger.info(f"Connected to Vespa at {vespa_app_url}")

    return vespa_app
