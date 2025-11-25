from pipelines.InstaDeepMHCIPresentation.pipeline import get_pipeline
import boto3
import os
import sys

# Add the parent directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_pipeline_execution():
    """Test that the SageMaker pipeline executes successfully."""
    region = boto3.Session().region_name
    role = "arn:aws:iam::975628797022:role/service-role/AmazonSageMaker-ExecutionRole-20250826T095642"

    model_package_group_name = "InstaDeepMHCIPresentationModelPackageGroup"
    pipeline_name = "InstaDeepMHCIPresentationPipeline"

    pipeline = get_pipeline(
        region=region,
        role=role,
        model_package_group_name=model_package_group_name,
        pipeline_name=pipeline_name,
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    execution.wait(delay=60, max_attempts=200)
    status = execution.describe()
    print(status)

    assert status["PipelineExecutionStatus"] == "Succeeded", (
        f"Pipeline failed with status: {status['PipelineExecutionStatus']}"
    )
    print("Success")


# Optional: Keep the script functionality when running directly
if __name__ == "__main__":
    test_pipeline_execution()
