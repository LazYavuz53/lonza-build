"""SageMaker pipeline that runs the clinical biomarker preprocessing script."""
import os

import boto3
import sagemaker
import sagemaker.session
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """Gets the sagemaker client."""
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region."""
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region."""
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    pipeline_name="ClinicalBiomarkerPipeline",
    base_job_prefix="ClinicalBiomarker",
    processing_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker pipeline instance that runs the biomarker analysis preprocessing script."""
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    input_data = ParameterString(
        name="InputDataUrl",
        default_value="s3://instadeep53/datasets/clinical.csv",
    )

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-biomarker-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="analysis", source="/opt/ml/processing/output"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--csv-path", input_data, "--plots-dir", "/opt/ml/processing/output"],
    )
    step_process = ProcessingStep(
        name="PreprocessClinicalData",
        step_args=step_args,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[processing_instance_count, input_data],
        steps=[step_process],
        sagemaker_session=pipeline_session,
    )
    return pipeline
