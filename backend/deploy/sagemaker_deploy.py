"""AWS SageMaker deployment script for MBZUAI/geochat-7B.

This script automates:
1. Setting up SageMaker execution role (already created: BHUVISION-SageMakerExecutionRole)
2. Deploying MBZUAI/geochat-7B to an ml.g5.xlarge GPU real-time endpoint
3. Testing the endpoint with sample satellite image input
4. Deleting the endpoint to stop billing

Usage:
  python -m backend.deploy.sagemaker_deploy --deploy
  python -m backend.deploy.sagemaker_deploy --status
  python -m backend.deploy.sagemaker_deploy --test
  python -m backend.deploy.sagemaker_deploy --delete
"""

import argparse
import base64
import json
import os
import sys
import time

import boto3

DEFAULT_ENDPOINT_NAME = "bhuvision-geochat-7b"
DEFAULT_INSTANCE_TYPE = "ml.g5.xlarge"
DEFAULT_ROLE_NAME = "BHUVISION-SageMakerExecutionRole"
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def get_role_arn(iam_client) -> str:
    """Get or create the SageMaker execution role ARN."""
    try:
        role = iam_client.get_role(RoleName=DEFAULT_ROLE_NAME)
        return role["Role"]["Arn"]
    except Exception:
        print(f"Creating IAM role {DEFAULT_ROLE_NAME}...")
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        res = iam_client.create_role(
            RoleName=DEFAULT_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
        )
        iam_client.attach_role_policy(
            RoleName=DEFAULT_ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        )
        time.sleep(10)  # Wait for IAM role propagation
        return res["Role"]["Arn"]


def deploy_endpoint(instance_type: str = DEFAULT_INSTANCE_TYPE, endpoint_name: str = DEFAULT_ENDPOINT_NAME):
    """Deploy MBZUAI/geochat-7B as a SageMaker real-time endpoint."""
    import sagemaker
    from sagemaker.huggingface import HuggingFaceModel

    print(f"Deploying GeoChat-7B to SageMaker in {REGION} on {instance_type}...")
    boto_session = boto3.Session(region_name=REGION)
    iam = boto_session.client("iam")
    role_arn = get_role_arn(iam)
    print(f"Using execution role: {role_arn}")

    sess = sagemaker.Session(boto_session=boto_session)

    hub_config = {
        "HF_MODEL_ID": "MBZUAI/geochat-7B",
        "SM_NUM_GPUS": "1",
        "HF_TASK": "text-generation",
    }
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        hub_config["HUGGING_FACE_HUB_TOKEN"] = hf_token

    huggingface_model = HuggingFaceModel(
        env=hub_config,
        role=role_arn,
        transformers_version="4.37.0",
        pytorch_version="2.1.0",
        py_version="py310",
        sagemaker_session=sess,
        name=f"geochat-model-{int(time.time())}",
    )

    print(f"Creating endpoint '{endpoint_name}' (this takes 5-10 minutes)...")
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=900,
    )
    print(f"\nSUCCESS! GeoChat-7B endpoint is active: {predictor.endpoint_name}")
    print("\nTo integrate with BHUVISION, set these environment variables in your .env:")
    print(f"BHUVISION_GEOCHAT_BACKEND=sagemaker")
    print(f"BHUVISION_GEOCHAT_ENDPOINT_NAME={endpoint_name}")
    print(f"AWS_DEFAULT_REGION={REGION}")


def check_status(endpoint_name: str = DEFAULT_ENDPOINT_NAME):
    """Check the status of the SageMaker endpoint."""
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        print(f"Endpoint '{endpoint_name}': {status}")
        if "FailureReason" in desc:
            print(f"Failure Reason: {desc['FailureReason']}")
        return status
    except sm.exceptions.ClientError as exc:
        print(f"Endpoint '{endpoint_name}' not found: {exc}")
        return None


def test_endpoint(endpoint_name: str = DEFAULT_ENDPOINT_NAME):
    """Send a test query to the endpoint."""
    from PIL import Image
    import io

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    # Create small test image
    img = Image.new("RGB", (128, 128), color=(70, 130, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "inputs": "Describe the features visible in this satellite image.",
        "image": b64_img,
        "parameters": {"max_new_tokens": 64},
    }

    print(f"Invoking endpoint '{endpoint_name}'...")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = response["Body"].read().decode("utf-8")
    print(f"Response: {result}")


def delete_endpoint(endpoint_name: str = DEFAULT_ENDPOINT_NAME):
    """Delete endpoint and endpoint configuration to stop AWS billing."""
    sm = boto3.client("sagemaker", region_name=REGION)
    print(f"Deleting endpoint '{endpoint_name}'...")
    try:
        sm.delete_endpoint(EndpointName=endpoint_name)
        print("Endpoint deleted.")
    except Exception as exc:
        print(f"Could not delete endpoint: {exc}")

    try:
        sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print("Endpoint configuration deleted.")
    except Exception:
        pass
    print("Billing stopped for this endpoint.")


def main():
    parser = argparse.ArgumentParser(description="Manage SageMaker GeoChat-7B endpoint.")
    parser.add_argument("--deploy", action="store_true", help="Deploy the endpoint")
    parser.add_argument("--status", action="store_true", help="Check endpoint status")
    parser.add_argument("--test", action="store_true", help="Test the endpoint")
    parser.add_argument("--delete", action="store_true", help="Delete endpoint to stop billing")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="EC2 instance type")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME, help="Endpoint name")

    args = parser.parse_args()

    if args.deploy:
        deploy_endpoint(args.instance_type, args.endpoint_name)
    elif args.status:
        check_status(args.endpoint_name)
    elif args.test:
        test_endpoint(args.endpoint_name)
    elif args.delete:
        delete_endpoint(args.endpoint_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
