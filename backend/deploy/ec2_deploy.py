"""Deploy GeoChat-7B on an EC2 g5.xlarge GPU instance.

Usage:
    python -m backend.deploy.ec2_deploy --launch
    python -m backend.deploy.ec2_deploy --status
    python -m backend.deploy.ec2_deploy --terminate
"""

import argparse
import base64
import json
import os
import sys
import time

import boto3

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
INSTANCE_TYPE = "g5.xlarge"
AMI_ID = "ami-012ba162b9cd2729c"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)
KEY_NAME = "bhuvision-geochat-key"
SG_NAME = "bhuvision-geochat-sg"
INSTANCE_TAG = "BHUVISION-GeoChat-7B"


def get_or_create_security_group(ec2):
    """Create security group allowing port 8080 and SSH."""
    try:
        groups = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
        )["SecurityGroups"]
        if groups:
            sg_id = groups[0]["GroupId"]
            print(f"Using existing security group: {sg_id}")
            return sg_id
    except Exception:
        pass

    # Get default VPC
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
    vpc_id = vpcs[0]["VpcId"] if vpcs else ec2.describe_vpcs()["Vpcs"][0]["VpcId"]

    sg = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="BHUVISION GeoChat-7B inference server",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]

    # Allow SSH (22) and API (8080) from anywhere
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            {"IpProtocol": "tcp", "FromPort": 8080, "ToPort": 8080, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        ],
    )
    print(f"Created security group: {sg_id} (ports 22, 8080 open)")
    return sg_id


def get_or_create_key_pair(ec2):
    """Create key pair if it doesn't exist."""
    try:
        ec2.describe_key_pairs(KeyNames=[KEY_NAME])
        print(f"Using existing key pair: {KEY_NAME}")
        return KEY_NAME
    except Exception:
        kp = ec2.create_key_pair(KeyName=KEY_NAME)
        # Save private key
        key_path = os.path.join(os.path.dirname(__file__), f"{KEY_NAME}.pem")
        with open(key_path, "w") as f:
            f.write(kp["KeyMaterial"])
        print(f"Created key pair: {KEY_NAME} (saved to {key_path})")
        return KEY_NAME


def get_user_data():
    """Load and configure the user data script."""
    script_dir = os.path.dirname(__file__)
    script_path = os.path.join(script_dir, "ec2_userdata.sh")

    with open(script_path, "r") as f:
        script = f.read()

    # Inject HF token
    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    script = script.replace("__HF_TOKEN_PLACEHOLDER__", hf_token)

    return base64.b64encode(script.encode()).decode()


def find_instance(ec2):
    """Find running BHUVISION GeoChat instance."""
    reservations = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [INSTANCE_TAG]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ]
    )["Reservations"]
    for r in reservations:
        for inst in r["Instances"]:
            return inst
    return None


def launch(ec2):
    """Launch the EC2 GPU instance."""
    existing = find_instance(ec2)
    if existing:
        state = existing["State"]["Name"]
        iid = existing["InstanceId"]
        if state == "running":
            ip = existing.get("PublicIpAddress", "pending")
            print(f"Instance already running: {iid} at {ip}")
            print(f"API endpoint: http://{ip}:8080")
            return
        elif state == "stopped":
            print(f"Starting stopped instance: {iid}")
            ec2.start_instances(InstanceIds=[iid])
            print("Waiting for instance to start...")
            waiter = ec2.get_waiter("instance_running")
            waiter.wait(InstanceIds=[iid])
            inst = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
            print(f"Instance running at: {inst.get('PublicIpAddress')}")
            return

    sg_id = get_or_create_security_group(ec2)
    key_name = get_or_create_key_pair(ec2)
    user_data = get_user_data()

    print(f"Launching {INSTANCE_TYPE} with Deep Learning AMI...")
    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=key_name,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": 100,  # 100 GB for model weights
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": INSTANCE_TAG},
                    {"Key": "Project", "Value": "BHUVISION"},
                ],
            }
        ],
    )
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"Instance launched: {instance_id}")
    print("Waiting for instance to be running...")

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    inst = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    public_ip = inst.get("PublicIpAddress", "No public IP")
    print(f"\n{'='*50}")
    print(f"Instance running: {instance_id}")
    print(f"Public IP: {public_ip}")
    print(f"API endpoint: http://{public_ip}:8080")
    print(f"Health check: http://{public_ip}:8080/health")
    print(f"{'='*50}")
    print(f"\nThe model is downloading (~14GB). Server will be ready in ~10-15 min.")
    print(f"Monitor setup: ssh -i {KEY_NAME}.pem ubuntu@{public_ip} 'tail -f /var/log/geochat-setup.log'")


def status(ec2):
    """Check instance and server status."""
    inst = find_instance(ec2)
    if not inst:
        print("No BHUVISION GeoChat instance found.")
        return

    state = inst["State"]["Name"]
    iid = inst["InstanceId"]
    ip = inst.get("PublicIpAddress", "N/A")

    print(f"Instance: {iid}")
    print(f"State: {state}")
    print(f"Public IP: {ip}")
    print(f"Instance Type: {inst.get('InstanceType', 'N/A')}")

    if state == "running" and ip != "N/A":
        print(f"\nAPI endpoint: http://{ip}:8080")
        print(f"Health check: http://{ip}:8080/health")

        # Try health check
        try:
            import urllib.request
            req = urllib.request.Request(f"http://{ip}:8080/health", method="GET")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                print(f"Server status: {data.get('status', 'unknown')}")
                print(f"GPU available: {data.get('gpu_available', 'unknown')}")
        except Exception as e:
            print(f"Server not ready yet: {e}")


def terminate(ec2):
    """Terminate the instance to stop billing."""
    inst = find_instance(ec2)
    if not inst:
        print("No BHUVISION GeoChat instance found.")
        return

    iid = inst["InstanceId"]
    state = inst["State"]["Name"]
    if state in ("terminated", "shutting-down"):
        print(f"Instance {iid} already terminated.")
        return

    ec2.terminate_instances(InstanceIds=[iid])
    print(f"Terminating instance: {iid}")
    print("Instance will be fully terminated in ~1 minute.")
    print("Billing will stop once terminated.")


def main():
    parser = argparse.ArgumentParser(description="Deploy GeoChat-7B on EC2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true", help="Launch the GPU instance")
    group.add_argument("--status", action="store_true", help="Check instance status")
    group.add_argument("--terminate", action="store_true", help="Terminate instance (stop billing)")
    group.add_argument("--stop", action="store_true", help="Stop instance (keeps data)")

    args = parser.parse_args()
    ec2 = boto3.client("ec2", region_name=REGION)

    if args.launch:
        launch(ec2)
    elif args.status:
        status(ec2)
    elif args.terminate:
        terminate(ec2)
    elif args.stop:
        inst = find_instance(ec2)
        if inst:
            ec2.stop_instances(InstanceIds=[inst["InstanceId"]])
            print(f"Stopping instance: {inst['InstanceId']} (data preserved)")
        else:
            print("No instance found.")


if __name__ == "__main__":
    main()
