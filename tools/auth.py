#import os
import boto3
#import gradio as gr
import hmac
import hashlib
import base64
from tools.config import AWS_CLIENT_ID, AWS_CLIENT_SECRET, AWS_USER_POOL_ID, AWS_REGION

def calculate_secret_hash(client_id:str, client_secret:str, username:str):
    message = username + client_id
    dig = hmac.new(
        str(client_secret).encode('utf-8'),
        msg=str(message).encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    secret_hash = base64.b64encode(dig).decode()
    return secret_hash

def authenticate_user(username:str, password:str, user_pool_id:str=AWS_USER_POOL_ID, client_id:str=AWS_CLIENT_ID, client_secret:str=AWS_CLIENT_SECRET):
    """Authenticates a user against an AWS Cognito user pool.

    Args:
        user_pool_id (str): The ID of the Cognito user pool.
        client_id (str): The ID of the Cognito user pool client.
        username (str): The username of the user.
        password (str): The password of the user.
        client_secret (str): The client secret of the app client

    Returns:
        bool: True if the user is authenticated, False otherwise.
    """

    client = boto3.client('cognito-idp', region_name=AWS_REGION)  # Cognito Identity Provider client

    # Compute the secret hash
    secret_hash = calculate_secret_hash(client_id, client_secret, username)

    try:

        if client_secret == '':
            response = client.initiate_auth(
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password,
                },
                ClientId=client_id
            )

        else:
            response = client.initiate_auth(
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password,
                'SECRET_HASH': secret_hash
            },
            ClientId=client_id
            )

        # If successful, you'll receive an AuthenticationResult in the response
        if response.get('AuthenticationResult'):
            return True
        else:
            return False

    except client.exceptions.NotAuthorizedException:
        return False
    except client.exceptions.UserNotFoundException:
        return False
    except Exception as e:
        out_message = f"An error occurred: {e}"
        print(out_message)
        raise Exception(out_message)
        return False 