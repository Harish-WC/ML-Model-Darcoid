import boto3

# Replace these values with your own
endpoint_name = 'delm1ep4'
region_name = 'us-west-2'

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name=region_name)

# Sample input data (replace with your actual input data)
input_data = '{"data": "your_input_data"}'

try:
    # Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=input_data
    )

    # Check the HTTP response status
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        # Get the prediction result
        result = response['Body'].read()

        # Process or print the result based on your model's output format
        print("Prediction Result:", result.decode('utf-8'))
    else:
        print(f"Endpoint invocation failed with status code: {response['ResponseMetadata']['HTTPStatusCode']}")

except Exception as e:
    print(f"An error occurred: {e}")
