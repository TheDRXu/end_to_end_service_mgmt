aws configure set aws_access_key_id ""
aws configure set aws_secret_access_key ""
aws configure set aws_session_token ""

aws s3 cp ./front-end/index.html s3://end-to-end-service-mgmt/index.html --cache-control "no-cache" --content-type "text/html"