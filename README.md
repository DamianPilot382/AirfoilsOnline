# airfoils-online

Upload and get the party started

Test:
```bash
sam build --use-container
sam local start-api
```

Deploy:
```bash
sam build --use-container

sam deploy --no-confirm-changeset --no-fail-on-empty-changeset --stack-name airfoils-online --s3-bucket airfoils-online --capabilities CAPABILITY_IAM --region us-west-1 --resolve-image-repos
```


