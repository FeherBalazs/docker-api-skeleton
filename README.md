## Skeleton for building and shipping restful APIs with Docker

### Build docker image

	docker build --force-rm=true -t neuralmachines .

### Run docker image

	docker run -p 5001:5000 -d neuralmachines

The service should be now running at: localhost:5000/hello