# RabbitMQ

This repository contains code for a Spring Boot application that demonstrates how to use RabbitMQ for message queuing.

## RabbitMQConfig.java
This file contains the configuration for RabbitMQ. It defines the queue, exchange, and routing key using values from the application's properties file. It also sets up the binding between the queue and the exchange using the routing key.

## RabbitMQProducer.java
This service is responsible for sending messages to RabbitMQ. It uses the RabbitTemplate to send messages to the exchange with the specified routing key.

## RabbitMQConsumer.java
This service listens for messages on the RabbitMQ queue. When a message arrives, it logs the message to the console.

## MessageController.java
This controller exposes an endpoint that allows you to send messages to RabbitMQ. It uses the RabbitMQProducer to send the messages.

## SpringbootRabbitmqTutorialApplication.java
This is the entry point of the Spring Boot application. It runs the application and sets up the Spring context.

## docker-compose.yml
This file is used to set up a Docker container for RabbitMQ. It specifies the image to use, the ports to expose, and any environment variables needed by RabbitMQ.

## Running RabbitMQ
To run the application, you need to start RabbitMQ (using Docker or a local installation), then run the Spring Boot application. You can then send messages to RabbitMQ by making a GET request to `http://localhost:8080/api/v1/publish?message=hello`. The message will be logged to the console by the RabbitMQConsumer.
