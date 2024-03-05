import threading
from multiprocessing import Process
from web_app.web_app import WebApplication, WebApplicationSocketIO
from backend.be_rabbitmq_connector import MLBackendRabbitMQConnector


def start_ml_backend_processor():
    processor = MLBackendRabbitMQConnector()
    processor.start_processing()


def start_web_application():
    web_application = WebApplication()
    socket_io = WebApplicationSocketIO(web_application.app, web_application.socket_io, web_application.backend_server)
    web_application.run()


if __name__ == '__main__':
    # Start the web application in its own thread
    web_app_thread = threading.Thread(target=start_web_application)
    web_app_thread.start()

    num_processors = 5
    processors = []
    for _ in range(num_processors):
        processor = Process(target=start_ml_backend_processor)
        processor.start()
        processors.append(processor)

    # Optionally join the processors if you need to wait for them to complete
    for processor in processors:
        processor.join()

    web_app_thread.join()
