import grpc
import sys
import os

from concurrent import futures
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.generated import spam_filter_pb2 as spam_pb2
from src.generated import spam_filter_pb2_grpc as spam_pb2_grpc

from src.predict import SpamFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpamServicer(spam_pb2_grpc.SpamFilterServicer):
    def __init__(self):
        logger.info("Loading SpamFilter model...")
        self.filter = SpamFilter()
        logger.info("The model is successfully loaded.")

    def Predict(self, request, context):
        try:
            text = request.text
            logger.info(f"Received a text-checking request: {text[:50]}")

            result = self.filter.predict(text)

            return spam_pb2.PredictReply(
                is_spam = result['is_spam'],
                confidence = result['confidence']
            )
        except Exception as e:
            logger.error(e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return spam_pb2.PredictReply()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    spam_pb2_grpc.add_SpamFilterServicer_to_server(
        SpamServicer(),
        server
    )

    server.add_insecure_port('[::]:5001')

    server.start()

    logger.info(f"gRPC server started on port 5001")
    logger.info("Waiting for requests...")

    server.wait_for_termination()


if __name__ == '__main__':
    serve()