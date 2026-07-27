import sys
import os
import grpc

project_root = os.path.dirname(os.path.abspath(__file__))
generated_path = os.path.join(project_root, 'src', 'generated')

if generated_path not in sys.path:
    sys.path.insert(0, generated_path)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.generated import spam_filter_pb2 as spam_pb2
from src.generated import spam_filter_pb2_grpc as spam_pb2_grpc


def test():
    channel = grpc.insecure_channel('localhost:5001')
    stub = spam_pb2_grpc.SpamFilterStub(channel)

    print("Connected to the grpc server.")

    request1 = spam_pb2.PredictRequest(text="WIN FREE IPHONE!!! CLICK HERE")
    response1 = stub.Predict(request1)
    print(f"text: WIN FREE IPHONE!!! CLICK HERE")
    print(f"answer: is_spam - {response1.is_spam}, confidence - {response1.confidence:.4f}")

    request2 = spam_pb2.PredictRequest(text="Hi, can we reschedule tomorrow's meeting?")
    response2 = stub.Predict(request2)
    print(f"text: Hi, can we reschedule tomorrow's meeting?")
    print(f"answer: is_spam - {response2.is_spam}, confidence - {response2.confidence:.4f}")


if __name__ == '__main__':
    test()