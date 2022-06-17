import json
import zmq
from AbortException import AbortException
class zmq_com:
    def get_prev_parameters(socket):
        """Performs the get_prev_parameters that get initial or preveious data before starting the optimization"""

        # Request initial parameters
        print("Sending parameters request")
        socket.send(b"parameters")

        # Receive initial parameters
        reply = socket.recv()
        parameters = json.loads(reply)
        print("Experimental parameters received: {}".format(parameters))

        return parameters
    def send_optimized_parameters(socket,best_combo):
        print("Sending parameters: {}".format(best_combo))

        socket.send(json.dumps(best_combo.tolist()).encode('utf-8'))

        reply = socket.recv()
        print("Received reply: {}".format(reply))
        
        if (reply == b"abort"):
            raise AbortException()

        # Process reply
        function_value = float(reply)
        
        #return function_value
    
    def init_socket(binding):
        print("Binding socket at {} ...".format(binding))

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(binding)
        
        print("Binding complete!")

        return socket



