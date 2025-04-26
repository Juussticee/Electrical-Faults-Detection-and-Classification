import socket
import threading
import struct

latest_result = ""  # Shared variable for Flask access
udp_thread_started = False  # Flag for thread start control

def get_latest_result():
    global latest_result
    print("got latest result")
    return latest_result

def launch_udp_server():
    global udp_thread_started
    if not udp_thread_started:
        print("going to launch the udp server")
        udp_thread = threading.Thread(target=start_udp_server, daemon=True)
        print("starting")
        udp_thread.start()
        print("udp launched")
        udp_thread_started = True

def start_udp_server(host="0.0.0.0", port=5005):
    from pipeline import real_time_prediction
    global latest_result
    print("connect socket")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    print(f"UDP server listening on {host}:{port}")

    while True:
        try:
            print("get the data")
            data, addr = sock.recvfrom(1024)
            print("decode the data")
            print("put it in an array")
            array = list(struct.unpack('<7d', data))  # 7 doubles, 8 bytes each
            print("Unpacked array:", array)

            # Process the data with your model
            print("send to the pipeline to get the result")
            result = real_time_prediction(array)

            # Update shared variable
            latest_result = result

            print(f"Received: {array} -> {result}")
            sock.sendto(b"OK", addr)  # Unblock MATLAB

        except Exception as e:
            print(f"Error: {e}")
