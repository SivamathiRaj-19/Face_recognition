import socket
import time
# --- Configuration ---
TARGET_IP = '192.168.1.200'
PORT = 502
CHANNELS = list(range(1, 9))  # 1 to 8


def build_modbus_write_coil(transaction_id, unit_id, coil_address, state):

    function_code = 0x05
    value = 0xFF00 if state == 1 else 0x0000
    protocol_id = 0x0000
    length = 6  
    packet = bytearray()
    packet += transaction_id.to_bytes(2, byteorder='big')
    packet += protocol_id.to_bytes(2, byteorder='big')
    packet += length.to_bytes(2, byteorder='big')
    packet += unit_id.to_bytes(1, byteorder='big')
    packet += function_code.to_bytes(1, byteorder='big')
    packet += coil_address.to_bytes(2, byteorder='big')
    packet += value.to_bytes(2, byteorder='big')

    return packet


def send_command(client, transaction_id, coil_address, state):
    packet = build_modbus_write_coil(
        transaction_id=transaction_id,
        unit_id=1,
        coil_address=coil_address,
        state=state
    )

    client.send(packet)
    response = client.recv(1024)
    return response


def control_relay(ch):
    if ch not in CHANNELS:
        print(f"Invalid channel {ch}!")
        return
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect((TARGET_IP, PORT))
            coil_address = ch - 1
            transaction_id = int(time.time()) % 65535
            
            # Pulse the relay: ON then OFF
            send_command(client, transaction_id, coil_address, 1)
            time.sleep(0.1)
            send_command(client, transaction_id + 1, coil_address, 0)
            print(f"Locker {ch} opened.")
    except Exception as e:
        print(f"Relay Error: {e}")
