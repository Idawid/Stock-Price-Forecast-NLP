def decode_header(header):
    system_id = header[0]
    conversation_id = header[1]
    message_id = int(header[2:5])
    response_message_id = int(header[5:8])
    return system_id, conversation_id, message_id, response_message_id


def process_message(system_id, conversation_id, message_id, response_message_id, data):
    print(f"Processing message: System ID: {system_id}, Conversation ID: {conversation_id}, Message ID: {message_id}, Response Message ID: {response_message_id}, Data: {data}")
