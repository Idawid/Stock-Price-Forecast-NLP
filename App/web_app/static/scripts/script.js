const socketBase = io('http://' + document.domain + ':' + location.port + '/new');

socketBase.on("connect", () => {
  console.log('Client connected:' + socketBase.id);
});

socketBase.on("disconnect", () => {
  console.log('Client disconnected:' + socketBase.id); // undefined on disconnect
});

function createMessageHeader(systemId, conversationId, messageId, responseMessageId, data) {
  const header = `${systemId}${conversationId}${String(messageId).padStart(3, '0')}${String(responseMessageId).padStart(3, '0')}`;
  return header;
}

function sendMessage(systemId, conversationId, messageId, responseMessageId, data) {
  const message = {
    header: createMessageHeader(systemId, conversationId, messageId, responseMessageId),
    data: data
  };

  socketBase.emit('frontend_message', message);
}

socketBase.on('backend_message', function (receivedMessage) {
  handleReceivedMessage(receivedMessage);
});

function handleReceivedMessage(receivedMessage) {
  console.log("Received Message:", receivedMessage);

  const {header, data} = receivedMessage;

  const systemId = header.charAt(0);
  const conversationId = header.charAt(1);
  const messageId = parseInt(header.substring(2, 5), 10);
  const responseMessageId = parseInt(header.substring(5, 8), 10);

  processResponse(systemId, conversationId, messageId, responseMessageId, data);
}

function processResponse(systemId, conversationId, messageId, responseMessageId, data) {
  console.log(`Processing message: System ID: ${systemId}, Conversation ID: ${conversationId}, Message ID: ${messageId}, Response Message ID: ${responseMessageId}, Data: ${data}`);
}

sendMessage('A', 'B', 123, 124, "Hello from Frontend");
