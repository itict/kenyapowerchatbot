document.addEventListener("DOMContentLoaded", function() {
    // Attach event listener to user input field
    document.getElementById("user-input").addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            sendMessage(); // Call sendMessage function on Enter key press
        }
    });
});

// Function to send user message to Flask server and handle bot response
function sendMessage() {
    const userInput = document.getElementById("user-input").value.trim();
    if (!userInput) return; // If input is empty, do nothing

    // Send POST request to Flask server's /api/chatbot endpoint
    fetch('http://127.0.0.1:5000/api/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput }) // Send user input as JSON payload
    })
    .then(handleResponse)
    .catch(handleError);

    // Display user's message in chat box
    displayMessage(userInput, true);

    // Clear user input field
    document.getElementById("user-input").value = '';

    // Scroll to bottom of chat box
    scrollToBottom();
}

// Function to handle JSON response from server
function handleResponse(response) {
    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json().then(data => {
        // Display bot's message from the server response
        displayMessage(data.response, false);
    });
}

// Function to handle errors during fetch request
function handleError(error) {
    console.error('Fetch Error:', error);
    displayMessage('Error occurred. Please try again.', false);
}

// Function to display messages in the chat box
function displayMessage(message, isUser) {
    const chatBox = document.getElementById("chat-box");

    // Create a new message element
    const messageElement = document.createElement("div");
    messageElement.className = `message ${isUser ? 'user' : 'bot'}`; // Apply user or bot styling
    messageElement.textContent = message; // Set message text

    chatBox.appendChild(messageElement); // Append message to chat box

    // Scroll to bottom of chat box
    scrollToBottom();
}

// Function to scroll to the bottom of chat box
function scrollToBottom() {
    const chatBox = document.getElementById("chat-box");
    chatBox.scrollTop = chatBox.scrollHeight;
}
