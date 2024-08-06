import { useState } from "react";
import axios from "axios";
import { Box, TextField, Button } from "@material-ui/core";

const Chatbot = () => {
  const [inputMessage, setInputMessage] = useState("");
  const [chatMessages, setChatMessages] = useState([]);

  const handleInputChange = (event) => {
    setInputMessage(event.target.value);
  };

  const sendMessage = async () => {
    if (inputMessage.trim() !== "") {
      setChatMessages((prevMessages) => [
        ...prevMessages,
        { content: inputMessage, sender: "user" },
      ]);
      setInputMessage("");

      try {
        const response = await axios.post("/chatbot", { message: inputMessage });
        const botResponse = response.data.response;

        setChatMessages((prevMessages) => [
          ...prevMessages,
          { content: botResponse, sender: "bot" },
        ]);
      } catch (error) {
        console.error("Error:", error);
      }
    }
  };

  return (
    <Box>
      <Box>
        {chatMessages.map((message, index) => (
          <Box
            key={index}
            display="flex"
            justifyContent={message.sender === "user" ? "flex-end" : "flex-start"}
          >
            <Box
              bgcolor={message.sender === "user" ? "#f0f0f0" : "#d3d3d3"}
              borderRadius="10px"
              p={1}
              m={1}
              maxWidth="70%"
            >
              {message.content}
            </Box>
          </Box>
        ))}
      </Box>
      <Box display="flex" alignItems="center">
        <TextField
          variant="outlined"
          label="Type your message"
          value={inputMessage}
          onChange={handleInputChange}
        />
        <Button variant="contained" color="primary" onClick={sendMessage}>
          Send
        </Button>
      </Box>
    </Box>
  );
};

export default Chatbot;