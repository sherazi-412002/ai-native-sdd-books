import React, { useState, useRef, useEffect } from 'react';
import styles from './ChatWidget.module.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const ChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your AI assistant for Physical AI & Humanoid Robotics. How can I help you today?',
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleClose = () => {
    setIsOpen(false);
  };

  const handleSend = () => {
    if (inputValue.trim() === '') return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    // Simulate bot response after a delay
    setTimeout(() => {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: getBotResponse(inputValue),
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botMessage]);
    }, 1000);
  };

  const getBotResponse = (userInput: string): string => {
    const input = userInput.toLowerCase();

    if (input.includes('hello') || input.includes('hi') || input.includes('hey')) {
      return 'Hello there! Welcome to the Physical AI & Humanoid Robotics learning platform. How can I assist you with your robotics journey?';
    } else if (input.includes('module') || input.includes('course')) {
      return 'We have 4 comprehensive modules: 1) The Robotic Nervous System (ROS 2), 2) The Digital Twin (Gazebo & Unity), 3) The AI-Robot Brain (NVIDIA Isaac), and 4) Vision-Language-Action. Which module interests you most?';
    } else if (input.includes('ros') || input.includes('nervous')) {
      return 'Module 1 covers ROS 2 fundamentals - the nervous system of your humanoid robot. You\'ll learn about nodes, topics, services, and actions that enable communication between robot components.';
    } else if (input.includes('simulation') || input.includes('gazebo') || input.includes('unity') || input.includes('digital twin')) {
      return 'Module 2 focuses on creating digital twins using Gazebo and Unity. These simulations are crucial for testing your robot behaviors in a safe, controlled environment before real-world deployment.';
    } else if (input.includes('ai') || input.includes('brain') || input.includes('isaac')) {
      return 'Module 3 teaches you to implement perception and planning with NVIDIA Isaac. You\'ll learn how to give your robot the ability to understand its environment and make intelligent decisions.';
    } else if (input.includes('vision') || input.includes('language') || input.includes('action') || input.includes('vla')) {
      return 'Module 4 integrates multimodal AI with real-world robotics. You\'ll learn how to combine visual, language, and action capabilities for sophisticated robot behaviors.';
    } else if (input.includes('thank')) {
      return 'You\'re welcome! Is there anything else I can help you with regarding Physical AI & Humanoid Robotics?';
    } else if (input.includes('bye') || input.includes('goodbye')) {
      return 'Goodbye! Feel free to return if you have more questions about Physical AI & Humanoid Robotics.';
    } else {
      return 'That\'s an interesting question about Physical AI & Humanoid Robotics! Our curriculum covers ROS 2, simulation environments, AI perception, and multimodal robotics. Would you like to know more about any specific module?';
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className={styles.chatWidgetContainer}>
      {isOpen ? (
        <div className={styles.chatWidgetExpanded}>
          <div className={styles.chatHeader}>
            <h3 className={styles.chatTitle}>Physical AI Assistant</h3>
            <button className={styles.chatClose} onClick={handleClose}>
              ×
            </button>
          </div>
          <div className={styles.chatMessages}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`${styles.message} ${
                  message.sender === 'user' ? styles.messageUser : styles.messageBot
                }`}
              >
                {message.text}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className={styles.chatInputContainer}>
            <input
              type="text"
              className={styles.chatInput}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about Physical AI & Robotics..."
            />
            <button className={styles.chatSendButton} onClick={handleSend}>
              Send
            </button>
          </div>
        </div>
      ) : (
        <button className={styles.chatWidgetCollapsed} onClick={toggleChat}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 13.54 2.36 15.01 3.02 16.32L2 22L7.68 20.98C8.99 21.64 10.46 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM8 16L10 14L12 16L16 12L14 10L12 12L8 8L6 10L10 14L8 16Z" fill="white"/>
          </svg>
        </button>
      )}
    </div>
  );
};

export default ChatWidget;