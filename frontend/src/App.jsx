import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Load conversations list on mount
  useEffect(() => {
    loadConversations();
  }, []);

  // Load full conversation when selection changes
  useEffect(() => {
    if (currentConversationId) {
      loadConversation(currentConversationId);
    }
  }, [currentConversationId]);

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      // Include the title from the server response (not just an empty object)
      setConversations([
        {
          id: newConv.id,
          created_at: newConv.created_at,
          title: newConv.title || 'New Conversation',
          message_count: 0,
        },
        ...conversations,
      ]);
      setCurrentConversationId(newConv.id);
      setCurrentConversation(newConv);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
  };

  const handleSendMessage = async (content) => {
    if (!currentConversationId) return;

    setIsLoading(true);

    // Optimistically add user message
    const userMessage = { role: 'user', content };
    setCurrentConversation((prev) => ({
      ...prev,
      messages: [...prev.messages, userMessage],
    }));

    // Add a partial assistant message that fills in progressively
    const assistantMessage = {
      role: 'assistant',
      stage1: null,
      stage2: null,
      stage2_5: undefined, // undefined = not yet received; null = received but empty
      stage3: null,
      metadata: null,
      loading: {
        stage1: false,
        stage2: false,
        stage2_5: false,
        stage3: false,
      },
    };

    setCurrentConversation((prev) => ({
      ...prev,
      messages: [...prev.messages, assistantMessage],
    }));

    try {
      await api.sendMessageStream(
        currentConversationId,
        content,
        (eventType, event) => {
          switch (eventType) {
            case 'stage1_start':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.loading.stage1 = true;
              }));
              break;

            case 'stage1_complete':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.stage1 = event.data;
                m.loading.stage1 = false;
              }));
              break;

            case 'stage2_start':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.loading.stage2 = true;
              }));
              break;

            case 'stage2_complete':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.stage2 = event.data;
                m.metadata = event.metadata;
                m.loading.stage2 = false;
              }));
              break;

            case 'stage2_5_start':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.loading.stage2_5 = true;
              }));
              break;

            case 'stage2_5_complete':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.stage2_5 = event.data; // may be null if debate was skipped
                m.loading.stage2_5 = false;
              }));
              break;

            case 'stage3_start':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.loading.stage3 = true;
                m.stage3 = { model: 'chairman', response: '' };
              }));
              break;

            case 'stage3_chunk':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.stage3 = {
                  ...m.stage3,
                  response: (m.stage3?.response || '') + event.content,
                };
              }));
              break;

            case 'stage3_complete':
              setCurrentConversation((prev) => updateLastMessage(prev, (m) => {
                m.stage3 = event.data;
                m.loading.stage3 = false;
              }));
              break;

            case 'title_complete':
              // Update the title in the sidebar list
              setConversations((prev) =>
                prev.map((c) =>
                  c.id === currentConversationId
                    ? { ...c, title: event.data.title }
                    : c
                )
              );
              break;

            case 'complete':
              // Refresh sidebar to get updated message count
              loadConversations();
              setIsLoading(false);
              break;

            case 'error':
              console.error('Stream error:', event.message);
              setIsLoading(false);
              break;

            default:
              console.log('Unknown event type:', eventType);
          }
        }
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      // Roll back optimistic messages on hard failure
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
      />
      <ChatInterface
        conversation={currentConversation}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />
    </div>
  );
}

/**
 * Immutably update the last message in the conversation via a mutator function.
 * The mutator receives a shallow copy of the message and can modify it directly.
 */
function updateLastMessage(prev, mutator) {
  const messages = [...prev.messages];
  const lastMsg = { ...messages[messages.length - 1] };
  lastMsg.loading = { ...lastMsg.loading };
  mutator(lastMsg);
  messages[messages.length - 1] = lastMsg;
  return { ...prev, messages };
}

export default App;
