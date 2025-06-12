<template>
  <div class="flex h-screen bg-gray-100 text-gray-800">
    <div class="w-64 bg-gray-900 text-white flex flex-col">
      <div class="p-4 border-b border-gray-700">
        <h1 class="text-xl font-bold">Chat Sessions</h1>
      </div>
      <button @click="addSession" class="m-4 p-2 bg-blue-600 hover:bg-blue-700 rounded">
        New Chat
      </button>
      <nav class="flex-1 p-4 space-y-2">
        <a v-for="session in sessions" :key="session.id" @click.prevent="selectSession(session.id)" href="#" class="block p-2 rounded" :class="{'bg-gray-700': session.id === activeSessionId, 'hover:bg-gray-800': session.id !== activeSessionId}">
          {{ session.name }}
        </a>
      </nav>
    </div>
    <div class="flex-1 flex flex-col">
      <div class="flex-1 p-6 overflow-y-auto">
        <div v-if="activeSession" class="space-y-4">
          <div v-for="message in activeSession.messages" :key="message.id" class="flex" :class="{ 'justify-end': message.sender === 'user' }">
            <div class="max-w-xs lg:max-w-md p-3 rounded-lg" :class="{'bg-blue-500 text-white': message.sender === 'user', 'bg-gray-300 text-gray-900': message.sender === 'bot'}">
              <p>{{ message.text }}</p>
            </div>
          </div>
        </div>
      </div>

      <div class="p-4 bg-white border-t border-gray-200">
        <div class="flex items-center">
          <input v-model="newMessage" @keyup.enter="sendMessage" type="text" placeholder="Type your message..." class="flex-1 p-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"/>
          <button @click="sendMessage" class="px-4 py-2 bg-blue-500 text-white rounded-r-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500">
            Send
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, computed } from 'vue'

type Message = {
  id: number
  text: string
  sender: 'user' | 'bot'
}

type Session = {
  id: number
  name: string
  messages: Message[]
}

const sessions = ref<Session[]>([
  {
    id: 1,
    name: 'Welcome',
    messages: [{ id: 1, text: 'How can I help you today?', sender: 'bot' }]
  },
])

const activeSessionId = ref<number>(1)
const newMessage = ref('')
let nextSessionId = 2

const activeSession = computed(() => {
  return sessions.value.find((session) => session.id === activeSessionId.value)
})

function addSession() {
  const newSession: Session = {
    id: nextSessionId++,
    name: `Convesation ${nextSessionId - 1}`,
    messages: [{ id: 1, text: 'How can I help you today?', sender: 'bot' }]
  }
  sessions.value.push(newSession)
  activeSessionId.value = newSession.id
}

function selectSession(id: number) {
  activeSessionId.value = id
}

async function sendMessage() {
  if (newMessage.value.trim() === '' || !activeSession.value) return
  const currentNewMessage = newMessage.value
  newMessage.value = ''
  
  const userMessage: Message = {
    id: Date.now(),
    text: currentNewMessage,
    sender: 'user',
  }
  activeSession.value.messages.push(userMessage)

  const chatbotResponse: Message = {
    id: Date.now() + 1,
    text: '',
    sender: 'bot',
  }
  activeSession.value.messages.push(chatbotResponse)
  const chatbotMessageRef = activeSession.value.messages[activeSession.value.messages.length - 1];

  try {
    const response = await fetch('http://104.5.62.23:3002/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama3.2-1b-instruct-fp32',
        message: currentNewMessage,
        stream: true,
      }),
    })
    if (!response.body) return

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (line.startsWith('data:')) {
          let data = line.substring(5);
          if (data.startsWith(' ') && data.length > 1) {
              data = data.substring(1);
          }
          if (data === '[DONE]') {
              return
          }
          chatbotMessageRef.text += data;
        }
      }
    }
  } catch (error) {
    console.error(error)
    chatbotMessageRef.text = 'Error: Could not connect to the server.'
  }
}
</script>

<style>
::-webkit-scrollbar {
  width: 8px;
}
::-webkit-scrollbar-track {
  background: #f1f1f1;
}
::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: #555;
}
</style>
