import { apiClient } from "./api"

export interface ChatSummary {
  id: string
  title: string | null
  created_at: string
  updated_at: string
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  created_at: string
}

export interface ChatDetail {
  id: string
  title: string | null
  created_at: string
  updated_at: string
  messages: ChatMessage[]
}

export async function renameChat(
  orgId: string,
  chatId: string,
  title: string,
): Promise<ChatSummary> {
  const res = await apiClient.patch<ChatSummary>(
    `/api/v1/orgs/${orgId}/chats/${chatId}`,
    { title },
  )
  return res.data
}

export async function deleteChat(
  orgId: string,
  chatId: string,
): Promise<void> {
  await apiClient.delete(`/api/v1/orgs/${orgId}/chats/${chatId}`)
}

export async function listChats(orgId: string): Promise<ChatSummary[]> {
  const res = await apiClient.get<{ chats: ChatSummary[] }>(
    `/api/v1/orgs/${orgId}/chats`,
  )
  return res.data.chats
}

export async function getChat(
  orgId: string,
  chatId: string,
): Promise<ChatDetail> {
  const res = await apiClient.get<ChatDetail>(
    `/api/v1/orgs/${orgId}/chats/${chatId}`,
  )
  return res.data
}

export interface StreamEvent {
  event: string
  data: Record<string, unknown>
}

export function streamMessage(
  orgId: string,
  message: string,
  chatId: string | null,
  onEvent: (event: StreamEvent) => void,
  onError: (message: string) => void,
  onDone: () => void,
): AbortController {
  const controller = new AbortController()
  const baseUrl = apiClient.defaults.baseURL?.replace(/\/$/, "") ?? ""
  const token = localStorage.getItem("auth_token")

  fetch(`${baseUrl}/api/v1/orgs/${orgId}/chats/stream`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message, chat_id: chatId }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const text = await response.text()
        onError(`HTTP ${response.status}: ${text}`)
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        onError("No response body")
        return
      }

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        let eventType = "message"
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              onEvent({ event: eventType, data })
            } catch {
              // skip malformed
            }
            eventType = "message"
          }
        }
      }
      onDone()
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onError(err.message || "Connection failed")
      }
    })

  return controller
}