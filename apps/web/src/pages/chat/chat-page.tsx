import { useCallback, useEffect, useRef, useState } from "react"
import { useParams, useNavigate } from "react-router-dom"
import { BrainCircuit } from "lucide-react"
import toast from "react-hot-toast"

import { AppSidebar } from "@/components/app-sidebar"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"
import { PageTransition } from "@/components/page-transition"
import { useAuth } from "@/contexts/auth-context"

import { ChatMessage } from "@/components/chat/chat-message"
import { ChatStatus } from "@/components/chat/chat-status"
import { ChatInput } from "@/components/chat/chat-input"
import {
  getChat,
  streamMessage,
  type ChatMessage as ChatMessageType,
} from "@/lib/chat-api"

interface DisplayMessage {
  id: string
  role: "user" | "assistant"
  content: string
}

export function ChatPage() {
  const { orgId, chatId } = useParams()
  const navigate = useNavigate()
  const { user } = useAuth()

  const orgName =
    user?.orgs.find((o) => o.id === orgId)?.name ?? "Organization"

  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string | null>(null)
  const [chatTitle, setChatTitle] = useState<string | null>(null)
  const [loadingChat, setLoadingChat] = useState(false)

  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const activeChatIdRef = useRef<string | null>(chatId ?? null)
  const isStreamingRef = useRef(false)
  const assistantMsgIdRef = useRef<string | null>(null)

  // Keep refs in sync
  useEffect(() => {
    activeChatIdRef.current = chatId ?? null
  }, [chatId])
  useEffect(() => {
    isStreamingRef.current = isStreaming
  }, [isStreaming])

  // Load existing chat messages when chatId changes — skip if we're streaming
  useEffect(() => {
    if (isStreamingRef.current) return
    if (!chatId || !orgId) {
      setMessages([])
      setChatTitle(null)
      return
    }

    let cancelled = false
    setLoadingChat(true)

    getChat(orgId, chatId)
      .then((data) => {
        if (cancelled) return
        setChatTitle(data.title)
        setMessages(
          data.messages.map((m: ChatMessageType) => ({
            id: m.id,
            role: m.role,
            content: m.content,
          })),
        )
      })
      .catch(() => {
        if (!cancelled) toast.error("Failed to load chat")
      })
      .finally(() => {
        if (!cancelled) setLoadingChat(false)
      })

    return () => {
      cancelled = true
    }
  }, [chatId, orgId])

  // Auto-scroll to bottom
  useEffect(() => {
    const el = scrollRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages, statusMessage])

  const handleSend = useCallback(() => {
    if (!input.trim() || !orgId || isStreaming) return

    const userMessage = input.trim()
    setInput("")

    // Optimistically add user message
    const tempId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      { id: tempId, role: "user", content: userMessage },
    ])

    setIsStreaming(true)
    setStatusMessage("Thinking...")
    assistantMsgIdRef.current = null

    const controller = streamMessage(
      orgId,
      userMessage,
      activeChatIdRef.current,
      // onEvent
      (event) => {
        switch (event.event) {
          case "chat_init": {
            const newChatId = event.data.chat_id as string
            if (!activeChatIdRef.current && newChatId) {
              activeChatIdRef.current = newChatId
              // Update URL without triggering React Router re-render
              window.history.replaceState(null, "", `/org/${orgId}/chat/${newChatId}`)
            }
            break
          }
          case "status":
            setStatusMessage(event.data.message as string)
            break
          case "answer_chunk": {
            const chunk = event.data.chunk as string
            setStatusMessage(null)
            // Create assistant message on first chunk, append on subsequent
            if (!assistantMsgIdRef.current) {
              const msgId = crypto.randomUUID()
              assistantMsgIdRef.current = msgId
              setMessages((prev) => [
                ...prev,
                { id: msgId, role: "assistant", content: chunk },
              ])
            } else {
              const id = assistantMsgIdRef.current
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === id ? { ...m, content: m.content + chunk } : m,
                ),
              )
            }
            break
          }
          case "answer_replace": {
            // Output guardrail modified the answer — replace it
            const replaced = event.data.content as string
            setStatusMessage(null)
            if (assistantMsgIdRef.current) {
              const id = assistantMsgIdRef.current
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === id ? { ...m, content: replaced } : m,
                ),
              )
            } else {
              const msgId = crypto.randomUUID()
              assistantMsgIdRef.current = msgId
              setMessages((prev) => [
                ...prev,
                { id: msgId, role: "assistant", content: replaced },
              ])
            }
            break
          }
          case "answer_done":
            break
          case "title": {
            const t = event.data.title as string | undefined
            if (t) setChatTitle(t)
            break
          }
          case "complete": {
            const title = event.data.title as string | undefined
            if (title) setChatTitle(title)
            break
          }
          case "error":
            toast.error((event.data.message as string) || "An error occurred")
            setStatusMessage(null)
            break
        }
      },
      // onError
      (errMsg) => {
        toast.error(errMsg)
        setIsStreaming(false)
        setStatusMessage(null)
      },
      // onDone
      () => {
        setIsStreaming(false)
        setStatusMessage(null)
        assistantMsgIdRef.current = null
        // Sync React Router with the URL set via replaceState during streaming
        if (activeChatIdRef.current && !chatId) {
          navigate(`/org/${orgId}/chat/${activeChatIdRef.current}`, { replace: true })
        }
      },
    )

    abortRef.current = controller
  }, [input, orgId, isStreaming, navigate])

  function handleStop() {
    abortRef.current?.abort()
    setIsStreaming(false)
    setStatusMessage(null)
  }

  const showEmptyState = !chatId && messages.length === 0 && !isStreaming

  return (
    <PageTransition>
      <TooltipProvider delayDuration={0}>
        <SidebarProvider>
          <AppSidebar />
          <SidebarInset className="flex h-dvh flex-col overflow-hidden">
            <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
              <SidebarTrigger />
              <h1 className="text-sm font-medium">
                {chatTitle || orgName}
              </h1>
            </header>
            <div className="flex min-h-0 flex-1 flex-col">
              <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto px-4 py-6"
              >
                {loadingChat ? (
                  <div className="flex h-full items-center justify-center">
                    <div className="text-muted-foreground text-sm">
                      Loading conversation...
                    </div>
                  </div>
                ) : showEmptyState ? (
                  <div className="flex h-full items-center justify-center">
                    <div className="flex flex-col items-center gap-4">
                      <div className="bg-muted flex size-16 items-center justify-center rounded-2xl">
                        <BrainCircuit className="text-muted-foreground size-8" />
                      </div>
                      <div className="text-center">
                        <h2 className="text-lg font-semibold">
                          Start a Conversation
                        </h2>
                        <p className="text-muted-foreground mt-1 max-w-sm text-sm">
                          Ask questions about your knowledge base and get
                          AI-powered answers grounded in your data.
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="mx-auto flex max-w-3xl flex-col gap-4">
                    {messages.map((msg) => (
                      <ChatMessage
                        key={msg.id}
                        role={msg.role}
                        content={msg.content}
                      />
                    ))}
                    {statusMessage && <ChatStatus message={statusMessage} />}
                  </div>
                )}
              </div>
              <div className="shrink-0 border-t px-4 pb-4 pt-3">
                <div className="mx-auto max-w-3xl">
                  <ChatInput
                    value={input}
                    onChange={setInput}
                    onSend={handleSend}
                    onStop={handleStop}
                    disabled={loadingChat}
                    isStreaming={isStreaming}
                  />
                </div>
              </div>
            </div>
          </SidebarInset>
        </SidebarProvider>
      </TooltipProvider>
    </PageTransition>
  )
}
