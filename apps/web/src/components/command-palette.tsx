import { useEffect, useRef, useState, useMemo } from "react"
import { Search, MessageSquare } from "lucide-react"
import Fuse from "fuse.js"
import { Kbd } from "@/components/ui/kbd"
import { listChats, type ChatSummary } from "@/lib/chat-api"
import { useNavigate } from "react-router-dom"

interface CommandPaletteProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  orgId: string
}

export function CommandPalette({ open, onOpenChange, orgId }: CommandPaletteProps) {
  const [query, setQuery] = useState("")
  const [allChats, setAllChats] = useState<ChatSummary[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  // Fuzzy search setup
  const fuse = useMemo(() => {
    return new Fuse(allChats, {
      keys: ["title"],
      threshold: 0.4,
      ignoreLocation: true,
      includeScore: true,
    })
  }, [allChats])

  // Results based on fuzzy search
  const results = useMemo(() => {
    if (!query.trim()) return []
    return fuse.search(query).map((result) => result.item)
  }, [query, fuse])

  // Fetch all chats when opened
  useEffect(() => {
    if (!open) {
      setQuery("")
      setAllChats([])
      setSelectedIndex(0)
    } else {
      setIsLoading(true)
      setTimeout(() => inputRef.current?.focus(), 50)
      listChats(orgId)
        .then(setAllChats)
        .catch(() => setAllChats([]))
        .finally(() => setIsLoading(false))
    }
  }, [open, orgId])

  useEffect(() => {
    setSelectedIndex(0)
  }, [results])

  function handleSelect(chat: ChatSummary) {
    navigate(`/org/${orgId}/chat/${chat.id}`)
    onOpenChange(false)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault()
      setSelectedIndex((i) => (i < results.length - 1 ? i + 1 : 0))
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      setSelectedIndex((i) => (i > 0 ? i - 1 : results.length - 1))
    } else if (e.key === "Enter" && results[selectedIndex]) {
      e.preventDefault()
      handleSelect(results[selectedIndex])
    } else if (e.key === "Escape") {
      onOpenChange(false)
    }
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      <div
        className="fixed inset-0 bg-black/50"
        onClick={() => onOpenChange(false)}
      />
      <div
        className="relative z-50 w-full max-w-lg overflow-hidden rounded-xl border bg-background shadow-2xl"
        onKeyDown={handleKeyDown}
      >
        <div className="flex items-center gap-3 border-b px-4 py-3">
          <Search className="h-5 w-5 shrink-0 text-muted-foreground" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search chats..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          <Kbd className="text-xs">Esc</Kbd>
        </div>

        <div className="max-h-[300px] overflow-y-auto p-2">
          {isLoading ? (
            <div className="px-3 py-6 text-center text-sm text-muted-foreground">
              Loading chats...
            </div>
          ) : query && results.length === 0 ? (
            <div className="px-3 py-6 text-center text-sm text-muted-foreground">
              No chats found
            </div>
          ) : results.length > 0 ? (
            <div className="space-y-1">
              <div className="px-3 py-1.5 text-xs font-medium text-muted-foreground">
                Chats
              </div>
              {results.map((chat, index) => (
                <button
                  key={chat.id}
                  onClick={() => handleSelect(chat)}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                    index === selectedIndex
                      ? "bg-accent text-accent-foreground"
                      : "hover:bg-accent/50"
                  }`}
                >
                  <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{chat.title || "Untitled"}</span>
                </button>
              ))}
            </div>
          ) : (
            <div className="px-3 py-6 text-center text-sm text-muted-foreground">
              Start typing to search chats
            </div>
          )}
        </div>
      </div>
    </div>
  )
}