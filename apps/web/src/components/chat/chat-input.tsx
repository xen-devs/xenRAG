import { type KeyboardEvent, useRef, useEffect } from "react"
import { ArrowUp, Square } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onStop?: () => void
  disabled?: boolean
  isStreaming?: boolean
}

export function ChatInput({
  value,
  onChange,
  onSend,
  onStop,
  disabled,
  isStreaming,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [value])

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (!disabled && value.trim()) {
        onSend()
      }
    }
  }

  return (
    <div className="border-border bg-background/80 flex items-end gap-2 rounded-2xl border p-2 backdrop-blur-sm">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about your data..."
        disabled={disabled}
        className="placeholder:text-muted-foreground max-h-[200px] min-h-[36px] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none disabled:opacity-50"
      />
      {isStreaming ? (
        <Button
          size="icon"
          variant="destructive"
          className="size-8 shrink-0 rounded-xl"
          onClick={onStop}
        >
          <Square className="size-3.5" />
        </Button>
      ) : (
        <Button
          size="icon"
          className="size-8 shrink-0 rounded-xl"
          disabled={disabled || !value.trim()}
          onClick={onSend}
        >
          <ArrowUp className="size-4" />
        </Button>
      )}
    </div>
  )
}