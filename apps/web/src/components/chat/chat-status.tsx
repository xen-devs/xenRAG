import { Loader2 } from "lucide-react"

interface ChatStatusProps {
  message: string
}

export function ChatStatus({ message }: ChatStatusProps) {
  return (
    <div className="flex items-center gap-2 px-11">
      <Loader2 className="text-muted-foreground size-3.5 animate-spin" />
      <span className="text-muted-foreground text-xs">{message}</span>
    </div>
  )
}