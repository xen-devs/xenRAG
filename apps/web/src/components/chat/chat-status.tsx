import { Loader2 } from "lucide-react"

interface ChatStatusProps {
  message: string
}

export function ChatStatus({ message }: ChatStatusProps) {
  return (
    <div className="flex">
      <div className="bg-muted w-full max-w-full rounded-2xl rounded-tl-sm px-4 py-3 sm:max-w-[80%]">
        <div className="flex items-center gap-2">
          <Loader2 className="text-muted-foreground size-3.5 animate-spin" />
          <span className="text-muted-foreground text-xs">{message}</span>
        </div>
      </div>
    </div>
  )
}