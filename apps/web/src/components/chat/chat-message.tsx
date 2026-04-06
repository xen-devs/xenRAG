import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeRaw from "rehype-raw"
import { cn } from "@/lib/utils"

interface ChatMessageProps {
  role: "user" | "assistant"
  content: string
}

function Table({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-3 w-full overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">{children}</table>
    </div>
  )
}

function TableHead({ children }: { children: React.ReactNode }) {
  return (
    <thead className="bg-muted/60 border-b border-border">{children}</thead>
  )
}

function TableRow({ children }: { children: React.ReactNode }) {
  return <tr className="border-b border-border last:border-b-0">{children}</tr>
}

function TableTh({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      {children}
    </th>
  )
}

function TableTd({ children }: { children: React.ReactNode }) {
  return <td className="px-3 py-2.5">{children}</td>
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  const isUser = role === "user"

  return (
    <div className={cn("flex", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "max-w-[80%] bg-primary text-primary-foreground rounded-tr-sm"
            : "max-w-full bg-muted rounded-tl-sm sm:max-w-[80%]",
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <div className="space-y-2">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeRaw]}
              components={{
                p: ({ children }) => <p className="my-2 leading-7 first:mt-0 last:mb-0">{children}</p>,
                h1: ({ children }) => <h1 className="my-3 text-xl font-semibold leading-tight">{children}</h1>,
                h2: ({ children }) => <h2 className="my-3 text-lg font-semibold leading-tight">{children}</h2>,
                h3: ({ children }) => <h3 className="my-2.5 text-base font-semibold leading-tight">{children}</h3>,
                ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5 marker:text-muted-foreground">{children}</ul>,
                ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5 marker:text-muted-foreground">{children}</ol>,
                li: ({ children }) => <li className="pl-1 leading-7">{children}</li>,
                blockquote: ({ children }) => (
                  <blockquote className="my-2 border-l-4 border-muted-foreground/30 pl-4 italic text-muted-foreground">
                    {children}
                  </blockquote>
                ),
                a: ({ children, href }) => (
                  <a href={href} className="text-primary underline underline-offset-2 hover:text-primary/80" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
                hr: () => <hr className="my-3 border-border" />,
                table: ({ children }) => <Table>{children}</Table>,
                thead: ({ children }) => <TableHead>{children}</TableHead>,
                tbody: ({ children }) => <tbody>{children}</tbody>,
                tr: ({ children }) => <TableRow>{children}</TableRow>,
                th: ({ children }) => <TableTh>{children}</TableTh>,
                td: ({ children }) => <TableTd>{children}</TableTd>,
                pre: ({ children }) => (
                  <pre className="my-2 overflow-x-auto rounded-lg bg-muted/30 p-3 font-mono text-[13px]">
                    {children}
                  </pre>
                ),
                code: ({ className, children, ...props }) => {
                  const isInline = !className
                  if (isInline) {
                    return (
                      <code
                        className="rounded bg-muted/40 px-1.5 py-0.5 font-mono text-[12px]"
                        {...props}
                      >
                        {children}
                      </code>
                    )
                  }
                  return (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  )
                },
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}
