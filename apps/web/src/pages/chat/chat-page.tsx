import { useParams } from "react-router-dom"
import { AppSidebar } from "@/components/app-sidebar"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"
import { PageTransition } from "@/components/page-transition"
import { useAuth } from "@/contexts/auth-context"

export function ChatPage() {
  const { orgId, chatId } = useParams()
  const { user } = useAuth()

  const orgName =
    user?.orgs.find((o) => o.id === orgId)?.name ?? "Organization"

  return (
    <PageTransition>
      <TooltipProvider delayDuration={0}>
        <SidebarProvider>
          <AppSidebar />
          <SidebarInset>
            <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
              <SidebarTrigger />
              <h1 className="text-sm font-medium">{orgName}</h1>
            </header>
            <main className="flex flex-1 items-center justify-center p-6">
              <div className="text-center">
                {chatId ? (
                  <p className="text-muted-foreground text-sm">
                    Chat: {chatId}
                  </p>
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <h2 className="text-xl font-semibold">
                      Start a Conversation
                    </h2>
                    <p className="text-muted-foreground max-w-sm text-sm">
                      Ask questions about your data and get explainable,
                      AI-powered answers.
                    </p>
                  </div>
                )}
              </div>
            </main>
          </SidebarInset>
        </SidebarProvider>
      </TooltipProvider>
    </PageTransition>
  )
}
