import { Outlet, useNavigate, useParams } from "react-router-dom"
import { AppSidebar } from "@/components/app-sidebar"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"
import { SettingsNav } from "./settings-nav"
import { SettingsTabBar } from "./settings-tab-bar"

export function SettingsLayout() {
  const navigate = useNavigate()
  const { orgId } = useParams()

  return (
    <TooltipProvider delayDuration={0}>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset className="flex h-screen flex-col overflow-hidden">
          <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
            <SidebarTrigger />
            <button
              onClick={() => navigate(`/org/${orgId}/chat`)}
              className="text-muted-foreground hover:text-foreground text-sm transition-colors"
            >
              Back to Chat
            </button>
            <span className="text-muted-foreground text-sm">/</span>
            <h1 className="text-sm font-medium">Settings</h1>
          </header>

          <div className="shrink-0 border-b md:hidden">
            <SettingsTabBar />
          </div>

          <div className="flex min-h-0 flex-1">
            <aside className="hidden w-64 shrink-0 overflow-y-auto border-r md:block">
              <div className="p-4">
                <SettingsNav />
              </div>
            </aside>

            <main className="flex-1 overflow-y-auto p-4 sm:p-6">
              <div className="mx-auto max-w-3xl">
                <Outlet />
              </div>
            </main>
          </div>
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  )
}