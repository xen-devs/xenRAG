import { AppSidebar } from "@/components/app-sidebar"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"

export function App() {
  return (
    <TooltipProvider delayDuration={0}>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
          <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
            <SidebarTrigger />
            <h1 className="text-sm font-medium">xenRAG Frontend</h1>
          </header>
          <main className="flex-1 p-6">
            <div className="mx-auto flex max-w-3xl flex-col gap-3">
              <h2 className="text-xl font-semibold">Sidebar is ready</h2>
              <p className="text-sm text-muted-foreground">
                Use the trigger in the header to collapse or expand the sidebar.
              </p>
              <p className="text-sm text-muted-foreground">
                Press <kbd>Ctrl</kbd> + <kbd>b</kbd> to toggle quickly.
              </p>
            </div>
          </main>
        </SidebarInset>
      </SidebarProvider>
    </TooltipProvider>
  )
}

export default App
