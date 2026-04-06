import { type FormEvent, useEffect, useRef, useState } from "react"
import {
  BrainCircuit,
  Check,
  ChevronsUpDown,
  ChevronUp,
  MoreHorizontal,
  Loader2,
  LogOut,
  MessageSquarePlus,
  Monitor,
  Moon,
  Plus,
  Settings2,
  Sun,
  Building2,
  Search,
} from "lucide-react"
import { useNavigate, useParams, useLocation } from "react-router-dom"
import toast from "react-hot-toast"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { useSidebar } from "@/components/ui/sidebar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuPortal,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Kbd } from "@/components/ui/kbd"
import { ConfirmModal } from "@/components/ui/confirm-modal"
import { CommandPalette } from "@/components/command-palette"
import { useAuth } from "@/contexts/auth-context"
import { useTheme } from "@/components/theme-provider"
import { createOrg } from "@/lib/auth-api"
import { deleteChat, listChats, renameChat, type ChatSummary } from "@/lib/chat-api"

function getInitials(name?: string) {
  if (!name) return "U"
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2)
}

export function AppSidebar() {
  const { user, logout, refreshUser } = useAuth()
  const { theme, setTheme } = useTheme()
  const navigate = useNavigate()
  const location = useLocation()
  const { orgId } = useParams()

  const settingsUrl = orgId ? `/org/${orgId}/settings` : "#"
  const [isCreateOrgOpen, setIsCreateOrgOpen] = useState(false)
  const [newOrgName, setNewOrgName] = useState("")
  const [newProductName, setNewProductName] = useState("")
  const [newDescription, setNewDescription] = useState("")
  const [createOrgLoading, setCreateOrgLoading] = useState(false)
  const [chats, setChats] = useState<ChatSummary[]>([])
  const [editingChatId, setEditingChatId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState("")
  const editingInputRef = useRef<HTMLInputElement>(null)
  const [deleteTarget, setDeleteTarget] = useState<ChatSummary | null>(null)
  const [deleteLoading, setDeleteLoading] = useState(false)
  const { state: sidebarState } = useSidebar()
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false)

  const currentOrg = user?.orgs.find((o) => o.id === orgId)
  const orgName = currentOrg?.name ?? "Organization"

  const activeChatId = location.pathname.match(
    /\/org\/[^/]+\/chat\/([^/]+)/,
  )?.[1]

  useEffect(() => {
    if (!orgId) return
    listChats(orgId)
      .then(setChats)
      .catch(() => {})
  }, [orgId, location.pathname])

  useEffect(() => {
    if (editingChatId && editingInputRef.current) {
      setTimeout(() => editingInputRef.current?.select(), 50)
    }
  }, [editingChatId])

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "/" && !e.ctrlKey && !e.metaKey) {
        const target = e.target as HTMLElement
        if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return
        e.preventDefault()
        setIsCommandPaletteOpen(true)
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault()
        setIsCommandPaletteOpen(true)
      }
    }
    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [])

  function handleLogout() {
    logout()
    navigate("/auth")
  }

  function handleOrgSwitch(id: string) {
    navigate(`/org/${id}/chat`)
  }

  async function handleCreateOrg(e: FormEvent) {
    e.preventDefault()
    if (newOrgName.trim().length < 2) {
      toast.error("Organization name must be at least 2 characters")
      return
    }
    if (!newProductName.trim()) {
      toast.error("Product name is required")
      return
    }
    if (!newDescription.trim()) {
      toast.error("Description is required")
      return
    }

    setCreateOrgLoading(true)

    try {
      const org = await createOrg({
        name: newOrgName.trim(),
        product_name: newProductName.trim(),
        description: newDescription.trim(),
      })
      await refreshUser()
      setIsCreateOrgOpen(false)
      setNewOrgName("")
      setNewProductName("")
      setNewDescription("")
      toast.success("Organization created!")
      navigate(`/org/${org.id}/chat`)
    } catch (err: unknown) {
      if (
        err &&
        typeof err === "object" &&
        "response" in err &&
        err.response &&
        typeof err.response === "object" &&
        "data" in err.response
      ) {
        const data = err.response.data as { detail?: string }
        toast.error(data.detail ?? "Failed to create organization")
      } else {
        toast.error("Failed to create organization. Please try again.")
      }
    } finally {
      setCreateOrgLoading(false)
    }
  }

  return (
    <Sidebar collapsible="icon">
      {/* Org Switcher Header */}
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                >
                  <div className="bg-sidebar-primary text-sidebar-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg">
                    <BrainCircuit className="size-4" />
                  </div>
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-medium">{orgName}</span>
                    <span className="text-muted-foreground truncate text-xs">
                      xenRAG
                    </span>
                  </div>
                  <ChevronsUpDown className="ml-auto h-4 w-4" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="min-w-56 rounded-lg"
                align="start"
                side="right"
                sideOffset={4}
              >
                {user?.orgs.map((org) => (
                  <DropdownMenuItem
                    key={org.id}
                    className="cursor-pointer gap-2"
                    onClick={() => handleOrgSwitch(org.id)}
                  >
                    <div className="flex size-6 items-center justify-center rounded-md border">
                      <Building2 className="size-3.5 shrink-0" />
                    </div>
                    <span className="truncate">{org.name}</span>
                    {org.id === orgId && (
                      <Check className="ml-auto h-4 w-4" />
                    )}
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="cursor-pointer gap-2"
                  onClick={() => setIsCreateOrgOpen(true)}
                >
                  <div className="flex size-6 items-center justify-center rounded-md border bg-transparent">
                    <Plus className="size-4" />
                  </div>
                  <span className="text-muted-foreground font-medium">
                    Add Organization
                  </span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      {/* Create Org Dialog */}
      <Dialog
        open={isCreateOrgOpen}
        onOpenChange={(open) => {
          setIsCreateOrgOpen(open)
          if (!open) {
            setNewOrgName("")
            setNewProductName("")
            setNewDescription("")
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Organization</DialogTitle>
            <DialogDescription>
              Organizations help you separate workspaces for different teams or
              projects.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleCreateOrg} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <label htmlFor="org-name" className="text-sm font-medium">
                Organization Name
              </label>
              <Input
                id="org-name"
                type="text"
                placeholder="e.g., Acme Corporation"
                required
                value={newOrgName}
                onChange={(e) => setNewOrgName(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-2">
              <label htmlFor="product-name" className="text-sm font-medium">
                Product Name
              </label>
              <Input
                id="product-name"
                type="text"
                placeholder="e.g., Fire Stick"
                autoComplete="off"
                required
                value={newProductName}
                onChange={(e) => setNewProductName(e.target.value)}
              />
            </div>
            <div className="flex flex-col gap-2">
              <label htmlFor="description" className="text-sm font-medium">
                Description
              </label>
              <textarea
                id="description"
                className="flex min-h-[80px] w-full rounded-lg border border-input bg-transparent px-2.5 py-2 text-sm text-foreground placeholder:text-muted-foreground outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
                placeholder="Describe your product and feedback context..."
                autoComplete="off"
                required
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                maxLength={1000}
              />
            </div>
            <Button type="submit" className="w-full" disabled={createOrgLoading}>
              {createOrgLoading && (
                <Loader2 className="mr-2 size-4 animate-spin" />
              )}
              Create Organization
            </Button>
          </form>
        </DialogContent>
      </Dialog>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  tooltip="New Chat"
                  className="cursor-pointer"
                  onClick={() => {
                    if (orgId) navigate(`/org/${orgId}/chat`)
                  }}
                >
                  <MessageSquarePlus />
                  <span>New Chat</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  tooltip="Settings"
                  className={
                    settingsUrl !== "#" ? "cursor-pointer" : "cursor-default"
                  }
                  onClick={() => {
                    if (settingsUrl !== "#") navigate(settingsUrl)
                  }}
                >
                  <Settings2 />
                  <span>Settings</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  tooltip="Search chats"
                  className="cursor-pointer"
                  onClick={() => setIsCommandPaletteOpen(true)}
                >
                  <Search />
                  <span>Search</span>
                  <Kbd className="ml-auto text-xs font-sans">⌘K</Kbd>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {chats.length > 0 && sidebarState === "expanded" && (
          <SidebarGroup className="flex-1 overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
            <SidebarGroupLabel>Chats</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {chats.map((chat) => (
                  <SidebarMenuItem key={chat.id}>
                    <div className="group/chat relative flex w-full items-center">
                      {editingChatId === chat.id ? (
                        <input
                          ref={editingInputRef}
                          className="truncate flex-1 bg-transparent border border-ring rounded px-2 py-1 text-sm outline-none"
                          value={editingTitle}
                          onClick={(e) => e.stopPropagation()}
                          onMouseDown={(e) => e.stopPropagation()}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          onBlur={async () => {
                            const trimmed = editingTitle.trim()
                            if (trimmed && trimmed !== (chat.title || "")) {
                              try {
                                const updated = await renameChat(orgId!, chat.id, trimmed)
                                setChats((prev) =>
                                  prev.map((c) => c.id === chat.id ? updated : c)
                                )
                                toast.success("Chat renamed")
                              } catch {
                                toast.error("Failed to rename chat")
                              }
                            }
                            setEditingChatId(null)
                          }}
                          onKeyDown={async (e) => {
                            if (e.key === "Enter") {
                              e.preventDefault()
                              e.stopPropagation()
                              editingInputRef.current?.blur()
                            } else if (e.key === "Escape") {
                              setEditingChatId(null)
                            }
                          }}
                        />
                      ) : (
                        <SidebarMenuButton
                          tooltip={chat.title || "Untitled"}
                          isActive={activeChatId === chat.id}
                          className="cursor-pointer w-full pr-8"
                          onClick={() =>
                            navigate(`/org/${orgId}/chat/${chat.id}`)
                          }
                        >
                          <span className="truncate block">
                            {chat.title || "Untitled"}
                          </span>
                        </SidebarMenuButton>
                      )}

                      {editingChatId !== chat.id && sidebarState === "expanded" && (
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <button
                              className="absolute right-1.5 opacity-0 group-hover/chat:opacity-100 p-1 rounded hover:bg-accent"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <MoreHorizontal className="h-4 w-4" />
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" side="right" sideOffset={4}>
                            <DropdownMenuItem
                              className="cursor-pointer"
                              onClick={() => {
                                setEditingTitle(chat.title || "")
                                setEditingChatId(chat.id)
                              }}
                            >
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="cursor-pointer text-red-500"
                              onClick={() => setDeleteTarget(chat)}
                            >
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      )}
                    </div>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
      </SidebarContent>

      {/* User Footer with Theme & Logout */}
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                >
                  <Avatar className="h-8 w-8">
                    <AvatarFallback className="bg-primary/10 text-primary text-xs font-medium">
                      {getInitials(user?.name)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-medium">{user?.name}</span>
                    <span className="text-muted-foreground truncate text-xs">
                      {user?.email}
                    </span>
                  </div>
                  <ChevronUp className="ml-auto h-4 w-4" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="min-w-56 rounded-lg"
                side="right"
                align="end"
                sideOffset={4}
              >
                <div className="flex items-center gap-2 px-2 py-1.5 text-left text-sm">
                  <Avatar className="h-8 w-8">
                    <AvatarFallback className="bg-primary/10 text-primary text-xs font-medium">
                      {getInitials(user?.name)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-medium">{user?.name}</span>
                    <span className="text-muted-foreground truncate text-xs">
                      {user?.email}
                    </span>
                  </div>
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger className="cursor-pointer gap-2">
                    {theme === "dark" ? (
                      <Moon className="h-4 w-4" />
                    ) : theme === "light" ? (
                      <Sun className="h-4 w-4" />
                    ) : (
                      <Monitor className="h-4 w-4" />
                    )}
                    <span>Theme</span>
                  </DropdownMenuSubTrigger>
                  <DropdownMenuPortal>
                    <DropdownMenuSubContent>
                      <DropdownMenuItem
                        className="cursor-pointer gap-2"
                        onClick={() => setTheme("system")}
                      >
                        <Monitor className="h-4 w-4" />
                        <span>System</span>
                        {theme === "system" && (
                          <Check className="ml-auto h-4 w-4" />
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        className="cursor-pointer gap-2"
                        onClick={() => setTheme("light")}
                      >
                        <Sun className="h-4 w-4" />
                        <span>Light</span>
                        {theme === "light" && (
                          <Check className="ml-auto h-4 w-4" />
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        className="cursor-pointer gap-2"
                        onClick={() => setTheme("dark")}
                      >
                        <Moon className="h-4 w-4" />
                        <span>Dark</span>
                        {theme === "dark" && (
                          <Check className="ml-auto h-4 w-4" />
                        )}
                      </DropdownMenuItem>
                    </DropdownMenuSubContent>
                  </DropdownMenuPortal>
                </DropdownMenuSub>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="cursor-pointer gap-2 text-red-500"
                  onClick={handleLogout}
                >
                  <LogOut className="h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <ConfirmModal
        open={!!deleteTarget}
        onOpenChange={(open) => { if (!open) setDeleteTarget(null) }}
        title="Delete chat?"
        description={`"${deleteTarget?.title || "Untitled"}" will be permanently deleted.`}
        confirmLabel="Delete"
        variant="destructive"
        isLoading={deleteLoading}
        onConfirm={async () => {
          if (!deleteTarget) return
          setDeleteLoading(true)
          try {
            await deleteChat(orgId!, deleteTarget.id)
            setChats((prev) => prev.filter((c) => c.id !== deleteTarget.id))
            if (activeChatId === deleteTarget.id) {
              navigate(`/org/${orgId}/chat`)
            }
            setDeleteTarget(null)
            toast.success("Chat deleted")
          } catch {
            toast.error("Failed to delete chat")
          } finally {
            setDeleteLoading(false)
          }
        }}
      />

      <CommandPalette
        open={isCommandPaletteOpen}
        onOpenChange={setIsCommandPaletteOpen}
        orgId={orgId!}
      />
    </Sidebar>
  )
}
