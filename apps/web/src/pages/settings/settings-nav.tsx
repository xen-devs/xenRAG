import { NavLink, useParams } from "react-router-dom"
import { BookOpen, UserCircle, type LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

export interface NavItem {
  label: string
  to: string
  icon: LucideIcon
}

export const NAV_ITEMS: NavItem[] = [
  {
    label: "My Account",
    to: "my-account",
    icon: UserCircle,
  },
  {
    label: "Knowledge Base",
    to: "knowledge-base",
    icon: BookOpen,
  },
  // Future sections can be added here:
  // { label: "AI / ML", to: "ai-ml", icon: BrainCircuit },
  // { label: "Integrations", to: "integrations", icon: Plug },
  // { label: "Members", to: "members", icon: Users },
]

interface SettingsNavProps {
  onNavigate?: () => void
}

export function SettingsNav({ onNavigate }: SettingsNavProps) {
  const { orgId } = useParams()

  return (
    <nav className="flex flex-col gap-1">
      <p className="text-muted-foreground mb-2 px-3 text-xs font-medium uppercase tracking-wider">
        Settings
      </p>
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={`/org/${orgId}/settings/${item.to}`}
          onClick={onNavigate}
          className={({ isActive }) =>
            cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
              isActive
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/50 hover:text-accent-foreground",
            )
          }
        >
          <item.icon className="h-4 w-4 shrink-0" />
          {item.label}
        </NavLink>
      ))}
    </nav>
  )
}