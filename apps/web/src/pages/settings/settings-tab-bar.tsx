import { NavLink, useParams } from "react-router-dom"
import { cn } from "@/lib/utils"
import { NAV_ITEMS } from "./settings-nav"

export function SettingsTabBar() {
  const { orgId } = useParams()

  return (
    <nav className="scrollbar-none flex overflow-x-auto px-4">
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={`/org/${orgId}/settings/${item.to}`}
          className={({ isActive }) =>
            cn(
              "flex shrink-0 items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium whitespace-nowrap transition-colors",
              isActive
                ? "border-primary text-foreground"
                : "text-muted-foreground hover:text-foreground border-transparent",
            )
          }
        >
          <item.icon className="h-4 w-4" />
          {item.label}
        </NavLink>
      ))}
    </nav>
  )
}