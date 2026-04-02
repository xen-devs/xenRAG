import type { ReactNode } from "react"
import { Navigate, useLocation } from "react-router-dom"
import { useAuth } from "@/contexts/auth-context"
import { Skeleton } from "@/components/ui/skeleton"

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isLoading, isAuthenticated, user } = useAuth()
  const location = useLocation()

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-32" />
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/auth" replace />
  }

  // If user has no orgs and isn't already on onboarding, redirect there
  if (user && user.orgs.length === 0 && location.pathname !== "/onboarding") {
    return <Navigate to="/onboarding" replace />
  }

  return children
}
