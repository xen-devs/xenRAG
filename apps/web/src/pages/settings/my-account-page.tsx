import { useParams } from "react-router-dom"
import { Mail, User, Building2, Shield } from "lucide-react"
import { useAuth } from "@/contexts/auth-context"

export function MyAccountPage() {
  const { user } = useAuth()
  const { orgId } = useParams()

  const currentOrg = user?.orgs.find((o) => o.id === orgId)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold">My Account</h2>
        <p className="text-muted-foreground text-sm">
          View your profile details and organization membership.
        </p>
      </div>

      <section className="space-y-4 rounded-lg border p-5">
        <h3 className="text-sm font-semibold">Profile</h3>
        <dl className="divide-y">
          <div className="flex items-center justify-between py-3">
            <dt className="text-muted-foreground flex items-center gap-2 text-sm">
              <User className="h-4 w-4" />
              Name
            </dt>
            <dd className="text-sm font-medium">{user?.name || "-"}</dd>
          </div>
          <div className="flex items-center justify-between py-3">
            <dt className="text-muted-foreground flex items-center gap-2 text-sm">
              <Mail className="h-4 w-4" />
              Email
            </dt>
            <dd className="text-sm font-medium">{user?.email || "-"}</dd>
          </div>
        </dl>
      </section>

      {currentOrg && (
        <section className="space-y-4 rounded-lg border p-5">
          <h3 className="text-sm font-semibold">Current Organization</h3>
          <dl className="divide-y">
            <div className="flex items-center justify-between py-3">
              <dt className="text-muted-foreground flex items-center gap-2 text-sm">
                <Building2 className="h-4 w-4" />
                Organization
              </dt>
              <dd className="text-sm font-medium">{currentOrg.name}</dd>
            </div>
            <div className="flex items-center justify-between py-3">
              <dt className="text-muted-foreground flex items-center gap-2 text-sm">
                <Shield className="h-4 w-4" />
                Role
              </dt>
              <dd>
                <span className="bg-primary/10 text-primary inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium capitalize">
                  {currentOrg.role}
                </span>
              </dd>
            </div>
          </dl>
        </section>
      )}

      {user && user.orgs.length > 1 && (
        <section className="space-y-4 rounded-lg border p-5">
          <h3 className="text-sm font-semibold">All Organizations</h3>
          <div className="rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
                    Name
                  </th>
                  <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
                    Role
                  </th>
                </tr>
              </thead>
              <tbody>
                {user.orgs.map((org) => (
                  <tr key={org.id} className="border-b last:border-b-0">
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{org.name}</span>
                        {org.id === orgId && (
                          <span className="bg-muted text-muted-foreground rounded-full px-2 py-0.5 text-[10px]">
                            current
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="text-muted-foreground px-3 py-2 capitalize">
                      {org.role}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}